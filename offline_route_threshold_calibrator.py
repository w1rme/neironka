import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from image_loading import safe_read_image
from neural_bot import (
    ALLOWED_TRANSITIONS,
    AuctionBotController,
    CLASS_ID_TO_FOLDER_NAME,
    MIN_STABLE_COUNT,
    NeuralRecognizer,
    PROFILE_BYPASS_CONFIDENCE,
    STATE_CONFIDENCE_THRESHOLDS,
    STATE_HISTORY_SIZE,
)


PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
BOT_PATH = PROJECT_DIR / "neural_bot.py"
REPORT_PATH = PROJECT_DIR / "offline_route_threshold_report.json"

THRESHOLD_MARKER_START = "# THRESHOLD_CONFIG_START"
THRESHOLD_MARKER_END = "# THRESHOLD_CONFIG_END"

ROUTES = (
    ("no_auctions_return", [0, 1, 2, 0]),
    ("failed_buy_return", [0, 1, 7, 3, 4, 6, 0]),
    ("success_buy", [0, 1, 7, 3, 4, 5]),
)

MIN_THRESHOLD = 0.85
MAX_THRESHOLD = 0.995
THRESHOLD_STEP = 0.01
SEARCH_PASSES = 2
MAX_FRAMES_PER_STAGE = 4
DEFAULT_EPISODES_PER_ROUTE = 200
TRUE_SEARCH_QUANTILE = 0.10
TRUE_PREFERRED_QUANTILE = 0.20
TRUE_CONF_MARGIN_DOWN = 0.02
FALSE_CONF_MARGIN_UP = 0.01
HARD_MIN_SEARCH_THRESHOLD = 0.75
SCORE_EPSILON = 1e-9
SAFETY_SCORE_TRADEOFF = 6


def should_bypass_profile_reject(state_id, confidence):
    min_confidence = PROFILE_BYPASS_CONFIDENCE.get(state_id)
    if min_confidence is None:
        return False
    return float(confidence) >= float(min_confidence)


def format_threshold_block(thresholds):
    lines = [THRESHOLD_MARKER_START, "STATE_CONFIDENCE_THRESHOLDS = {"]
    for state_id in sorted(CLASS_ID_TO_FOLDER_NAME):
        lines.append(f"    {state_id}: {float(thresholds[state_id]):.3f},")
    lines.append("}")
    lines.append(THRESHOLD_MARKER_END)
    return "\n".join(lines)


def update_neural_bot_thresholds(thresholds):
    source = BOT_PATH.read_text(encoding="utf-8")
    start = source.find(THRESHOLD_MARKER_START)
    end = source.find(THRESHOLD_MARKER_END)

    if start == -1 or end == -1 or end < start:
        raise RuntimeError("Threshold config markers not found in neural_bot.py")

    end += len(THRESHOLD_MARKER_END)
    replacement = format_threshold_block(thresholds)
    updated = source[:start] + replacement + source[end:]
    BOT_PATH.write_text(updated, encoding="utf-8")


def build_records():
    recognizer = NeuralRecognizer()
    records_by_state = defaultdict(list)

    print("[*] Building cached predictions from dataset...")

    for true_state, folder_name in CLASS_ID_TO_FOLDER_NAME.items():
        class_dir = DATA_DIR / folder_name
        image_paths = sorted(class_dir.glob("*.png"))
        print(f"[*] {folder_name}: {len(image_paths)} files")

        for image_path in image_paths:
            image = safe_read_image(image_path)
            if image is None:
                continue

            prediction = recognizer.predict(image)
            records_by_state[true_state].append(
                {
                    "path": str(image_path),
                    "true_state": true_state,
                    "prediction": {
                        "state_id": int(prediction["state_id"]),
                        "folder_name": prediction["folder_name"],
                        "display_name": prediction["display_name"],
                        "confidence": float(prediction["confidence"]),
                        "crop_mode": prediction["crop_mode"],
                        "in_profile": bool(prediction["in_profile"]),
                    },
                }
            )

    for state_id in sorted(CLASS_ID_TO_FOLDER_NAME):
        if not records_by_state[state_id]:
            raise RuntimeError(
                f"No valid cached predictions for class {state_id} "
                f"({CLASS_ID_TO_FOLDER_NAME[state_id]})"
            )

    return dict(records_by_state)


def build_confidence_ranges(records_by_state):
    ranges = {}

    for state_id in sorted(CLASS_ID_TO_FOLDER_NAME):
        true_confidences = []
        false_confidences = []

        for true_state, record_list in records_by_state.items():
            for record in record_list:
                prediction = record["prediction"]

                if int(prediction["state_id"]) != state_id:
                    continue

                confidence = float(prediction["confidence"])
                in_profile = bool(prediction["in_profile"])
                accepted_like = in_profile or should_bypass_profile_reject(
                    state_id,
                    confidence,
                )

                if not accepted_like:
                    continue

                if true_state == state_id:
                    true_confidences.append(confidence)
                else:
                    false_confidences.append(confidence)

        if not true_confidences:
            raise RuntimeError(
                f"No accepted-like true confidences collected for class {state_id}"
            )

        true_min = min(true_confidences)
        true_q05 = float(np.quantile(true_confidences, 0.05))
        true_q10 = float(np.quantile(true_confidences, TRUE_SEARCH_QUANTILE))
        true_q20 = float(np.quantile(true_confidences, TRUE_PREFERRED_QUANTILE))
        false_max = max(false_confidences) if false_confidences else None

        lower_bound = max(
            HARD_MIN_SEARCH_THRESHOLD,
            true_q10 - TRUE_CONF_MARGIN_DOWN,
        )
        if false_max is not None:
            lower_bound = max(lower_bound, false_max + FALSE_CONF_MARGIN_UP)

        lower_bound = min(lower_bound, MAX_THRESHOLD)
        preferred_threshold = min(MAX_THRESHOLD, max(lower_bound, true_q20))

        ranges[state_id] = {
            "true_min": float(true_min),
            "true_q05": float(true_q05),
            "true_q10": float(true_q10),
            "true_q20": float(true_q20),
            "false_max": None if false_max is None else float(false_max),
            "search_min": round(lower_bound, 3),
            "preferred_threshold": round(preferred_threshold, 3),
            "search_max": MAX_THRESHOLD,
        }

    return ranges


def evaluate_acceptance(prediction, thresholds, state_history):
    state_id = int(prediction["state_id"])
    confidence = float(prediction["confidence"])
    required_confidence = float(thresholds[state_id])

    if confidence < required_confidence:
        return None, "low_conf"

    if not prediction["in_profile"]:
        if not should_bypass_profile_reject(state_id, confidence):
            return None, "profile_reject"

    state_history.append(state_id)
    if len(state_history) > STATE_HISTORY_SIZE:
        state_history.pop(0)

    if state_history.count(state_id) < MIN_STABLE_COUNT:
        return None, "unstable"

    return state_id, "accepted"


def simulate_route_episode(route, records_by_state, thresholds, rng):
    controller = AuctionBotController()
    state_history = []
    stage_misses = 0
    wrong_accepts = 0
    accepted_true = 0
    now = 0.0

    for true_state in route:
        record = records_by_state[true_state][
            int(rng.integers(0, len(records_by_state[true_state])))
        ]
        prediction = record["prediction"]
        stage_done = False

        for _ in range(MAX_FRAMES_PER_STAGE):
            accepted_state, reason = evaluate_acceptance(
                prediction,
                thresholds,
                state_history,
            )
            now += 0.1

            if accepted_state is None:
                continue

            if accepted_state != true_state:
                wrong_accepts += 1
                return {
                    "success": False,
                    "stage_misses": stage_misses,
                    "wrong_accepts": wrong_accepts,
                    "accepted_true": accepted_true,
                    "failed_true_state": true_state,
                    "failed_reason": f"accepted_wrong_{accepted_state}",
                    "failed_path": record["path"],
                }

            controller.update_observation(accepted_state, now, preconfirmed=True)
            action = controller.handle_state(accepted_state, now)
            if action is not None:
                controller.notify_action_executed(now)
                accepted_true += 1
                stage_done = True
                break

        if not stage_done:
            stage_misses += 1
            return {
                "success": False,
                "stage_misses": stage_misses,
                "wrong_accepts": wrong_accepts,
                "accepted_true": accepted_true,
                "failed_true_state": true_state,
                "failed_reason": "no_action",
                "failed_path": record["path"],
            }

    return {
        "success": True,
        "stage_misses": stage_misses,
        "wrong_accepts": wrong_accepts,
        "accepted_true": accepted_true,
        "failed_true_state": None,
        "failed_reason": None,
        "failed_path": None,
    }


def evaluate_isolated_acceptance(records_by_state, thresholds):
    per_state = {}
    total_tp = 0
    total_fp = 0
    total_low_conf = 0

    for true_state in sorted(CLASS_ID_TO_FOLDER_NAME):
        tp = 0
        fp = 0
        low_conf = 0
        total = 0

        for record_list_state, record_list in records_by_state.items():
            for record in record_list:
                state_history = []
                accepted_state = None
                reason = None

                for _ in range(MIN_STABLE_COUNT):
                    accepted_state, reason = evaluate_acceptance(
                        record["prediction"],
                        thresholds,
                        state_history,
                    )

                if record_list_state == true_state:
                    total += 1
                    if accepted_state == true_state:
                        tp += 1
                    elif reason == "low_conf":
                        low_conf += 1
                elif accepted_state == true_state:
                    fp += 1

        per_state[true_state] = {
            "tp": tp,
            "fp": fp,
            "low_conf": low_conf,
            "total": total,
        }
        total_tp += tp
        total_fp += fp
        total_low_conf += low_conf

    return per_state, total_tp, total_fp, total_low_conf


def evaluate_thresholds(records_by_state, thresholds, episodes_per_route, seed):
    rng = np.random.default_rng(seed)
    route_results = {}
    total_route_successes = 0
    total_wrong_accepts = 0
    total_stage_misses = 0

    for route_name, route in ROUTES:
        successes = 0
        wrong_accepts = 0
        stage_misses = 0
        failure_examples = []

        for _ in range(episodes_per_route):
            result = simulate_route_episode(route, records_by_state, thresholds, rng)
            if result["success"]:
                successes += 1
            else:
                wrong_accepts += int(result["wrong_accepts"])
                stage_misses += int(result["stage_misses"])
                if len(failure_examples) < 3:
                    failure_examples.append(
                        {
                            "true_state": result["failed_true_state"],
                            "reason": result["failed_reason"],
                            "path": result["failed_path"],
                        }
                    )

        route_results[route_name] = {
            "episodes": episodes_per_route,
            "successes": successes,
            "success_rate": successes / max(1, episodes_per_route),
            "wrong_accepts": wrong_accepts,
            "stage_misses": stage_misses,
            "fail_examples": failure_examples,
        }
        total_route_successes += successes
        total_wrong_accepts += wrong_accepts
        total_stage_misses += stage_misses

    isolated_metrics, total_tp, total_fp, total_low_conf = evaluate_isolated_acceptance(
        records_by_state,
        thresholds,
    )

    score = (
        total_route_successes * 1000
        + total_tp * 2
        - total_fp * 20
        - total_wrong_accepts * 25
        - total_stage_misses * 15
        - total_low_conf * 2
    )

    return {
        "score": score,
        "route_results": route_results,
        "isolated_metrics": isolated_metrics,
        "totals": {
            "route_successes": total_route_successes,
            "route_episodes": episodes_per_route * len(ROUTES),
            "wrong_accepts": total_wrong_accepts,
            "stage_misses": total_stage_misses,
            "isolated_tp": total_tp,
            "isolated_fp": total_fp,
            "isolated_low_conf": total_low_conf,
        },
    }


def iter_candidate_thresholds(search_min, search_max):
    current = float(search_min)
    while current <= MAX_THRESHOLD + 1e-9:
        if current > float(search_max) + 1e-9:
            break
        yield round(current, 3)
        current += THRESHOLD_STEP


def candidate_is_safe_bias_upgrade(
    current_report,
    candidate_report,
    current_threshold,
    candidate_threshold,
    preferred_threshold,
):
    if candidate_threshold <= current_threshold:
        return False

    if candidate_threshold > preferred_threshold:
        return False

    current_totals = current_report["totals"]
    candidate_totals = candidate_report["totals"]

    if candidate_totals["route_successes"] < current_totals["route_successes"]:
        return False
    if candidate_totals["wrong_accepts"] > current_totals["wrong_accepts"]:
        return False
    if candidate_totals["stage_misses"] > current_totals["stage_misses"]:
        return False
    if candidate_totals["isolated_fp"] > current_totals["isolated_fp"]:
        return False

    score_drop = float(current_report["score"]) - float(candidate_report["score"])
    return score_drop <= SAFETY_SCORE_TRADEOFF


def optimize_thresholds(records_by_state, confidence_ranges, episodes_per_route, seed):
    thresholds = {
        state_id: float(STATE_CONFIDENCE_THRESHOLDS[state_id])
        for state_id in sorted(CLASS_ID_TO_FOLDER_NAME)
    }

    for state_id, stats in confidence_ranges.items():
        thresholds[state_id] = max(
            float(thresholds[state_id]),
            float(stats["search_min"]),
        )

    best_report = evaluate_thresholds(records_by_state, thresholds, episodes_per_route, seed)

    print(
        f"[*] Initial score={best_report['score']} "
        f"route_successes={best_report['totals']['route_successes']}/"
        f"{best_report['totals']['route_episodes']}"
    )

    for search_pass in range(SEARCH_PASSES):
        print(f"[*] Search pass {search_pass + 1}/{SEARCH_PASSES}")

        for state_id in sorted(CLASS_ID_TO_FOLDER_NAME):
            state_best_threshold = float(thresholds[state_id])
            state_best_report = best_report
            search_min = float(confidence_ranges[state_id]["search_min"])
            preferred_threshold = float(confidence_ranges[state_id]["preferred_threshold"])
            search_max = float(confidence_ranges[state_id]["search_max"])

            for candidate in iter_candidate_thresholds(search_min, search_max):
                trial = deepcopy(thresholds)
                trial[state_id] = float(candidate)
                report = evaluate_thresholds(records_by_state, trial, episodes_per_route, seed)

                score_delta = float(report["score"]) - float(state_best_report["score"])
                same_score = abs(score_delta) <= SCORE_EPSILON

                if score_delta > SCORE_EPSILON:
                    state_best_threshold = float(candidate)
                    state_best_report = report
                elif same_score and float(candidate) > state_best_threshold:
                    state_best_threshold = float(candidate)
                    state_best_report = report
                elif candidate_is_safe_bias_upgrade(
                    state_best_report,
                    report,
                    state_best_threshold,
                    float(candidate),
                    preferred_threshold,
                ):
                    state_best_threshold = float(candidate)
                    state_best_report = report

            thresholds[state_id] = state_best_threshold
            best_report = state_best_report
            print(
                f"    state {state_id} {CLASS_ID_TO_FOLDER_NAME[state_id]} -> "
                f"{thresholds[state_id]:.3f} score={best_report['score']} "
                f"(search_min={search_min:.3f}, preferred={preferred_threshold:.3f})"
            )

    return thresholds, best_report


def build_report_payload(thresholds, report, confidence_ranges, episodes_per_route, seed):
    return {
        "seed": seed,
        "episodes_per_route": episodes_per_route,
        "thresholds": {
            str(state_id): float(thresholds[state_id])
            for state_id in sorted(thresholds)
        },
        "confidence_ranges": {
            str(state_id): stats
            for state_id, stats in confidence_ranges.items()
        },
        "score": float(report["score"]),
        "totals": report["totals"],
        "route_results": report["route_results"],
        "isolated_metrics": {
            str(state_id): metrics
            for state_id, metrics in report["isolated_metrics"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Offline route-based threshold calibration for neural_bot.py",
    )
    parser.add_argument("--episodes-per-route", type=int, default=DEFAULT_EPISODES_PER_ROUTE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records_by_state = build_records()
    confidence_ranges = build_confidence_ranges(records_by_state)
    print("[*] Confidence-based search ranges:")
    for state_id in sorted(CLASS_ID_TO_FOLDER_NAME):
        stats = confidence_ranges[state_id]
        false_max_text = "none" if stats["false_max"] is None else f"{stats['false_max']:.4f}"
        print(
            f"    state {state_id} {CLASS_ID_TO_FOLDER_NAME[state_id]} "
            f"true_min={stats['true_min']:.4f} true_q05={stats['true_q05']:.4f} "
            f"true_q10={stats['true_q10']:.4f} true_q20={stats['true_q20']:.4f} "
            f"false_max={false_max_text} search_min={stats['search_min']:.3f} "
            f"preferred={stats['preferred_threshold']:.3f}"
        )
    thresholds, report = optimize_thresholds(
        records_by_state,
        confidence_ranges,
        episodes_per_route=args.episodes_per_route,
        seed=args.seed,
    )

    payload = build_report_payload(
        thresholds,
        report,
        confidence_ranges,
        episodes_per_route=args.episodes_per_route,
        seed=args.seed,
    )
    REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[*] Final thresholds:")
    for state_id in sorted(CLASS_ID_TO_FOLDER_NAME):
        print(
            f"    {state_id} {CLASS_ID_TO_FOLDER_NAME[state_id]} = "
            f"{thresholds[state_id]:.3f}"
        )

    print(
        f"[*] Final route successes: {report['totals']['route_successes']}/"
        f"{report['totals']['route_episodes']}"
    )
    print(f"[*] Report saved to {REPORT_PATH.name}")

    if not args.dry_run:
        update_neural_bot_thresholds(thresholds)
        print("[*] Updated thresholds in neural_bot.py")


if __name__ == "__main__":
    main()
