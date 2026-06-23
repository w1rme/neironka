import argparse
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

import dxcam
import vgamepad as vg

from neural_bot import (
    ALLOWED_TRANSITIONS,
    CLASS_ID_TO_FOLDER_NAME,
    NeuralRecognizer,
    STATE_CONFIDENCE_THRESHOLDS,
    STATE_PRE_ACTION_DELAYS,
)


PROJECT_DIR = Path(__file__).parent
BOT_PATH = PROJECT_DIR / "neural_bot.py"
PYTHON_EXE = "python"

ACTION_MARKER_START = "# ACTION_DELAY_CONFIG_START"
ACTION_MARKER_END = "# ACTION_DELAY_CONFIG_END"

POLL_INTERVAL = 0.05
RESTART_LIMIT = 20
REQUIRED_SUCCESSFUL_ACTIONS = 12
DEFAULT_UNCHANGED_SCREEN_TIMEOUT = 2.0
STATE_UNCHANGED_SCREEN_TIMEOUTS = {
    4: 5.0,
}
DELAY_STEP_UP = 0.03
MIN_DELAY = 0.0
MAX_DELAY = 0.80
STATE_HISTORY_SIZE = 4
MIN_STABLE_COUNT = 2
RAW_SAME_SCREEN_MARGIN = 0.05
ACTION_DEDUP_WINDOW = 0.75
PRELAUNCH_BUTTON_HOLD = 0.02
PRELAUNCH_B_DELAY = 0.10

WATCHED_ACTION_STATES = {1, 3, 4, 7}
EXPECTED_ACTIONS = {
    1: "A",
    3: "DOWN + A",
    4: "A",
    7: "Y",
}

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
ACTION_LINE_RE = re.compile(r"^ACTION:\s*(.*)$")
STATUS_LINE_RE = re.compile(r"^STATUS:\s*(.*)$")
FLOW_LINE_RE = re.compile(r"^FLOW:\s*(.*)$")
SPECIAL_PRELAUNCH_FLOW = "route 0 -> 1 -> 7 -> 3 -> 4 | next 5, 6"


def format_action_delay_block(delays):
    lines = [
        ACTION_MARKER_START,
        "DEFAULT_PRE_ACTION_DELAY = 0.0",
        "STATE_PRE_ACTION_DELAYS = {",
    ]

    for state_id in sorted(CLASS_ID_TO_FOLDER_NAME):
        delay = float(delays.get(state_id, 0.0))
        lines.append(f"    {state_id}: {delay:.2f},")

    lines.append("}")
    lines.append(ACTION_MARKER_END)
    return "\n".join(lines)


def update_neural_bot_delays(delays):
    source = BOT_PATH.read_text(encoding="utf-8")
    start = source.find(ACTION_MARKER_START)
    end = source.find(ACTION_MARKER_END)

    if start == -1 or end == -1 or end < start:
        raise RuntimeError("Action delay config markers not found in neural_bot.py")

    end += len(ACTION_MARKER_END)
    replacement = format_action_delay_block(delays)
    updated = source[:start] + replacement + source[end:]
    BOT_PATH.write_text(updated, encoding="utf-8")


class ScreenObserver:
    def __init__(self):
        self.recognizer = NeuralRecognizer()
        self.camera = dxcam.create(output_idx=0, output_color="BGR")
        if self.camera is None:
            raise RuntimeError("dxcam.create(output_idx=0) returned None")

        if hasattr(self.camera, "start"):
            try:
                self.camera.start(target_fps=30, video_mode=True)
            except TypeError:
                self.camera.start(target_fps=30)

        self.state_history = []
        self.last_state = None

    def stop(self):
        if hasattr(self.camera, "stop"):
            try:
                self.camera.stop()
            except Exception:
                pass

    def screenshot(self):
        if hasattr(self.camera, "get_latest_frame"):
            return self.camera.get_latest_frame()
        return self.camera.grab()

    def poll(self):
        frame = self.screenshot()
        if frame is None:
            return None, None

        prediction = self.recognizer.predict(frame)
        state_id = prediction["state_id"]
        confidence = float(prediction["confidence"])
        required_confidence = float(STATE_CONFIDENCE_THRESHOLDS[state_id])

        accepted_state = None
        if confidence >= required_confidence and prediction["in_profile"]:
            self.state_history.append(state_id)
            if len(self.state_history) > STATE_HISTORY_SIZE:
                self.state_history.pop(0)
            if self.state_history.count(state_id) >= MIN_STABLE_COUNT:
                accepted_state = state_id

        return accepted_state, prediction


class BotProcess:
    def __init__(self):
        self.process = None
        self.stdout_queue = queue.Queue()
        self.reader_thread = None

    def start(self):
        self.process = subprocess.Popen(
            [PYTHON_EXE, "-u", str(BOT_PATH), "--bot-child"],
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

    def _reader_loop(self):
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.stdout_queue.put(line.rstrip())

    def read_lines(self):
        lines = []
        while True:
            try:
                lines.append(self.stdout_queue.get_nowait())
            except queue.Empty:
                return lines

    def poll(self):
        if self.process is None:
            return None
        return self.process.poll()

    def stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3.0)
        self.process = None


def tap_button(gamepad, button, hold=PRELAUNCH_BUTTON_HOLD):
    gamepad.press_button(button=button)
    gamepad.update()
    time.sleep(hold)
    gamepad.release_button(button=button)
    gamepad.update()


def run_prelaunch_combo(gamepad):
    print("[CAL] prelaunch combo: A then 100ms B", flush=True)
    tap_button(gamepad, vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    time.sleep(PRELAUNCH_B_DELAY)
    tap_button(gamepad, vg.XUSB_BUTTON.XUSB_GAMEPAD_B)


def run_special_prelaunch_combo(gamepad):
    print("[CAL] prelaunch combo: A -> A -> B -> B (100ms gaps)", flush=True)
    tap_button(gamepad, vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    time.sleep(PRELAUNCH_B_DELAY)
    tap_button(gamepad, vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    time.sleep(PRELAUNCH_B_DELAY)
    tap_button(gamepad, vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    time.sleep(PRELAUNCH_B_DELAY)
    tap_button(gamepad, vg.XUSB_BUTTON.XUSB_GAMEPAD_B)


def get_unchanged_screen_timeout(state_id):
    return float(STATE_UNCHANGED_SCREEN_TIMEOUTS.get(state_id, DEFAULT_UNCHANGED_SCREEN_TIMEOUT))


def parse_action_label(line):
    cleaned = ANSI_ESCAPE_RE.sub("", line).strip()
    match = ACTION_LINE_RE.match(cleaned)
    if not match:
        return None
    return match.group(1).strip()


def parse_status_label(line):
    cleaned = ANSI_ESCAPE_RE.sub("", line).strip()
    match = STATUS_LINE_RE.match(cleaned)
    if not match:
        return None
    return match.group(1).strip()


def parse_flow_label(line):
    cleaned = ANSI_ESCAPE_RE.sub("", line).strip()
    match = FLOW_LINE_RE.match(cleaned)
    if not match:
        return None
    return match.group(1).strip()


def build_initial_delays():
    delays = {state_id: 0.0 for state_id in sorted(CLASS_ID_TO_FOLDER_NAME)}
    for state_id, value in STATE_PRE_ACTION_DELAYS.items():
        delays[int(state_id)] = float(value)
    return delays


def increase_delay(delays, state_id):
    old_value = float(delays.get(state_id, 0.0))
    new_value = min(MAX_DELAY, old_value + DELAY_STEP_UP)
    delays[state_id] = max(MIN_DELAY, new_value)
    return old_value, float(delays[state_id])


def is_screen_still_same(raw_prediction, accepted_state, source_state):
    if accepted_state == source_state:
        return True

    if raw_prediction is None:
        return False

    raw_state = int(raw_prediction["state_id"])
    raw_conf = float(raw_prediction["confidence"])
    raw_in_profile = bool(raw_prediction["in_profile"])

    if raw_state != source_state or not raw_in_profile:
        return False

    required = float(STATE_CONFIDENCE_THRESHOLDS[source_state])
    return raw_conf >= max(MIN_THRESHOLD_FALLBACK(required), required - RAW_SAME_SCREEN_MARGIN)


def MIN_THRESHOLD_FALLBACK(required):
    return max(0.60, required - 0.10)


def main():
    parser = argparse.ArgumentParser(
        description="Live button-delay calibrator for neural_bot.py",
    )
    parser.add_argument("--required-successes", type=int, default=REQUIRED_SUCCESSFUL_ACTIONS)
    args = parser.parse_args()

    delays = build_initial_delays()
    update_neural_bot_delays(delays)
    observer = ScreenObserver()
    gamepad = vg.VX360Gamepad()
    successful_actions = 0
    restart_count = 0
    next_prelaunch_mode = "default"

    print("[*] Live action delay calibrator started")
    print(f"[*] target successful actions in a row: {args.required_successes}")

    try:
        while restart_count < RESTART_LIMIT:
            if next_prelaunch_mode == "special":
                run_special_prelaunch_combo(gamepad)
            else:
                run_prelaunch_combo(gamepad)
            bot = BotProcess()
            bot.start()
            restart_count += 1
            pending_action = None
            last_action_signature = None
            last_flow_label = None
            next_prelaunch_mode = "default"

            print(f"[*] bot run {restart_count}/{RESTART_LIMIT} started")

            try:
                while True:
                    accepted_state, prediction = observer.poll()
                    now = time.perf_counter()

                    for line in bot.read_lines():
                        cleaned = ANSI_ESCAPE_RE.sub("", line).strip()
                        if cleaned:
                            print(cleaned, flush=True)

                        status_label = parse_status_label(line)
                        if status_label is not None and "success route completed" in status_label.lower():
                            print("[CAL] success route completed", flush=True)

                        flow_label = parse_flow_label(line)
                        if flow_label is not None:
                            last_flow_label = flow_label

                        action_label = parse_action_label(line)
                        if action_label is None:
                            continue

                        if accepted_state is None or accepted_state not in WATCHED_ACTION_STATES:
                            continue

                        if EXPECTED_ACTIONS.get(accepted_state) != action_label:
                            continue

                        action_signature = (accepted_state, action_label)
                        if pending_action is not None:
                            same_pending = (
                                pending_action["source_state"] == accepted_state
                                and pending_action["source_action"] == action_label
                            )
                            if same_pending:
                                continue

                        if last_action_signature is not None:
                            last_state, last_action, last_started_at = last_action_signature
                            if (
                                last_state == accepted_state
                                and last_action == action_label
                                and now - last_started_at < ACTION_DEDUP_WINDOW
                            ):
                                continue

                        pending_action = {
                            "source_state": accepted_state,
                            "source_action": action_label,
                            "started_at": now,
                            "unchanged_timeout": get_unchanged_screen_timeout(accepted_state),
                            "deadline": now + get_unchanged_screen_timeout(accepted_state),
                            "expected_states": set(ALLOWED_TRANSITIONS.get(accepted_state, set())),
                            "best_expected_raw": {},
                        }
                        last_action_signature = (accepted_state, action_label, now)
                        print(
                            f"[CAL] watching action {action_label} from state {accepted_state} "
                            f"delay={delays.get(accepted_state, 0.0):.2f}s "
                            f"timeout={pending_action['unchanged_timeout']:.1f}s",
                            flush=True,
                        )

                    if pending_action is not None:
                        source_state = pending_action["source_state"]
                        expected_states = pending_action["expected_states"]
                        raw_state = None if prediction is None else int(prediction["state_id"])
                        raw_conf = 0.0 if prediction is None else float(prediction["confidence"])

                        if raw_state in expected_states:
                            best_conf = pending_action["best_expected_raw"].get(raw_state, 0.0)
                            if raw_conf > best_conf:
                                pending_action["best_expected_raw"][raw_state] = raw_conf

                        if accepted_state is not None and accepted_state in expected_states:
                            successful_actions += 1
                            print(
                                f"[CAL] action ok: {source_state} -> {accepted_state} "
                                f"stable={successful_actions}/{args.required_successes}",
                                flush=True,
                            )
                            pending_action = None
                            last_action_signature = None

                            if successful_actions >= args.required_successes:
                                print("[CAL] target reached, keeping current delays", flush=True)
                                bot.stop()
                                return

                        elif now >= pending_action["deadline"]:
                            if pending_action["best_expected_raw"]:
                                target_state, target_conf = max(
                                    pending_action["best_expected_raw"].items(),
                                    key=lambda item: item[1],
                                )
                                print(
                                    f"[CAL] skip delay increase for state {source_state}: "
                                    f"expected raw state {target_state} was seen with conf={target_conf:.4f}",
                                    flush=True,
                                )
                                pending_action = None
                                last_action_signature = None
                                continue

                            if is_screen_still_same(prediction, accepted_state, source_state):
                                old_value, new_value = increase_delay(delays, source_state)
                                print(
                                    f"[CAL] screen stayed on state {source_state} for >{pending_action['unchanged_timeout']:.1f}s "
                                    f"after {pending_action['source_action']}: "
                                    f"delay {old_value:.2f}s -> {new_value:.2f}s",
                                    flush=True,
                                )
                                update_neural_bot_delays(delays)
                                successful_actions = 0
                                if last_flow_label == SPECIAL_PRELAUNCH_FLOW:
                                    next_prelaunch_mode = "special"
                                bot.stop()
                                pending_action = None
                                last_action_signature = None
                                break

                            print(
                                f"[CAL] timeout observed after state {source_state}, "
                                f"but unchanged-screen evidence was not strong enough",
                                flush=True,
                            )
                            pending_action = None
                            last_action_signature = None

                    poll_result = bot.poll()
                    if poll_result is not None:
                        if last_flow_label == SPECIAL_PRELAUNCH_FLOW:
                            next_prelaunch_mode = "special"
                        print(f"[CAL] bot exited with code {poll_result}, restarting...", flush=True)
                        break

                    time.sleep(POLL_INTERVAL)

            finally:
                bot.stop()

        print("[CAL] restart limit reached, last delays saved to neural_bot.py", flush=True)

    finally:
        observer.stop()


if __name__ == "__main__":
    main()
