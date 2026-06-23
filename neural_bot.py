import atexit
import ctypes
import datetime
import json
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import vgamepad as vg

from image_loading import safe_read_image
from model_config import MODEL_INPUT_SIZE

try:
    import dxcam
    DXCAM_AVAILABLE = True
except ImportError:
    DXCAM_AVAILABLE = False


BUTTON_HOLD = 0.02
DOWN_HOLD = 0.02
DOWN_TO_A_DELAY = 0.04
A_CONFIRM_HOLD = 0.02
A_FINAL_HOLD = 0.02
CONFIRM_BUY_SPAM_COUNT = 4
CONFIRM_BUY_SPAM_GAP = 0.05
FAILED_A_TO_B_DELAY = 0.30
# ACTION_DELAY_CONFIG_START
DEFAULT_PRE_ACTION_DELAY = 0.0
STATE_PRE_ACTION_DELAYS = {
    0: 0.40,
    1: 0.40,
    2: 0.15,
    3: 0.20,
    4: 0.25,
    5: 0.10,
    6: 0.20,
    7: 0.10,
}
# ACTION_DELAY_CONFIG_END
STATE_ACTION_CONFIRM_COUNTS = {}
RETURN_B_HOLD = 0.03
RETURN_B_INTERVAL = 1.00
RETURN_POLL_INTERVAL = 0.03
CAPTURE_IDLE_SLEEP = 0.001
INFER_IDLE_SLEEP = 0.001
RETURN_ZERO_CONFIRM_COUNT = 3
RETURN_ZERO_SOFT_THRESHOLD = 0.92
LOOP_SLEEP = 0.01
CONSOLE_MAIN_WIDTH = 72
CONSOLE_LINE_WIDTH = 104
STATE_HISTORY_SIZE = 4
MIN_STABLE_COUNT = 2
MAX_RETURN_B_PRESSES = 2
NO_DETECTION_RETURN_TIMEOUT = 2.0
SEARCH_FORM_RETRY_TIMEOUT = 2.0
PROFILE_SAMPLE_LIMIT = 48
PROFILE_ZSCORE_LIMIT = 6.0
PROFILE_MEAN_ZSCORE_LIMIT = 3.5
PROFILE_STD_FLOOR = np.array([6.0, 0.01, 8.0], dtype=np.float32)
PROFILE_BYPASS_CONFIDENCE = {
    1: 0.950,
}
PROFILE_FAST_SKIP_MARGIN = 0.05
CUDA_RESTART_EXIT_CODE = 86
CUDA_RESTART_DELAY = 3.0
CUDA_RESTART_LIMIT = 20
APP_ROOT = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", APP_ROOT))
TELEGRAM_FILE = APP_ROOT / "250k_telegram.txt"
BALANCE_FILE = APP_ROOT / "250k_balance.txt"
RUNTIME_CONFIG_FILE = APP_ROOT / "bot_runtime_config.json"
DEFAULT_PURCHASE_COST = 250_000
DEFAULT_STARTING_BALANCE = 20_000_000
PURCHASE_PRICE_SLIDER_MIN = 20_000
PURCHASE_PRICE_SLIDER_MAX = 20_000_000
PURCHASE_PRICE_SLIDER_STEP = 1_000

# TIMING_CONFIG_START
STATE_TIMINGS = {
    0: {"wait_timeout": 2.50},
    1: {"wait_timeout": 1.50},
    2: {"wait_timeout": 15.00},
    3: {"wait_timeout": 1.50},
    4: {"wait_timeout": 5.00},
    5: {"wait_timeout": 6.00},
    6: {"wait_timeout": 5.00},
    7: {"wait_timeout": 15.00},
}
# TIMING_CONFIG_END

# THRESHOLD_CONFIG_START
STATE_CONFIDENCE_THRESHOLDS = {
    0: 0.900,
    1: 0.895,
    2: 0.900,
    3: 0.900,
    4: 0.900,
    5: 0.960,
    6: 0.900,
    7: 0.900,
}
# THRESHOLD_CONFIG_END

CLASS_ID_TO_FOLDER_NAME = {
    0: "0_auction_house",
    1: "1_search_form",
    2: "2_no_auctions",
    3: "3_buy_menu",
    4: "4_confirm_buy",
    5: "5_buy_success",
    6: "6_buy_failed",
    7: "7_my_auction",
}

CLASS_ID_TO_DISPLAY_NAME = {
    0: "auction_house",
    1: "search_form",
    2: "no_auctions",
    3: "buy_menu",
    4: "confirm_buy",
    5: "buy_success",
    6: "buy_failed",
    7: "my_auction",
}

CLASS_FOLDER_NAME_TO_ID = {
    value: key for key, value in CLASS_ID_TO_FOLDER_NAME.items()
}

ALLOWED_TRANSITIONS = {
    None: {0},
    0: {1},
    1: {2, 7},
    2: {0},
    3: {4},
    4: {5, 6},
    5: set(),
    6: {0},
    7: {3},
}

TIMEOUT_RETURN_ROUTES = (
    (0, 1, 7),
)

BUTTONS = {
    "A": vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
    "B": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
    "Y": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
    "DOWN": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
}


@dataclass
class BotRuntimeConfig:
    stop_after_purchase: bool = True
    purchase_cost: int = DEFAULT_PURCHASE_COST
    starting_balance: int = DEFAULT_STARTING_BALANCE

    @classmethod
    def from_dict(cls, payload):
        payload = payload or {}
        stop_after_purchase = bool(payload.get("stop_after_purchase", True))
        purchase_cost = sanitize_positive_int(
            payload.get("purchase_cost", DEFAULT_PURCHASE_COST),
            DEFAULT_PURCHASE_COST,
        )
        starting_balance = sanitize_non_negative_int(
            payload.get("starting_balance", DEFAULT_STARTING_BALANCE),
            DEFAULT_STARTING_BALANCE,
        )
        return cls(
            stop_after_purchase=stop_after_purchase,
            purchase_cost=purchase_cost,
            starting_balance=starting_balance,
        )

    def to_dict(self):
        return {
            "stop_after_purchase": bool(self.stop_after_purchase),
            "purchase_cost": int(self.purchase_cost),
            "starting_balance": int(self.starting_balance),
        }


def sanitize_positive_int(value, default):
    try:
        value = int(str(value).replace(" ", "").replace("_", ""))
    except (TypeError, ValueError):
        return int(default)
    return int(value) if value > 0 else int(default)


def sanitize_non_negative_int(value, default):
    try:
        value = int(str(value).replace(" ", "").replace("_", ""))
    except (TypeError, ValueError):
        return int(default)
    return int(value) if value >= 0 else int(default)


def read_balance(balance_path, default_value=DEFAULT_STARTING_BALANCE):
    if not balance_path.exists():
        return int(default_value)

    text = balance_path.read_text(encoding="utf-8").strip()
    if not text:
        return int(default_value)

    try:
        value = int(text.replace(" ", "").replace("_", ""))
    except ValueError:
        return int(default_value)

    return max(0, value)


def write_balance(balance_path, value):
    balance_path.write_text(str(max(0, int(value))), encoding="utf-8")


def load_runtime_config(config_path):
    if not config_path.exists():
        return BotRuntimeConfig()

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return BotRuntimeConfig()

    return BotRuntimeConfig.from_dict(payload)


def save_runtime_config(config_path, runtime_config):
    config_path.write_text(
        json.dumps(runtime_config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_runtime_config_arg(argv):
    for arg in argv:
        if arg.startswith("--runtime-config="):
            return Path(arg.split("=", 1)[1]).resolve()
    return RUNTIME_CONFIG_FILE


def resolve_resource_path(relative_name):
    app_candidate = APP_ROOT / relative_name
    if app_candidate.exists():
        return app_candidate
    return RESOURCE_ROOT / relative_name


def ensure_runtime_file(file_path):
    bundled_path = RESOURCE_ROOT / file_path.name
    if file_path.exists() or not bundled_path.exists():
        return
    try:
        file_path.write_text(bundled_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass


def get_state_wait_timeout(timings, state_id, default=2.0):
    state_timing = timings.get(state_id, {})

    if "wait_timeout" in state_timing:
        return float(state_timing["wait_timeout"])

    if "min_visible" in state_timing:
        return float(state_timing["min_visible"])

    return float(default)


def get_state_action_delay(state_id):
    return float(STATE_PRE_ACTION_DELAYS.get(state_id, DEFAULT_PRE_ACTION_DELAY))


def get_action_delay(action):
    return float(action.get("delay_before", DEFAULT_PRE_ACTION_DELAY))


def get_required_confirm_count(state_id):
    return int(STATE_ACTION_CONFIRM_COUNTS.get(state_id, MIN_STABLE_COUNT))


def should_bypass_profile_reject(state_id, confidence):
    min_confidence = PROFILE_BYPASS_CONFIDENCE.get(state_id)
    if min_confidence is None:
        return False
    return float(confidence) >= float(min_confidence)


def should_skip_profile_check(state_id, confidence):
    required_confidence = STATE_CONFIDENCE_THRESHOLDS.get(state_id, 1.0)
    return float(confidence) >= float(required_confidence + PROFILE_FAST_SKIP_MARGIN)


def estimate_action_duration(action):
    action_type = action["type"]
    delay = get_action_delay(action)

    if action_type == "tap":
        return delay + float(action.get("hold", BUTTON_HOLD))

    if action_type == "down_a_combo":
        return delay + DOWN_HOLD + DOWN_TO_A_DELAY + BUTTON_HOLD

    if action_type == "tap_burst":
        repeat_count = int(action.get("repeat_count", 1))
        hold = float(action.get("hold", BUTTON_HOLD))
        repeat_gap = float(action.get("repeat_gap", 0.0))
        return delay + repeat_count * hold + max(0, repeat_count - 1) * repeat_gap

    if action_type == "final_success":
        return delay + A_FINAL_HOLD

    if action_type == "failed_and_return":
        return delay + A_FINAL_HOLD + FAILED_A_TO_B_DELAY + RETURN_B_HOLD

    return delay


class CudaRestartRequired(RuntimeError):
    pass


def read_telegram_config(config_path):
    config = {
        "enabled": "0",
        "bot_token": "",
        "chat_id": "",
    }

    if not config_path.exists():
        return config

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()

    return config


class TelegramNotifier:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = read_telegram_config(config_path)

    @property
    def enabled(self):
        return str(self.config.get("enabled", "0")).strip() == "1"

    @property
    def bot_token(self):
        return str(self.config.get("bot_token", "")).strip()

    @property
    def chat_id(self):
        return str(self.config.get("chat_id", "")).strip()

    def can_send(self):
        return self.enabled and bool(self.bot_token) and bool(self.chat_id)

    def send_message(self, text):
        if not self.can_send():
            return False, "telegram disabled or config incomplete"

        payload = urllib.parse.urlencode(
            {
                "chat_id": self.chat_id,
                "text": text,
            }
        ).encode("utf-8")
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        try:
            request = urllib.request.Request(url, data=payload, method="POST")
            with urllib.request.urlopen(request, timeout=10) as response:
                body = response.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            return bool(data.get("ok")), body
        except Exception as exc:
            return False, str(exc)


def is_cuda_runtime_error(exc):
    text = str(exc).lower()
    markers = (
        "cuda error",
        "cudnn",
        "cublas",
        "device-side assert",
        "launch failure",
        "cuda kernel",
    )
    return any(marker in text for marker in markers)


class TransferLearningCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def smart_crop(frame, state_group="full"):
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame passed to smart_crop")
    return frame


class NeuralRecognizer:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_cuda = self.device.type == "cuda"
        self.use_half = self.use_cuda
        self.input_dtype = torch.float16 if self.use_half else torch.float32
        self.target_size = (MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0])

        if self.use_cuda:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        print(f"[+] Neural device: {self.device}")

        self.model = TransferLearningCNN(8).to(self.device)
        if self.use_half:
            self.model = self.model.half()
        self.model = self.model.to(memory_format=torch.channels_last)
        self.model.eval()
        self.classes = [
            CLASS_ID_TO_FOLDER_NAME[idx]
            for idx in range(len(CLASS_ID_TO_FOLDER_NAME))
        ]
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=self.device,
            dtype=self.input_dtype,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=self.device,
            dtype=self.input_dtype,
        ).view(1, 3, 1, 1)

        self.load_weights()
        self.class_profiles = self.build_class_profiles()

    def load_weights(self):
        weights_path = resolve_resource_path("model_weights.pth")

        try:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print("[+] Weights loaded")
        except Exception as exc:
            print(f"[!] Weights load failed: {exc}")

    def preprocess(self, frame):
        frame = np.ascontiguousarray(frame)
        tensor = torch.from_numpy(frame).to(self.device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor[:, [2, 1, 0], :, :]
        tensor = tensor.to(dtype=self.input_dtype)
        tensor = tensor.div_(255.0)
        tensor = F.interpolate(
            tensor,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )
        tensor = tensor.sub_(self.mean).div_(self.std)
        return tensor.contiguous(memory_format=torch.channels_last)

    def predict_probabilities(self, frame):
        with torch.inference_mode():
            tensor = self.preprocess(frame)
            output = self.model(tensor)
            probs = torch.softmax(output.float(), dim=1)
            return probs.squeeze(0).detach().cpu().numpy()

    def extract_feature_vector(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray, 80, 160)

        return np.array(
            [
                float(gray.std()),
                float((edges > 0).mean()),
                float(hsv[:, :, 1].mean()),
            ],
            dtype=np.float32,
        )

    def build_class_profiles(self):
        profiles = {}
        data_dir = resolve_resource_path("data")

        if not data_dir.exists():
            print("[i] data folder not found; appearance profiles disabled")
            return profiles

        for state_id, folder_name in CLASS_ID_TO_FOLDER_NAME.items():
            class_dir = data_dir / folder_name
            feature_vectors = []

            all_paths = sorted(class_dir.glob("*.png"))

            if len(all_paths) > PROFILE_SAMPLE_LIMIT:
                sample_indices = np.linspace(
                    0,
                    len(all_paths) - 1,
                    PROFILE_SAMPLE_LIMIT,
                    dtype=int,
                )
                selected_paths = [all_paths[idx] for idx in sample_indices]
            else:
                selected_paths = all_paths

            for img_path in selected_paths:
                img = safe_read_image(img_path)

                if img is None:
                    continue

                cropped = smart_crop(img, "full")
                feature_vectors.append(self.extract_feature_vector(cropped))

            if not feature_vectors:
                continue

            stacked = np.stack(feature_vectors)
            profiles[state_id] = {
                "mean": stacked.mean(axis=0),
                "std": np.maximum(stacked.std(axis=0), PROFILE_STD_FLOOR),
            }

        print(f"[+] Class appearance profiles: {len(profiles)}/8 loaded")
        return profiles

    def is_prediction_in_profile(self, frame, prediction):
        state_id = prediction["state_id"]
        profile = self.class_profiles.get(state_id)

        if profile is None:
            return True, None

        cropped = smart_crop(frame, prediction["crop_mode"])
        feature_vector = self.extract_feature_vector(cropped)
        zscores = np.abs(
            (feature_vector - profile["mean"]) / profile["std"]
        )

        allowed = (
            float(zscores.max()) <= PROFILE_ZSCORE_LIMIT and
            float(zscores.mean()) <= PROFILE_MEAN_ZSCORE_LIMIT
        )

        return allowed, zscores

    def predict(self, frame):
        full_frame = smart_crop(frame, state_group="full")
        probs = self.predict_probabilities(full_frame)
        state_id = int(np.argmax(probs))
        confidence = float(probs[state_id])
        crop_mode = "full"

        prediction = {
            "state_id": state_id,
            "folder_name": CLASS_ID_TO_FOLDER_NAME[state_id],
            "display_name": CLASS_ID_TO_DISPLAY_NAME[state_id],
            "confidence": confidence,
            "crop_mode": crop_mode,
        }

        if should_skip_profile_check(state_id, confidence):
            in_profile, zscores = True, None
        else:
            in_profile, zscores = self.is_prediction_in_profile(frame, prediction)
        prediction["in_profile"] = in_profile
        prediction["profile_zscores"] = zscores
        return prediction


class AuctionBotController:
    def __init__(self, timings=None):
        self.timings = timings or STATE_TIMINGS
        self.reset()

    def reset(self):
        self.route_history = []
        self.current_state = None
        self.current_state_since = 0.0
        self.current_state_token = 0
        self.current_state_count = 0
        self.last_acted_token = None
        self.return_mode = False
        self.stop_requested = False
        self.expected_states = set()
        self.wait_started_at = None

    def update_observation(self, state_id, now, preconfirmed=False):
        if state_id in self.expected_states:
            self.clear_wait_expectation()

        if state_id != self.current_state:
            self.current_state = state_id
            if preconfirmed:
                # `detect_state()` already enforced stability, so react immediately.
                self.current_state_since = now
                self.current_state_count = MIN_STABLE_COUNT
            else:
                self.current_state_since = now
                self.current_state_count = 1
            self.current_state_token += 1
            return

        if preconfirmed:
            self.current_state_count = max(
                self.current_state_count + 1,
                MIN_STABLE_COUNT,
            )
            return

        self.current_state_count += 1

    def is_state_confirmed(self, state_id, now):
        if state_id != self.current_state:
            return False

        return self.current_state_count >= get_required_confirm_count(state_id)

    def is_transition_allowed(self, state_id):
        # Treat failed-buy screen as a safe terminal screen: even if route tracking
        # was reset after a fallback, seeing state 6 should still trigger A -> return.
        if state_id == 6:
            return True
        previous = self.route_history[-1] if self.route_history else None
        allowed = ALLOWED_TRANSITIONS.get(previous, set())
        return state_id in allowed

    def register_transition(self, state_id):
        if self.route_history and self.route_history[-1] == state_id:
            return
        self.route_history.append(state_id)

    def get_timeout_return_route(self):
        for route in TIMEOUT_RETURN_ROUTES:
            route_len = len(route)
            if self.route_history[-route_len:] == list(route):
                return route
        return None

    def should_handle_state(self, state_id, now):
        if not self.is_state_confirmed(state_id, now):
            return False

        if self.return_mode:
            if self.last_acted_token == self.current_state_token:
                return False
            return state_id in {0, 6}

        if self.last_acted_token == self.current_state_token:
            return False

        return self.is_transition_allowed(state_id)

    def mark_handled(self):
        self.last_acted_token = self.current_state_token

    def clear_wait_expectation(self):
        self.expected_states = set()
        self.wait_started_at = None

    def notify_action_executed(self, executed_at):
        if self.expected_states:
            self.wait_started_at = executed_at

    def start_wait_for_states(self, state_ids, now):
        self.expected_states = set(state_ids)
        self.wait_started_at = now if self.expected_states else None

    def get_wait_timeout(self):
        if not self.expected_states:
            return None

        return max(
            get_state_wait_timeout(self.timings, state_id)
            for state_id in self.expected_states
        )

    def is_wait_timeout_exceeded(self, now):
        if not self.expected_states or self.wait_started_at is None:
            return False

        timeout = self.get_wait_timeout()
        if timeout is None:
            return False

        return now - self.wait_started_at >= timeout

    def handle_state(self, state_id, now):
        if not self.should_handle_state(state_id, now):
            return None

        self.mark_handled()

        if state_id == 0:
            self.return_mode = False
            self.register_transition(0)
            self.start_wait_for_states(ALLOWED_TRANSITIONS[0], now)
            return {
                "type": "tap",
                "button": "A",
                "hold": BUTTON_HOLD,
                "delay_before": get_state_action_delay(0),
            }

        if state_id == 1:
            self.register_transition(1)
            self.start_wait_for_states(ALLOWED_TRANSITIONS[1], now)
            return {
                "type": "tap",
                "button": "A",
                "hold": BUTTON_HOLD,
                "delay_before": get_state_action_delay(1),
            }

        if state_id == 2:
            self.register_transition(2)
            self.return_mode = True
            self.start_wait_for_states(ALLOWED_TRANSITIONS[2], now)
            return {"type": "return_to_zero", "button": "B"}

        if state_id == 3:
            self.register_transition(3)
            self.start_wait_for_states(ALLOWED_TRANSITIONS[3], now)
            return {
                "type": "down_a_combo",
                "buttons": ["DOWN", "A"],
                "delay_before": get_state_action_delay(3),
            }

        if state_id == 4:
            self.register_transition(4)
            self.start_wait_for_states(ALLOWED_TRANSITIONS[4], now)
            return {
                "type": "tap_burst",
                "button": "A",
                "hold": A_CONFIRM_HOLD,
                "repeat_count": CONFIRM_BUY_SPAM_COUNT,
                "repeat_gap": CONFIRM_BUY_SPAM_GAP,
            }

        if state_id == 5:
            self.register_transition(5)
            self.stop_requested = True
            self.clear_wait_expectation()
            return {"type": "final_success", "button": "A"}

        if state_id == 6:
            self.register_transition(6)
            self.return_mode = True
            self.start_wait_for_states(ALLOWED_TRANSITIONS[6], now)
            return {"type": "failed_and_return", "button": "A"}

        if state_id == 7:
            self.register_transition(7)
            self.start_wait_for_states(ALLOWED_TRANSITIONS[7], now)
            return {
                "type": "tap",
                "button": "Y",
                "hold": BUTTON_HOLD,
                "delay_before": get_state_action_delay(7),
            }

        return None


class BalanceManagedAuctionBotController(AuctionBotController):
    def handle_state(self, state_id, now):
        if state_id != 5:
            return super().handle_state(state_id, now)

        if not self.should_handle_state(state_id, now):
            return None

        self.mark_handled()
        self.register_transition(5)
        self.clear_wait_expectation()
        return {"type": "purchase_success", "button": "A"}


class ForzaBot:
    def __init__(self):
        self.gamepad = vg.VX360Gamepad()
        self.recognizer = NeuralRecognizer()
        self.controller = AuctionBotController()
        self.state_history = []
        self.no_detection_since = None
        self.capture_backend = "dxcam"
        self.last_ai_text = "waiting for screen"
        self.last_action_text = "-"
        self.last_status_text = "initializing"
        self.last_mode_text = "boot"
        self.current_fps = 0.0
        self.fps_frames = 0
        self.fps_started_at = time.perf_counter()
        self.last_render_at = 0.0
        self.last_render_signature = None
        self.last_prediction = None
        self.console_ready = False
        self.logs_path = APP_ROOT / "logs" / "logs.txt"
        self.telegram = TelegramNotifier(TELEGRAM_FILE)
        self.success_action_count = 0
        self.last_screen_signature = None
        self.camera_started = False
        self.capture_lock = threading.Lock()
        self.prediction_lock = threading.Lock()
        self.pipeline_stop_event = threading.Event()
        self.pipeline_exception = None
        self.capture_thread = None
        self.infer_thread = None
        self.latest_frame = None
        self.latest_frame_index = 0
        self.latest_prediction_bundle = None
        self.latest_prediction_index = 0
        self.last_logic_prediction_index = 0

        if not DXCAM_AVAILABLE:
            raise RuntimeError("dxcam is required for screen capture")

        self.camera = dxcam.create(
            output_idx=0,
            output_color="BGR"
        )
        if self.camera is None:
            raise RuntimeError("dxcam camera creation failed")
        try:
            self.camera.start(target_fps=120, video_mode=True)
            self.camera_started = True
        except TypeError:
            try:
                self.camera.start(target_fps=120)
                self.camera_started = True
            except Exception:
                self.camera_started = False
        except Exception:
            self.camera_started = False

        self.logs_path.parent.mkdir(parents=True, exist_ok=True)
        with self.logs_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"\n[{self.timestamp_text()}] [BOOT] ForzaBot started "
                f"backend={self.capture_backend}\n"
            )

    def timestamp_text(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log_event(self, tag, message):
        line = f"[{self.timestamp_text()}] [{tag}] {message}"
        with self.logs_path.open("a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")

    def send_purchase_notification(self):
        text = "neural_bot purchase success."
        ok, details = self.telegram.send_message(text)
        if ok:
            self.log_event("TG", "purchase notification sent")
        else:
            self.log_event("TG", f"purchase notification skipped/failed: {details}")

    def is_important_status(self, text):
        important_markers = (
            "timeout",
            "error",
            "returning to 0",
            "return fallback",
            "blocked B",
            "dxcam frame unavailable",
            "stopped by keyboard",
        )
        lowered = text.lower()
        return any(marker in lowered for marker in important_markers)

    def log_important_status(self, text):
        route = " -> ".join(str(item) for item in self.controller.route_history) or "-"
        self.log_event(
            "STATUS",
            f"{text} | ai={self.last_ai_text} | action={self.last_action_text} | route={route}",
        )

    def log_success_checkpoint(self):
        route = " -> ".join(str(item) for item in self.controller.route_history) or "-"
        self.log_event(
            "OK",
            f"count={self.success_action_count} action={self.last_action_text} "
            f"ai={self.last_ai_text} route={route}",
        )

    def enable_ansi_console(self):
        if sys.platform != "win32":
            return

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)

        if handle == 0 or handle == -1:
            return

        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return

        kernel32.SetConsoleMode(handle, mode.value | 0x0004)

    def setup_console(self):
        if self.console_ready:
            return

        self.enable_ansi_console()
        sys.stdout.write("\x1b[2J\x1b[H\x1b[?25l")
        sys.stdout.flush()
        self.console_ready = True
        atexit.register(self.restore_console)

    def restore_console(self):
        if not self.console_ready:
            return

        sys.stdout.write("\x1b[?25h\x1b[6;1H")
        sys.stdout.flush()
        self.console_ready = False

    def set_status(self, text):
        self.last_status_text = text
        if self.is_important_status(text):
            self.log_important_status(text)

    def set_action(self, text):
        self.last_action_text = text

    def update_fps(self, now):
        self.fps_frames += 1
        elapsed = now - self.fps_started_at

        if elapsed >= 0.5:
            self.current_fps = self.fps_frames / elapsed
            self.fps_frames = 0
            self.fps_started_at = now

    def format_route_text(self):
        route = " -> ".join(str(item) for item in self.controller.route_history)
        if not route:
            route = "-"

        expected = ", ".join(
            str(item) for item in sorted(self.controller.expected_states)
        )
        if not expected:
            expected = "-"

        return f"route {route} | next {expected}"

    def format_wait_text(self, now):
        timeout = self.controller.get_wait_timeout()

        if timeout is None or self.controller.wait_started_at is None:
            return "wait idle"

        elapsed = max(0.0, now - self.controller.wait_started_at)
        left = max(0.0, timeout - elapsed)
        return f"wait {elapsed:.2f}/{timeout:.2f}s left {left:.2f}s"

    def render_console(self, now, force=False):
        self.setup_console()

        self.last_render_at = now

        line1 = f"AI: {self.last_ai_text}"
        line2 = f"ACTION: {self.last_action_text}"
        line3 = f"STATUS: {self.last_status_text}"
        line4 = f"FLOW: {self.format_route_text()}"
        line5 = (
            f"MODE: {self.last_mode_text} | {self.format_wait_text(now)}"
        )

        fps_note = f"fps {self.current_fps:5.1f}"
        line1 = f"{line1[:CONSOLE_MAIN_WIDTH]:<{CONSOLE_MAIN_WIDTH}}  {fps_note}"

        lines = [line1, line2, line3, line4, line5]
        padded = [f"{line[:CONSOLE_LINE_WIDTH]:<{CONSOLE_LINE_WIDTH}}" for line in lines]
        signature = tuple(padded)
        if not force and signature == self.last_render_signature:
            return
        self.last_render_signature = signature
        block = "\x1b[H" + "\n".join(padded)
        sys.stdout.write(block)
        sys.stdout.flush()

    def screenshot(self):
        if self.camera_started and hasattr(self.camera, "get_latest_frame"):
            return self.camera.get_latest_frame()
        return self.camera.grab()

    def start_pipeline(self):
        self.pipeline_stop_event.clear()
        self.pipeline_exception = None
        self.latest_frame = None
        self.latest_frame_index = 0
        self.latest_prediction_bundle = None
        self.latest_prediction_index = 0
        self.last_logic_prediction_index = 0
        self.capture_thread = threading.Thread(
            target=self.capture_loop,
            name="capture-loop",
            daemon=True,
        )
        self.infer_thread = threading.Thread(
            target=self.infer_loop,
            name="infer-loop",
            daemon=True,
        )
        self.capture_thread.start()
        self.infer_thread.start()

    def capture_loop(self):
        try:
            while not self.pipeline_stop_event.is_set():
                frame = self.screenshot()
                if frame is None:
                    time.sleep(CAPTURE_IDLE_SLEEP)
                    continue

                with self.capture_lock:
                    self.latest_frame = frame
                    self.latest_frame_index += 1
        except Exception as exc:
            self.pipeline_exception = exc
            self.pipeline_stop_event.set()

    def infer_loop(self):
        last_seen_frame_index = 0

        try:
            while not self.pipeline_stop_event.is_set():
                with self.capture_lock:
                    frame_index = self.latest_frame_index
                    frame = self.latest_frame

                if frame is None or frame_index == 0 or frame_index == last_seen_frame_index:
                    time.sleep(INFER_IDLE_SLEEP)
                    continue

                prediction = self.recognizer.predict(frame)
                last_seen_frame_index = frame_index

                with self.prediction_lock:
                    self.latest_prediction_bundle = {
                        "frame_index": frame_index,
                        "prediction": prediction,
                    }
                    self.latest_prediction_index += 1

                self.update_fps(time.perf_counter())
        except Exception as exc:
            self.pipeline_exception = exc
            self.pipeline_stop_event.set()

    def get_prediction_snapshot(self):
        with self.prediction_lock:
            if self.latest_prediction_bundle is None:
                return None, 0
            return dict(self.latest_prediction_bundle), self.latest_prediction_index

    def ensure_pipeline_ok(self):
        if self.pipeline_exception is not None:
            raise self.pipeline_exception

    def stop_capture(self):
        self.pipeline_stop_event.set()
        for thread in (self.capture_thread, self.infer_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=0.5)
        if self.camera_started and hasattr(self.camera, "stop"):
            try:
                self.camera.stop()
            except Exception:
                pass

    def press_button(self, button_name, hold):
        button = BUTTONS[button_name]
        self.gamepad.press_button(button=button)
        self.gamepad.update()
        time.sleep(hold)
        self.gamepad.release_button(button=button)
        self.gamepad.update()

    def tap(self, button_name, hold=BUTTON_HOLD):
        self.press_button(button_name, hold)

    def down_a_combo(self):
        self.press_button("DOWN", DOWN_HOLD)
        time.sleep(DOWN_TO_A_DELAY)
        self.press_button("A", BUTTON_HOLD)

    def detect_state(self, frame, prediction=None):
        if prediction is None and frame is None:
            self.last_ai_text = "dxcam frame unavailable"
            self.last_prediction = None
            return None

        if prediction is None:
            prediction = self.recognizer.predict(frame)
        self.last_prediction = prediction
        state_id = prediction["state_id"]
        confidence = prediction["confidence"]
        required_confidence = STATE_CONFIDENCE_THRESHOLDS[state_id]
        self.last_ai_text = (
            f"{state_id} {prediction['folder_name']} conf={confidence:.4f}"
        )

        if confidence < required_confidence:
            self.last_ai_text = (
                f"low conf {state_id} {prediction['folder_name']} "
                f"{confidence:.4f} < {required_confidence:.3f}"
            )
            return None

        if not prediction["in_profile"]:
            if should_bypass_profile_reject(state_id, confidence):
                self.last_ai_text = (
                    f"{state_id} {prediction['folder_name']} "
                    f"conf={confidence:.4f} profile-bypass"
                )
            else:
                self.last_ai_text = (
                    f"reject {prediction['folder_name']} conf={confidence:.4f}"
                )
                return None

        self.state_history.append(state_id)

        if len(self.state_history) > STATE_HISTORY_SIZE:
            self.state_history.pop(0)

        if self.state_history.count(state_id) < MIN_STABLE_COUNT:
            self.last_ai_text = (
                f"{state_id} {prediction['folder_name']} conf={confidence:.4f} "
                f"stable {self.state_history.count(state_id)}/{MIN_STABLE_COUNT}"
            )
            return None

        self.last_ai_text = (
            f"{state_id} {prediction['folder_name']} "
            f"conf={confidence:.4f} crop={prediction['crop_mode']}"
        )
        return state_id

    def detect_state_immediate(self, frame=None, prediction=None):
        if prediction is None and frame is None:
            return None

        if prediction is None:
            prediction = self.recognizer.predict(frame)
        state_id = prediction["state_id"]
        confidence = prediction["confidence"]
        required_confidence = STATE_CONFIDENCE_THRESHOLDS[state_id]

        if confidence < required_confidence:
            return None

        if not prediction["in_profile"]:
            if should_bypass_profile_reject(state_id, confidence):
                return state_id
            return None

        return state_id

    def is_probably_zero_screen(self, frame=None, prediction=None):
        if prediction is None and frame is None:
            return False

        if prediction is None:
            prediction = self.recognizer.predict(frame)

        if prediction["state_id"] != 0:
            return False

        if not prediction["in_profile"]:
            return False

        required_confidence = max(
            RETURN_ZERO_SOFT_THRESHOLD,
            STATE_CONFIDENCE_THRESHOLDS[0] - 0.05,
        )
        return prediction["confidence"] >= required_confidence

    def finish_return_to_zero(self):
        self.set_status("return finished: 0 detected")
        self.set_action("B return complete")
        self.controller.return_mode = False
        self.controller.route_history = []
        self.reset_tracking_after_return()

    def reset_tracking_after_return(self):
        self.state_history.clear()
        self.no_detection_since = None
        self.controller.current_state = None
        self.controller.current_state_since = 0.0
        self.controller.current_state_count = 0
        self.controller.last_acted_token = None
        self.controller.clear_wait_expectation()

    def should_force_return(self, now):
        return self.controller.is_wait_timeout_exceeded(now)

    def should_retry_search_form(self):
        if self.controller.return_mode:
            return False

        if self.controller.current_state != 1:
            return False

        return self.controller.expected_states == {2, 7}

    def should_retry_search_form_now(self, now):
        if not self.should_retry_search_form():
            return False

        if self.controller.wait_started_at is None:
            return False

        return now - self.controller.wait_started_at >= SEARCH_FORM_RETRY_TIMEOUT

    def retry_search_form_action(self):
        self.set_status("screen 1 timeout -> retry A")
        action = {
            "type": "tap",
            "button": "A",
            "hold": BUTTON_HOLD,
            "delay_before": get_state_action_delay(1),
        }
        executed_at = self.execute_action(action)
        if executed_at is not None:
            self.controller.notify_action_executed(executed_at)
        return executed_at

    def execute_action(self, action):
        if action is None:
            return None

        time.sleep(get_action_delay(action))

        action_type = action["type"]

        if action_type == "tap":
            button = action["button"]
            hold = action["hold"]
            self.set_action(button)
            self.set_status(f"pressed {button}")
            self.tap(button, hold)
            if button != "B":
                self.success_action_count += 1
                if self.success_action_count % 5 == 0:
                    self.log_success_checkpoint()
            return time.perf_counter()

        if action_type == "down_a_combo":
            self.set_action("DOWN + A")
            self.set_status("pressed DOWN + A")
            self.down_a_combo()
            self.success_action_count += 1
            if self.success_action_count % 5 == 0:
                self.log_success_checkpoint()
            return time.perf_counter()

        if action_type == "tap_burst":
            button = action["button"]
            hold = float(action.get("hold", BUTTON_HOLD))
            repeat_count = int(action.get("repeat_count", 1))
            repeat_gap = float(action.get("repeat_gap", 0.0))
            self.set_action(button)
            self.set_status(
                f"pressed {button} x{repeat_count} every {repeat_gap:.2f}s"
            )
            for index in range(repeat_count):
                self.tap(button, hold)
                if index + 1 < repeat_count:
                    time.sleep(repeat_gap)
            if button != "B":
                self.success_action_count += 1
                if self.success_action_count % 5 == 0:
                    self.log_success_checkpoint()
            return time.perf_counter()

        if action_type == "final_success":
            self.set_action("A -> stop")
            self.set_status("success route completed")
            self.tap("A", A_FINAL_HOLD)
            self.success_action_count += 1
            if self.success_action_count % 5 == 0:
                self.log_success_checkpoint()
            self.send_purchase_notification()
            raise SystemExit

        if action_type == "failed_and_return":
            self.set_action("A -> return 0")
            self.set_status("failed buy, returning to 0")
            self.tap("A", A_FINAL_HOLD)
            self.success_action_count += 1
            if self.success_action_count % 5 == 0:
                self.log_success_checkpoint()
            time.sleep(FAILED_A_TO_B_DELAY)
            self.return_to_zero(force_initial_b=True)
            return time.perf_counter()

        if action_type == "return_to_zero":
            self.set_action("return to 0")
            self.set_status("manual return to 0")
            self.return_to_zero()
            return time.perf_counter()

    def return_to_zero(self, force_initial_b=False):
        self.last_mode_text = "return"
        self.set_status("returning to 0")
        zero_hits = 0
        last_seen_prediction_index = -1
        presses_used = 0

        if force_initial_b:
            self.set_action("B")
            self.set_status("pressing initial B to return")
            self.render_console(time.perf_counter())
            self.tap("B", RETURN_B_HOLD)
            presses_used += 1
            wait_deadline = time.perf_counter() + RETURN_B_INTERVAL
            while time.perf_counter() < wait_deadline:
                time.sleep(RETURN_POLL_INTERVAL)
                self.ensure_pipeline_ok()
                self.render_console(time.perf_counter())

                snapshot, prediction_index = self.get_prediction_snapshot()
                if snapshot is None or prediction_index == last_seen_prediction_index:
                    continue

                prediction = snapshot["prediction"]
                last_seen_prediction_index = prediction_index
                state_id = self.detect_state_immediate(prediction=prediction)

                if state_id == 0 or self.is_probably_zero_screen(prediction=prediction):
                    zero_hits += 1
                    if zero_hits >= RETURN_ZERO_CONFIRM_COUNT:
                        self.finish_return_to_zero()
                        self.render_console(time.perf_counter(), force=True)
                        return
                else:
                    zero_hits = 0

        while presses_used < MAX_RETURN_B_PRESSES:
            self.ensure_pipeline_ok()
            now = time.perf_counter()
            self.render_console(now)
            snapshot, prediction_index = self.get_prediction_snapshot()

            if snapshot is not None and prediction_index != last_seen_prediction_index:
                prediction = snapshot["prediction"]
                last_seen_prediction_index = prediction_index
                state_id = self.detect_state_immediate(prediction=prediction)
            else:
                prediction = None
                state_id = None

            if prediction is not None and (
                state_id == 0 or self.is_probably_zero_screen(prediction=prediction)
            ):
                zero_hits += 1
                if zero_hits >= RETURN_ZERO_CONFIRM_COUNT:
                    self.finish_return_to_zero()
                    self.render_console(time.perf_counter(), force=True)
                    return
            elif prediction is not None:
                zero_hits = 0

            if zero_hits > 0:
                self.set_status(
                    f"confirming 0 screen {zero_hits}/{RETURN_ZERO_CONFIRM_COUNT}"
                )
                self.render_console(time.perf_counter())
                continue

            self.set_action("B")
            self.set_status("pressing B to return")
            self.render_console(time.perf_counter())
            self.tap("B", RETURN_B_HOLD)
            presses_used += 1

            wait_deadline = time.perf_counter() + RETURN_B_INTERVAL
            while time.perf_counter() < wait_deadline:
                time.sleep(RETURN_POLL_INTERVAL)
                self.ensure_pipeline_ok()
                self.render_console(time.perf_counter())

                snapshot, prediction_index = self.get_prediction_snapshot()
                if snapshot is None or prediction_index == last_seen_prediction_index:
                    continue

                prediction = snapshot["prediction"]
                last_seen_prediction_index = prediction_index
                state_id = self.detect_state_immediate(prediction=prediction)

                if state_id == 0 or self.is_probably_zero_screen(prediction=prediction):
                    zero_hits += 1
                    if zero_hits >= RETURN_ZERO_CONFIRM_COUNT:
                        self.finish_return_to_zero()
                        self.render_console(time.perf_counter(), force=True)
                        return
                else:
                    zero_hits = 0

        self.set_status("return fallback reset to 0")
        self.set_action("return reset")
        self.controller.return_mode = False
        self.controller.route_history = []
        self.reset_tracking_after_return()

    def run(self):
        self.setup_console()
        self.start_pipeline()
        self.last_mode_text = "run"
        self.set_status(f"started with {self.capture_backend}")
        self.render_console(time.perf_counter(), force=True)

        while True:
            try:
                self.ensure_pipeline_ok()
                now = time.perf_counter()
                self.last_mode_text = "return" if self.controller.return_mode else "run"
                snapshot, prediction_index = self.get_prediction_snapshot()

                if snapshot is None or prediction_index == self.last_logic_prediction_index:
                    state_id = None
                    frame = None
                    prediction = None
                else:
                    self.last_logic_prediction_index = prediction_index
                    frame = None
                    prediction = snapshot["prediction"]
                    state_id = self.detect_state(frame, prediction=prediction)

                if state_id is None:
                    self.render_console(now)
                    if self.should_retry_search_form_now(now):
                        self.retry_search_form_action()
                        self.render_console(time.perf_counter(), force=True)
                        time.sleep(LOOP_SLEEP)
                        continue
                    if self.should_force_return(now):
                        timeout_route = self.controller.get_timeout_return_route()
                        if timeout_route is not None:
                            route_label = " -> ".join(str(item) for item in timeout_route)
                            self.set_status(f"timeout on route {route_label} -> return 0")
                        else:
                            self.set_status("expected screen timeout -> return 0")
                        self.return_to_zero()
                    time.sleep(LOOP_SLEEP)
                    continue

                self.controller.update_observation(
                    state_id,
                    now,
                    preconfirmed=True,
                )
                action = self.controller.handle_state(state_id, now)
                executed_at = self.execute_action(action)
                if action is not None and executed_at is not None:
                    self.controller.notify_action_executed(executed_at)

                now = time.perf_counter()

                if self.should_retry_search_form_now(now):
                    self.retry_search_form_action()
                    self.render_console(time.perf_counter(), force=True)
                    time.sleep(LOOP_SLEEP)
                    continue
                if self.should_force_return(now):
                    timeout_route = self.controller.get_timeout_return_route()
                    if timeout_route is not None:
                        route_label = " -> ".join(str(item) for item in timeout_route)
                        self.set_status(f"timeout on route {route_label} -> return 0")
                    else:
                        self.set_status("expected screen timeout -> return 0")
                    self.return_to_zero()
                    self.render_console(now)
                    time.sleep(LOOP_SLEEP)
                    continue

                if self.controller.stop_requested:
                    raise SystemExit

                self.render_console(now)
                time.sleep(LOOP_SLEEP)

            except KeyboardInterrupt:
                self.set_status("stopped by keyboard")
                self.render_console(time.perf_counter(), force=True)
                break
            except SystemExit:
                self.set_status("success route completed")
                self.render_console(time.perf_counter(), force=True)
                break
            except Exception as exc:
                if is_cuda_runtime_error(exc):
                    self.set_status("cuda failure -> restarting process")
                    self.log_event("STATUS", f"cuda failure detected: {exc}")
                    self.render_console(time.perf_counter(), force=True)
                    self.log_event("BOOT", "ForzaBot stopped after CUDA failure")
                    self.restore_console()
                    self.stop_capture()
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    raise CudaRestartRequired(str(exc)) from exc
                self.set_status(f"error: {exc}")
                self.render_console(time.perf_counter(), force=True)
                time.sleep(0.2)

        self.log_event("BOOT", "ForzaBot stopped")
        self.stop_capture()
        self.restore_console()


class BalanceManagedForzaBot(ForzaBot):
    def __init__(self, runtime_config):
        self.runtime_config = runtime_config
        self.purchase_cost = sanitize_positive_int(
            runtime_config.purchase_cost,
            DEFAULT_PURCHASE_COST,
        )
        self.starting_balance = sanitize_non_negative_int(
            runtime_config.starting_balance,
            DEFAULT_STARTING_BALANCE,
        )
        super().__init__()
        self.controller = BalanceManagedAuctionBotController()
        self.current_balance = read_balance(BALANCE_FILE, self.starting_balance)
        write_balance(BALANCE_FILE, self.current_balance)
        self.log_event(
            "BAL",
            f"managed mode started balance={self.current_balance} cost={self.purchase_cost}",
        )

    def can_afford_purchase(self):
        return self.current_balance >= self.purchase_cost

    def send_balance_exhausted_notification(self):
        text = (
            "neural_bot stopped: balance is no longer enough for one more purchase.\n"
            f"Balance: {self.current_balance}\n"
            f"Required: {self.purchase_cost}"
        )
        ok, details = self.telegram.send_message(text)
        if ok:
            self.log_event("TG", "balance exhausted notification sent")
        else:
            self.log_event("TG", f"notification skipped/failed: {details}")

    def send_bot_started_notification(self):
        text = (
            "neural_bot started in balance mode.\n"
            f"Balance: {self.current_balance}\n"
            f"Purchase cost: {self.purchase_cost}"
        )
        ok, details = self.telegram.send_message(text)
        if ok:
            self.log_event("TG", "bot started notification sent")
        else:
            self.log_event("TG", f"bot started notification skipped/failed: {details}")

    def send_purchase_notification(self):
        text = (
            "neural_bot purchase success.\n"
            f"Spent: {self.purchase_cost}\n"
            f"Remaining balance: {self.current_balance}"
        )
        ok, details = self.telegram.send_message(text)
        if ok:
            self.log_event("TG", "purchase notification sent")
        else:
            self.log_event("TG", f"purchase notification skipped/failed: {details}")

    def reset_for_next_cycle(self):
        self.controller.reset()
        self.reset_tracking_after_return()
        self.last_mode_text = "run"

    def execute_success_return_combo(self):
        self.set_status("success route return combo: A then controlled B return")
        self.set_action("A -> return 0")
        self.tap("A", A_FINAL_HOLD)
        time.sleep(0.20)
        self.return_to_zero(force_initial_b=True)
        self.reset_for_next_cycle()

    def execute_purchase_success(self):
        self.set_action("A -> return 0")
        self.set_status("purchase success, running A then controlled return")
        self.success_action_count += 1
        if self.success_action_count % 5 == 0:
            self.log_success_checkpoint()

        self.current_balance -= self.purchase_cost
        write_balance(BALANCE_FILE, self.current_balance)
        self.log_event(
            "BAL",
            f"purchase success cost={self.purchase_cost} remaining={self.current_balance}",
        )
        self.send_purchase_notification()

        if self.can_afford_purchase():
            self.execute_success_return_combo()
            return time.perf_counter()

        self.set_status(
            f"balance exhausted after purchase, remaining={self.current_balance}"
        )
        self.controller.stop_requested = True
        self.send_balance_exhausted_notification()
        return time.perf_counter()

    def execute_action(self, action):
        if action is not None and action.get("type") == "purchase_success":
            time.sleep(get_action_delay(action))
            return self.execute_purchase_success()
        return super().execute_action(action)

    def run(self):
        if not self.can_afford_purchase():
            self.set_status(
                f"balance too low to start: {self.current_balance} < {self.purchase_cost}"
            )
            self.log_event("BAL", self.last_status_text)
            self.send_balance_exhausted_notification()
            return

        self.send_bot_started_notification()
        super().run()


def prompt_runtime_config():
    ensure_runtime_file(TELEGRAM_FILE)
    ensure_runtime_file(BALANCE_FILE)
    saved_config = load_runtime_config(RUNTIME_CONFIG_FILE)
    saved_balance = read_balance(BALANCE_FILE, saved_config.starting_balance)

    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception:
        return prompt_runtime_config_console(saved_config, saved_balance)

    result = {"config": None}
    root = tk.Tk()
    root.title("Forza Auction Bot")
    root.geometry("620x430")
    root.minsize(620, 430)
    root.resizable(False, False)
    root.attributes("-topmost", True)

    outer = ttk.Frame(root, padding=16)
    outer.pack(fill="both", expand=True)

    container = ttk.Frame(outer)
    container.pack(fill="both", expand=True)

    ttk.Label(
        container,
        text="Режим запуска бота",
        font=("Segoe UI", 13, "bold"),
    ).pack(anchor="w")
    ttk.Label(
        container,
        text=(
            "Редкая машина: остановить после первой покупки.\n"
            "Обычная машина: продолжать покупки, пока хватает баланса."
        ),
        justify="left",
    ).pack(anchor="w", pady=(4, 12))

    mode_var = tk.StringVar(
        value="stop" if saved_config.stop_after_purchase else "balance"
    )
    balance_var = tk.StringVar(value=str(saved_balance))
    price_var = tk.IntVar(value=saved_config.purchase_cost)
    price_entry_var = tk.StringVar(value=str(saved_config.purchase_cost))

    mode_frame = ttk.LabelFrame(container, text="После покупки")
    mode_frame.pack(fill="x", pady=(0, 12))
    ttk.Radiobutton(
        mode_frame,
        text="Остановить бота после первой успешной покупки",
        variable=mode_var,
        value="stop",
    ).pack(anchor="w", padx=10, pady=(8, 6))
    ttk.Radiobutton(
        mode_frame,
        text="Не останавливать, работать до конца баланса",
        variable=mode_var,
        value="balance",
    ).pack(anchor="w", padx=10, pady=(0, 8))

    balance_frame = ttk.LabelFrame(container, text="Баланс и цена покупки")
    balance_frame.pack(fill="x")
    balance_frame.columnconfigure(1, weight=1)

    ttk.Label(balance_frame, text="Текущий баланс:").grid(
        row=0, column=0, sticky="w", padx=10, pady=(10, 6)
    )
    balance_entry = ttk.Entry(balance_frame, textvariable=balance_var, width=20)
    balance_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(10, 6))

    ttk.Label(balance_frame, text="Цена одной покупки:").grid(
        row=1, column=0, sticky="w", padx=10, pady=6
    )
    price_entry = ttk.Entry(balance_frame, textvariable=price_entry_var, width=20)
    price_entry.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=6)

    price_scale = tk.Scale(
        balance_frame,
        from_=PURCHASE_PRICE_SLIDER_MIN,
        to=PURCHASE_PRICE_SLIDER_MAX,
        orient="horizontal",
        resolution=PURCHASE_PRICE_SLIDER_STEP,
        variable=price_var,
        showvalue=False,
    )
    price_scale.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

    price_hint = ttk.Label(
        balance_frame,
        text=(
            f"Ползунок: {PURCHASE_PRICE_SLIDER_MIN:,} .. "
            f"{PURCHASE_PRICE_SLIDER_MAX:,}"
        ).replace(",", " "),
    )
    price_hint.grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

    def sync_price_from_scale(*_args):
        price_entry_var.set(str(int(price_var.get())))

    def sync_price_from_entry(*_args):
        value = sanitize_positive_int(price_entry_var.get(), price_var.get())
        value = max(PURCHASE_PRICE_SLIDER_MIN, min(PURCHASE_PRICE_SLIDER_MAX, value))
        if value != price_var.get():
            price_var.set(value)

    def toggle_balance_controls(*_args):
        enabled = mode_var.get() == "balance"
        state = "normal" if enabled else "disabled"
        for widget in (balance_entry, price_entry, price_scale):
            widget.configure(state=state)

    price_var.trace_add("write", sync_price_from_scale)
    mode_var.trace_add("write", toggle_balance_controls)
    price_entry.bind("<FocusOut>", sync_price_from_entry)
    price_entry.bind("<Return>", sync_price_from_entry)

    def start_bot():
        stop_after_purchase = mode_var.get() == "stop"
        starting_balance = sanitize_non_negative_int(
            balance_var.get(),
            saved_balance,
        )
        purchase_cost = sanitize_positive_int(
            price_entry_var.get(),
            saved_config.purchase_cost,
        )

        if not stop_after_purchase and purchase_cost > starting_balance:
            proceed = messagebox.askyesno(
                "Проверка баланса",
                (
                    "Цена покупки больше текущего баланса.\n"
                    "Бот сразу остановится по балансу. Запустить всё равно?"
                ),
                parent=root,
            )
            if not proceed:
                return

        result["config"] = BotRuntimeConfig(
            stop_after_purchase=stop_after_purchase,
            purchase_cost=purchase_cost,
            starting_balance=starting_balance,
        )
        root.destroy()

    buttons = ttk.Frame(outer)
    buttons.pack(fill="x", pady=(14, 0))
    ttk.Button(buttons, text="Старт", command=start_bot).pack(side="right")
    ttk.Button(buttons, text="Отмена", command=root.destroy).pack(side="right", padx=(0, 8))

    toggle_balance_controls()
    root.mainloop()
    return result["config"]


def prompt_runtime_config_console(saved_config, saved_balance):
    print("=" * 50)
    print("FORZA AUCTION BOT - SETUP")
    print("=" * 50)
    print("1. Остановить после первой покупки")
    print("2. Работать до конца баланса")
    mode = input("Выбор [1/2]: ").strip()
    if mode == "2":
        balance_text = input(f"Текущий баланс [{saved_balance}]: ").strip()
        price_text = input(f"Цена одной покупки [{saved_config.purchase_cost}]: ").strip()
        return BotRuntimeConfig(
            stop_after_purchase=False,
            purchase_cost=sanitize_positive_int(price_text or saved_config.purchase_cost, saved_config.purchase_cost),
            starting_balance=sanitize_non_negative_int(balance_text or saved_balance, saved_balance),
        )
    return BotRuntimeConfig(
        stop_after_purchase=True,
        purchase_cost=saved_config.purchase_cost,
        starting_balance=saved_balance,
    )


def prepare_runtime_files(runtime_config):
    ensure_runtime_file(TELEGRAM_FILE)
    save_runtime_config(RUNTIME_CONFIG_FILE, runtime_config)
    if not runtime_config.stop_after_purchase:
        write_balance(BALANCE_FILE, runtime_config.starting_balance)


def run_single_bot_process(runtime_config):
    ensure_runtime_file(TELEGRAM_FILE)
    ensure_runtime_file(BALANCE_FILE)
    if runtime_config.stop_after_purchase:
        bot = ForzaBot()
    else:
        bot = BalanceManagedForzaBot(runtime_config)
    bot.run()


def run_with_cuda_restart_supervisor():
    runtime_config = prompt_runtime_config()
    if runtime_config is None:
        return 0

    prepare_runtime_files(runtime_config)
    mode_label = (
        "STOP AFTER FIRST PURCHASE"
        if runtime_config.stop_after_purchase
        else f"BALANCE MODE | balance={runtime_config.starting_balance} cost={runtime_config.purchase_cost}"
    )
    print("=" * 50)
    print("FORZA AUCTION BOT - ROUTE AI")
    print(mode_label)
    print("=" * 50)
    time.sleep(1)
    restart_count = 0

    while True:
        if getattr(sys, "frozen", False):
            child_command = [
                sys.executable,
                "--bot-child",
                f"--runtime-config={RUNTIME_CONFIG_FILE}",
            ]
            child_cwd = APP_ROOT
        else:
            script_path = Path(__file__).resolve()
            child_command = [
                sys.executable,
                str(script_path),
                "--bot-child",
                f"--runtime-config={RUNTIME_CONFIG_FILE}",
            ]
            child_cwd = script_path.parent

        result = subprocess.run(child_command, cwd=child_cwd)

        if result.returncode != CUDA_RESTART_EXIT_CODE:
            return result.returncode

        restart_count += 1
        print(
            f"[SUP] CUDA failure detected. Restarting in "
            f"{CUDA_RESTART_DELAY:.1f}s ({restart_count}/{CUDA_RESTART_LIMIT})"
        )

        if restart_count >= CUDA_RESTART_LIMIT:
            print("[SUP] Restart limit reached. Bot stopped.")
            return CUDA_RESTART_EXIT_CODE

        time.sleep(CUDA_RESTART_DELAY)


if __name__ == "__main__":
    if "--bot-child" in sys.argv:
        runtime_config = load_runtime_config(parse_runtime_config_arg(sys.argv))
        try:
            run_single_bot_process(runtime_config)
        except CudaRestartRequired:
            sys.exit(CUDA_RESTART_EXIT_CODE)
        sys.exit(0)

    sys.exit(run_with_cuda_restart_supervisor())
