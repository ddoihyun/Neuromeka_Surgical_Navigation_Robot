import csv
import json
import os
import threading
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

CSV_HEADER = [
    "timestamp", "pose_id",
    "q0", "qx", "qy", "qz", "tx", "ty", "tz", "error",
    "x", "y", "z", "u", "v", "w"
]

# ===========================
# Action.json Watcher
# ===========================

class ActionFileHandler(FileSystemEventHandler):
    """
    ./shared/action.json 파일 변경을 감지하여
    콜백 함수로 새 action 데이터를 전달합니다.
    """
    def __init__(self, action_file_path: str, callback):
        super().__init__()
        self._action_file_path = os.path.abspath(action_file_path)
        self._callback = callback
        self._lock = threading.Lock()
        self._last_mtime = None

    def on_modified(self, event):
        if os.path.abspath(event.src_path) == self._action_file_path:
            self._trigger()

    def on_created(self, event):
        if os.path.abspath(event.src_path) == self._action_file_path:
            self._trigger()

    def _trigger(self):
        try:
            mtime = os.path.getmtime(self._action_file_path)
        except OSError:
            return

        with self._lock:
            # 동일 mtime 중복 이벤트 방지
            if mtime == self._last_mtime:
                return
            self._last_mtime = mtime

        try:
            with open(self._action_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._callback(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[ActionWatcher] Failed to read {self._action_file_path}: {e}")


class ActionWatcher:
    """
    watchdog Observer를 래핑한 헬퍼 클래스.

    사용법:
        watcher = ActionWatcher("./shared/action.json", on_action_received)
        watcher.start()
        ...
        watcher.stop()
    """
    def __init__(self, action_file_path: str, callback):
        self._action_file_path = os.path.abspath(action_file_path)
        self._callback = callback
        self._observer = None
        self._watch_dir = os.path.dirname(self._action_file_path)

    def start(self):
        os.makedirs(self._watch_dir, exist_ok=True)
        handler = ActionFileHandler(self._action_file_path, self._callback)
        self._observer = Observer()
        self._observer.schedule(handler, self._watch_dir, recursive=False)
        self._observer.start()
        print(f"[ActionWatcher] Watching: {self._action_file_path}")

    def stop(self):
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            print("[ActionWatcher] Stopped.")


# ===========================
# Config offset updater
# ===========================

def apply_navigation_offset_to_config(config: dict, offset: dict) -> dict:
    """
    action.json의 navigation offset 값을
    인메모리 config dict의 navigation 섹션에 반영합니다.

    offset 키: x, y, z  (u, v, w 는 로봇 이동 시 직접 사용되므로 config에는 저장하지 않음)
    """
    nav = config.setdefault("navigation", {})
    nav["x_offset"] = float(offset.get("x", nav.get("x_offset", 0.0)))
    nav["y_offset"] = float(offset.get("y", nav.get("y_offset", 0.0)))
    nav["z_offset"] = float(offset.get("z", nav.get("z_offset", 0.0)))
    return config


# ===========================
# 기존 유틸리티 함수
# ===========================

def delete_calibration_csv(robot_pose_file, dataset_root):
    paths = get_calibration_filepaths(robot_pose_file, dataset_root)
    csv_file = paths["csv"]
    if os.path.exists(csv_file):
        os.remove(csv_file)
        print(f"Deleted existing {csv_file}")
    return csv_file

def save_data_to_csv(filename, timestamp, pose_id, tool_data, robot_data=None):
    file_exists = os.path.exists(filename)
    with open(filename, mode="a", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerow([
            timestamp, pose_id,
            tool_data["q0"], tool_data["qx"], tool_data["qy"], tool_data["qz"],
            tool_data["tx"], tool_data["ty"], tool_data["tz"],
            tool_data["error"],
            robot_data["x"], robot_data["y"], robot_data["z"],
            robot_data["u"], robot_data["v"], robot_data["w"],
        ])


def load_config(path="config.json", base_dir=None):
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir).resolve()

    config_path = base_dir / path

    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found.")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if config["ndi"]["rom_dir"] is not None:
        config["ndi"]["rom_dir"] = str((base_dir / config["ndi"]["rom_dir"]).resolve())

    config["dataset"]["robot_pose_file"] = str((base_dir / config["dataset"]["robot_pose_file"]).resolve())

    return config


def get_calibration_filepaths(robot_pose_file, dataset_root):
    base = os.path.basename(robot_pose_file)
    name = os.path.splitext(base)[0]
    suffix = name.replace("robot_pose_", "")
    suffix = f"_{suffix}" if suffix else ""

    calib_dir = os.path.join(dataset_root, "calibration")
    results_dir = os.path.join(dataset_root, "results")
    os.makedirs(calib_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    return {
        "csv": os.path.join(calib_dir, f"calibration_data{suffix}.csv"),
        "json": os.path.join(results_dir, f"calibration_result{suffix}.json"),
        "png": os.path.join(results_dir, f"calibration_result{suffix}.png"),
    }