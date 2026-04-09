# /src/main.py
import sys
import os
import time
import json
import threading
from typing import Optional
import numpy as np
from pathlib import Path

import src.ndi.tracker as ndi
from src.robot.controller import RobotController
from src.calib.calibration import HandEyeCalibration
from src.calib.navigator import Navigator
import src.utils.io as io
from src.utils.logger import get_logger
log = get_logger(__name__)

# ===========================
# State machine
# ===========================
class State:
    IDLE          = "IDLE"
    TRACKING      = "TRACKING"
    CALIBRATION   = "CALIBRATION"
    READY_TO_NAV  = "READY_TO_NAV"
    ROBOT_MOVING  = "ROBOT_MOVING"
    SAFE_STOP     = "SAFE_STOP"
    EXIT          = "EXIT"

# ===========================
# Shared action state (thread-safe)
# ===========================
_action_lock    = threading.Lock()
_pending_action = None  # action.json으로부터 수신된 미처리 액션
_pending_move   = None  # 'move' 액션 데이터 (메인 루프에서 ROBOT_MOVING 처리)

def _get_and_clear_pending_action():
    global _pending_action
    with _action_lock:
        action = _pending_action
        _pending_action = None
    return action

def _set_pending_action(action_data: dict):
    global _pending_action
    with _action_lock:
        _pending_action = action_data
    log.info(f"[ActionWatcher] New action received: {action_data.get('action')} — {action_data.get('description', '')}")

# ===========================
# Non-blocking input helper
# ===========================

class NonBlockingInput:
    """
    별도 스레드에서 input()을 실행하여 메인 루프가 블로킹되지 않도록 합니다.

    사용법:
        nbi = NonBlockingInput("Select: ")
        nbi.start()
        while True:
            result = nbi.get()   # None이면 아직 입력 없음
            if result is not None:
                break
            time.sleep(0.05)
    """
    def __init__(self, prompt: str = ""):
        self._prompt = prompt
        self._result = None
        self._ready  = False
        self._lock   = threading.Lock()
        self._thread = None

    def start(self):
        self._result = None
        self._ready  = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            val = input(self._prompt)
        except EOFError:
            val = ""
        with self._lock:
            self._result = val
            self._ready  = True

    def get(self) -> Optional[str]:
        """입력이 완료되면 문자열, 아직 대기 중이면 None 반환."""
        with self._lock:
            if self._ready:
                return self._result
        return None


# ===========================
# action → State 매핑
# ===========================
_ACTION_TO_STATE = {
    "tracking":    State.TRACKING,
    "calibration": State.CALIBRATION,
    "navigation":  State.READY_TO_NAV,
    "stop":        State.SAFE_STOP,
}

def _resolve_state_from_action(action_data: dict, config: dict,
                                robot_controller: Optional[object] = None) -> Optional[str]:
    """
    action.json 데이터를 파싱하여 전이할 State를 반환합니다.
    action == "navigation"이면 offset을 config에 반영합니다.
    action == "move"이면 _pending_move에 저장 후 State.ROBOT_MOVING을 반환합니다.
      → 실제 이동은 메인 루프의 ROBOT_MOVING 분기에서 처리됩니다.

    반환값
    ------
    str   : 전이할 State (State.TRACKING 등)
    None  : 알 수 없는 명령 — 무시
    """
    global _pending_move
    action = action_data.get("action")

    if action is None:
        log.warning(f"[ActionWatcher] Unrecognized command: {action_data.get('description', '')}")
        return None

    if action == "navigation":
        offset = action_data.get("offset", {})
        io.apply_navigation_offset_to_config(config, offset)
        log.info(
            f"[ActionWatcher] Navigation offset applied → "
            f"x={config['navigation']['x_offset']}, "
            f"y={config['navigation']['y_offset']}, "
            f"z={config['navigation']['z_offset']}"
        )
        return _ACTION_TO_STATE.get(action)

    if action == "move":
        if robot_controller is None:
            log.warning("[ActionWatcher] 'move' 액션 수신했으나 robot_controller가 없습니다. 무시합니다.")
            return None
        with _action_lock:
            _pending_move = action_data
        return State.ROBOT_MOVING

    return _ACTION_TO_STATE.get(action)


# ===========================
# Non-blocking state prompts
# ===========================

def _idle_select(config: dict, current_state: str,
                 robot_controller: Optional[object] = None) -> str:
    """
    IDLE 상태: 키보드 입력과 action.json pending을 동시에 감시.
    0.05 s polling으로 어느 쪽이든 먼저 오는 것에 반응합니다.
    j: 키보드 조그 모드 (어느 상태에서든 실행 가능)
    """
    log.section("State: IDLE  |  1: Tracking  2: Calibration  3: Navigation  j: 키보드 조그  Q: Exit"
                "\n           (음성 명령 또는 키보드 입력 대기 중...)")

    nbi = NonBlockingInput("Select: ")
    nbi.start()

    while True:
        # ── 1. action.json 이벤트 우선 확인 ──────────────────────────────
        pending = _get_and_clear_pending_action()
        if pending is not None:
            new_state = _resolve_state_from_action(pending, config, robot_controller)
            if new_state is not None:
                log.info(f"[ActionWatcher] IDLE → {new_state}  (음성 명령)")
                return new_state
            log.warning(f"[ActionWatcher] 알 수 없는 명령, 무시: {pending.get('description', '')}")

        # ── 2. 키보드 입력 확인 ───────────────────────────────────────────
        sel = nbi.get()
        if sel is not None:
            sel = sel.strip()
            if sel == "1":
                return State.TRACKING
            elif sel == "2":
                return State.CALIBRATION
            elif sel == "3":
                return State.READY_TO_NAV
            elif sel.lower() == "j":
                if robot_controller is not None:
                    log.info("키보드 조그 모드 진입. (q / ESC: 조그 종료 후 IDLE 복귀)")
                    robot_controller.keyboard_jog(vel_ratio=10, acc_ratio=10)
                    log.info("키보드 조그 종료 → IDLE 복귀")
                else:
                    log.warning("robot_controller가 없어 키보드 조그를 실행할 수 없습니다.")
                # 조그 후 다시 IDLE 프롬프트 재시작
                nbi = NonBlockingInput("Select: ")
                nbi.start()
                continue
            elif sel.lower() == "q":
                return State.EXIT
            else:
                log.warning("Invalid selection.")
                return current_state  # 다시 IDLE로

        time.sleep(0.05)


def _safe_stop_select(config: dict,
                      robot_controller: Optional[object] = None) -> str:
    """
    SAFE_STOP 상태: 키보드 입력과 action.json pending을 동시에 감시.
    j: 키보드 조그 모드 (어느 상태에서든 실행 가능)
    """
    log.section("State: SAFE_STOP  |  로봇/트래커 안전 정지"
                "\n           Enter: RECOVERY_RETRY  j: 키보드 조그  q: Exit"
                "\n           (음성 명령 또는 키보드 입력 대기 중...)")

    nbi = NonBlockingInput("Select (Enter=RECOVERY_RETRY / j=키보드 조그 / q=Exit): ")
    nbi.start()

    while True:
        pending = _get_and_clear_pending_action()
        if pending is not None:
            new_state = _resolve_state_from_action(pending, config, robot_controller)
            if new_state is not None:
                log.info(f"[ActionWatcher] SAFE_STOP → {new_state}  (음성 명령)")
                return new_state
            log.warning(f"[ActionWatcher] 알 수 없는 명령, 무시: {pending.get('description', '')}")

        sel = nbi.get()
        if sel is not None:
            sel = sel.strip().lower()
            if sel == 'q':
                return State.EXIT
            elif sel == 'j':
                if robot_controller is not None:
                    log.info("키보드 조그 모드 진입. (q / ESC: 조그 종료 후 SAFE_STOP 복귀)")
                    robot_controller.keyboard_jog(vel_ratio=10, acc_ratio=10)
                    log.info("키보드 조그 종료 → SAFE_STOP 복귀")
                else:
                    log.warning("robot_controller가 없어 키보드 조그를 실행할 수 없습니다.")
                # 조그 후 다시 SAFE_STOP 프롬프트 재시작
                nbi = NonBlockingInput("Select (Enter=RECOVERY_RETRY / j=키보드 조그 / q=Exit): ")
                nbi.start()
                continue
            else:
                log.info("SAFE_STOP  RECOVERY_RETRY  →  IDLE")
                return State.IDLE

        time.sleep(0.05)


# ===========================
# CALIBRATION MODE
# ===========================
def run_calibration_mode(robot_controller, hostname, tools, rom_dir, encrypted, cipher,
                         robot_pose_file, dataset_root,
                         duration_sec, samples,
                         ):

    csv_file = io.delete_calibration_csv(robot_pose_file, dataset_root)

    robot_controller.move_to_home()

    with open(robot_pose_file, "r", encoding="utf-8") as f:
        pose_list = sorted(json.load(f), key=lambda x: x["sample_number"])

    api = ndi.connect_and_setup_calibration(hostname, tools, rom_dir, encrypted, cipher)

    try:
        for pose in pose_list:
            pose_id    = pose["sample_number"]
            target_pos = pose["pose"]

            log.info(f"[CALIB] Pose {pose_id}: Collecting {samples} samples (timeout: {duration_sec}s)...")

            robot_controller.movel_to_pose(target_pos, vel_ratio=10, acc_ratio=10, timeout=60)
            time.sleep(1)

            def on_sample(full_data):
                pos = full_data["position"]
                q   = full_data["quaternion"]
                err = full_data["error_mm"]

                tool_data = {
                    "q0": q["w"], "qx": q["x"],
                    "qy": q["y"], "qz": q["z"],
                    "tx": pos["x"], "ty": pos["y"], "tz": pos["z"],
                    "error": err,
                }

                pose_state = robot_controller.get_current_pose()
                robot_data = {
                    "x": pose_state[0], "y": pose_state[1], "z": pose_state[2],
                    "u": pose_state[3], "v": pose_state[4], "w": pose_state[5],
                }

                io.save_data_to_csv(csv_file, full_data["timestamp"], pose_id, tool_data, robot_data=robot_data)

            collected = ndi.collect_marker_samples(api, samples, duration_sec, pose_id, on_sample)

            log.info(f"Pose {pose_id} saved. samples={len(collected)}")

            if len(collected) == 0:
                log.error(f"Pose {pose_id}: NO VALID DATA!")
            elif len(collected) < samples:
                log.warning(f"Pose {pose_id}: Only {len(collected)}/{samples} samples (timeout).")

    finally:
        api.stopTracking()
        robot_controller.move_to_home()
        log.info("Calibration finished.")


# ===========================
# NAVIGATION MODE
# ===========================
def run_navigation_mode(robot_controller, hostname, ttool, rom_dir,
                        encrypted, cipher, calib_json_path,
                        x_offset, y_offset, z_offset,
                        config: Optional[dict] = None):
    """
    READY_TO_NAV → ROBOT_MOVING → ON_TARGET/IDLE 흐름을 내부에서 관리.
    반환값: 'IDLE' or 'SAFE_STOP'
    """
    log.section("NAVIGATION  ttool 마커 인식 → 로봇 EE 이동 모드  |  Enter: 이동 / j: 키보드 조그 / q: 종료")

    if not os.path.exists(calib_json_path):
        log.error(f"Calibration result not found: {calib_json_path}")
        return State.IDLE

    try:
        nav = Navigator(calib_path=calib_json_path)
        log.info(f"Navigator loaded (method: {nav.method}, unit: {nav.unit})")
    except Exception as e:
        log.error(f"Navigator init failed: {e}")
        return State.IDLE

    try:
        api, ttool_handle = ndi.connect_and_setup_navigation(
            hostname, ttool, rom_dir, encrypted, cipher
        )
    except RuntimeError as e:
        log.error(f"{e}")
        return State.IDLE

    if ttool_handle is None:
        return State.IDLE

    next_state = State.IDLE

    try:
        while True:
            # ── action.json 이벤트 감시 (navigation 루프 내) ─────────────────
            pending = _get_and_clear_pending_action()
            if pending is not None and config is not None:
                new_state = _resolve_state_from_action(pending, config, robot_controller)
                if new_state is not None:
                    log.info(f"[ActionWatcher] READY_TO_NAV → {new_state}  (음성 명령)")
                    next_state = new_state
                    break
                log.warning(f"[ActionWatcher] 알 수 없는 명령, 무시: {pending.get('description', '')}")

            log.info("READY_TO_NAV  마커 인식 중... (Planning)")

            raw_pose, reason = ndi.get_latest_valid_pose(
                api, ttool_handle, timeout_sec=10.0
            )

            if raw_pose is None:
                log.warning(f"{reason}")
                sel = input("  재시도(Enter) / 키보드조그(j) / 종료(q): ").strip().lower()
                if sel == 'q':
                    next_state = State.IDLE
                    break
                elif sel == 'j':
                    _ret = robot_controller.keyboard_jog(vel_ratio=10, acc_ratio=10)
                    if _ret == 'quit':
                        log.info("키보드 조그 종료 → 마커 재인식으로 복귀")
                continue

            q0, qx, qy, qz = raw_pose['q0'], raw_pose['qx'], raw_pose['qy'], raw_pose['qz']
            tx, ty, tz = raw_pose['tx'], raw_pose['ty'], raw_pose['tz']

            result = nav.compute(q0, qx, qy, qz, tx, ty, tz)
            pose   = {**raw_pose, **result}

            x = pose['x'] + x_offset
            y = pose['y'] + y_offset
            z = pose['z'] + z_offset
            u, v, w = pose['u'], pose['v'], pose['w']

            target_pose = [x, y, z, u, v, w]

            log.info(f"DETECTED  {pose['ts_str']} | NDI Error: {pose['err']:.3f} mm")
            log.info(f"NDI Raw  Pos  x={pose['tx']:10.3f}  y={pose['ty']:10.3f}  z={pose['tz']:10.3f} (mm)")
            log.info(f"NDI Raw  Quat w={pose['q0']:.5f}  x={pose['qx']:.5f}  y={pose['qy']:.5f}  z={pose['qz']:.5f}")
            log.info(f"Navigator Result  x={pose['x']:10.4f}  y={pose['y']:10.4f}  z={pose['z']:10.4f} (mm) | "
                     f"u={pose['u']:10.4f}  v={pose['v']:10.4f}  w={pose['w']:10.4f} (deg)")
            log.info(f"Robot Target  x={x:10.4f}  y={y:10.4f}  z={z:10.4f} (mm) | "
                     f"u={u:10.4f}  v={v:10.4f}  w={w:10.4f} (deg)")
            log.info(f"INDY 포맷  [{x:.4f}, {y:.4f}, {z:.4f}, {u:.4f}, {v:.4f}, {w:.4f}]")

            sel = input("  이동하려면 Enter / 키보드조그(j) / 재인식(r) / 종료(q): ").strip().lower()

            if sel == 'q':
                log.info("Navigation mode 종료.")
                next_state = State.IDLE
                break
            elif sel == 'r':
                log.info("마커 재인식합니다.")
                continue
            elif sel == 'j':
                log.info("키보드 조그 모드 진입. (q / ESC: 조그 종료 후 마커 재인식으로 복귀)")
                robot_controller.keyboard_jog(vel_ratio=10, acc_ratio=10)
                log.info("키보드 조그 종료 → 마커 재인식으로 복귀")
                continue

            log.info(f"ROBOT_MOVING  로봇 이동 시작...  목표: {target_pose}")
            try:
                robot_controller.movel_to_pose(
                    target_pose, vel_ratio=10, acc_ratio=10, timeout=60
                )
                log.success("ON_TARGET  로봇 이동 완료. IDLE 상태로 복귀합니다.")
                next_state = State.IDLE
                break
            except Exception as e:
                log.error(f"ERROR_DETECT  로봇 이동 실패: {e}  →  SAFE_STOP")
                next_state = State.SAFE_STOP
                break

    except KeyboardInterrupt:
        log.error("ERROR_DETECT  Navigation mode interrupted by user.  →  SAFE_STOP")
        next_state = State.SAFE_STOP

    finally:
        api.stopTracking()
        log.info("Tracking stopped.")

    return next_state


# ===========================
# MAIN
# ===========================
def main():
    STATE      = State.IDLE
    prev_state = State.IDLE
    config = io.load_config("config.json", base_dir=Path(__file__).resolve().parent)

    hostname  = config["ndi"]["hostname"]
    tools     = config["ndi"]["tools"]
    rom_dir   = config["ndi"]["rom_dir"]
    encrypted = config["ndi"]["encrypted"]
    cipher    = config["ndi"]["cipher"]

    robot_ip = config["robot"]["ip"]

    dataset_root    = config["dataset"]["dataset_root"]
    robot_pose_file = config["dataset"]["robot_pose_file"]

    ttool = config["navigation"]["ttool"]

    paths = io.get_calibration_filepaths(robot_pose_file, dataset_root)

    # ── ActionWatcher 시작 ─────────────────────────────────────────────────
    action_file = os.path.join(
        Path(__file__).resolve().parent.parent,  # HandEye_Calibration 상위 = 프로젝트 루트
        "shared", "action.json"
    )
    watcher = io.ActionWatcher(action_file, _set_pending_action)
    watcher.start()

    # ── RobotController: 메인 루프 전체에서 공유 ────────────────────────────
    # 키보드 조그 및 move 액션을 어느 상태에서든 사용하기 위해 상위 스코프에서 관리합니다.
    robot_controller: Optional[RobotController] = None

    def _ensure_robot_controller() -> bool:
        """robot_controller가 없거나 연결이 끊긴 경우 재연결을 시도합니다."""
        nonlocal robot_controller
        if robot_controller is None:
            robot_controller = RobotController(robot_ip=robot_ip)
        try:
            robot_controller.indy.get_control_state()
            return True
        except Exception as e:
            log.error(f"Robot not connected: {e}")
            robot_controller = None
            return False

    try:
        while True:

            # ── IDLE ──────────────────────────────────────────────────────────
            if STATE == State.IDLE:
                # IDLE 진입 시 robot_controller 확보 시도 (실패해도 IDLE 유지)
                if robot_controller is None:
                    try:
                        robot_controller = RobotController(robot_ip=robot_ip)
                        robot_controller.indy.get_control_state()
                        log.info("Robot connected.")
                    except Exception:
                        log.warning("Robot not available. 키보드 조그 및 move 액션 비활성화.")
                        robot_controller = None

                prev_state = State.IDLE
                STATE = _idle_select(config, STATE, robot_controller)

            # ── TRACKING ──────────────────────────────────────────────────────
            elif STATE == State.TRACKING:
                prev_state = State.TRACKING
                log.info("TRACKING  트래킹 시작... (CMD_STOP: q)")
                try:
                    ndi.run_tracking(
                        hostname, tools, rom_dir, encrypted, cipher,
                        print_tracking_data=ndi.print_tracking_data
                    )
                    log.info("TRACKING  CMD_STOP 수신  →  IDLE")
                    STATE = State.IDLE

                except Exception as e:
                    log.error(f"ERROR_DETECT  Tracking error: {e}  →  SAFE_STOP")
                    STATE = State.SAFE_STOP

            # ── CALIBRATION ───────────────────────────────────────────────────
            elif STATE == State.CALIBRATION:
                prev_state = State.CALIBRATION
                if not _ensure_robot_controller():
                    STATE = State.IDLE
                    continue

                try:
                    run_calibration_mode(
                        robot_controller,
                        hostname, tools, rom_dir, encrypted, cipher,
                        robot_pose_file, dataset_root,
                        duration_sec=config["calibration"]["duration_sec"],
                        samples=config["calibration"]["samples"]
                    )
                    calib = HandEyeCalibration(csv_path=paths["csv"])
                    calib.run()
                    log.success("CALIBRATION  완료  →  IDLE")
                    STATE = State.IDLE

                except Exception as e:
                    log.error(f"ERROR_DETECT  Calibration error: {e}  →  SAFE_STOP")
                    STATE = State.SAFE_STOP

            # ── READY_TO_NAV ──────────────────────────────────────────────────
            elif STATE == State.READY_TO_NAV:
                prev_state = State.READY_TO_NAV
                calib_json = paths["json"]

                # action.json offset이 반영된 최신 값 사용
                x_offset = config["navigation"]["x_offset"]
                y_offset = config["navigation"]["y_offset"]
                z_offset = config["navigation"]["z_offset"]

                if not _ensure_robot_controller():
                    STATE = State.IDLE
                    continue

                STATE = run_navigation_mode(
                    robot_controller=robot_controller,
                    hostname=hostname,
                    ttool=ttool, rom_dir=rom_dir,
                    encrypted=encrypted, cipher=cipher,
                    calib_json_path=calib_json,
                    x_offset=x_offset, y_offset=y_offset, z_offset=z_offset,
                    config=config
                )

            # ── ROBOT_MOVING ──────────────────────────────────────────────────
            elif STATE == State.ROBOT_MOVING:
                global _pending_move
                with _action_lock:
                    move_data = _pending_move
                    _pending_move = None

                if move_data is None:
                    log.warning("ROBOT_MOVING  이동 데이터 없음  →  이전 상태 복귀")
                    STATE = prev_state
                    continue

                offset      = move_data.get("offset", {})
                dx          = offset.get("x", 0.0)
                dy          = offset.get("y", 0.0)
                dz          = offset.get("z", 0.0)
                du          = offset.get("u", 0.0)
                dv          = offset.get("v", 0.0)
                dw          = offset.get("w", 0.0)
                description = move_data.get("description", "")

                log.section(f"State: ROBOT_MOVING  |  {description}"
                            f"\n           offset=[{dx}, {dy}, {dz}, {du}, {dv}, {dw}]")
                try:
                    robot_controller.movel_relative_to_pose(
                        [dx, dy, dz, du, dv, dw],
                        vel_ratio=10, acc_ratio=10, timeout=60
                    )
                    log.success(f"ROBOT_MOVING  이동 완료  →  {prev_state}")
                    STATE = prev_state
                except Exception as e:
                    log.error(f"ERROR_DETECT  로봇 이동 실패: {e}  →  SAFE_STOP")
                    STATE = State.SAFE_STOP

            # ── SAFE_STOP ─────────────────────────────────────────────────────
            elif STATE == State.SAFE_STOP:
                STATE = _safe_stop_select(config, robot_controller)

            # ── EXIT ──────────────────────────────────────────────────────────
            elif STATE == State.EXIT:
                log.info("Exit.")
                break

    finally:
        watcher.stop()

if __name__ == "__main__":
    main()