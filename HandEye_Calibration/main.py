# /src/main.py
import sys
import os
import time
import json
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

            # print(f"\n[CALIB] Pose {pose_id}: Collecting {samples} samples (timeout: {duration_sec}s)...")
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

            # print(f"\n[INFO] Pose {pose_id} saved. samples={len(collected)}")
            log.info(f"Pose {pose_id} saved. samples={len(collected)}")

            if len(collected) == 0:
                # print(f"[ERROR] Pose {pose_id}: NO VALID DATA!", flush=True)
                log.error(f"Pose {pose_id}: NO VALID DATA!")

            elif len(collected) < samples:
                # print(f"[WARNING] Pose {pose_id}: Only {len(collected)}/{samples} samples (timeout).", flush=True)
                log.warning(f"Pose {pose_id}: Only {len(collected)}/{samples} samples (timeout).")

    finally:
        api.stopTracking()
        robot_controller.move_to_home()
        # print("Calibration finished.", flush=True)
        log.info("Calibration finished.")

# ===========================
# NAVIGATION MODE
# ===========================
def run_navigation_mode(robot_controller, hostname, ttool, rom_dir,
                        encrypted, cipher, calib_json_path,
                        x_offset, y_offset, z_offset):
    """
    READY_TO_NAV → ROBOT_MOVING → ON_TARGET/IDLE 흐름을 내부에서 관리.
    반환값: 'IDLE' or 'SAFE_STOP'
    """

    # print("\n" + "=" * 70)
    # print("[NAVIGATION] ttool 마커 인식 → 로봇 EE 이동 모드")
    # print("  마커 인식 후 목표 좌표 출력 → Enter: 이동 / 'q': 종료")
    # print("=" * 70)
    log.section("NAVIGATION  ttool 마커 인식 → 로봇 EE 이동 모드  |  Enter: 이동 / 'q': 종료")

    if not os.path.exists(calib_json_path):
        # print(f"[ERROR] Calibration result not found: {calib_json_path}")
        log.error(f"Calibration result not found: {calib_json_path}")
        return State.IDLE

    try:
        nav = Navigator(calib_path=calib_json_path)
        # print(f"[INFO] Navigator loaded (method: {nav.method}, unit: {nav.unit})")
        log.info(f"Navigator loaded (method: {nav.method}, unit: {nav.unit})")
    except Exception as e:
        # print(f"[ERROR] Navigator init failed: {e}")
        log.error(f"Navigator init failed: {e}")
        return State.IDLE

    try:
        api, ttool_handle = ndi.connect_and_setup_navigation(
            hostname, ttool, rom_dir, encrypted, cipher
        )
    except RuntimeError as e:
        # print(f"[ERROR] {e}")
        log.error(f"{e}")
        return State.IDLE

    if ttool_handle is None:
        return State.IDLE

    next_state = State.IDLE

    try:
        # ── READY_TO_NAV: Planning & CMD_MOVE loop ──────────────────────────
        while True:
            # print("\n[READY_TO_NAV] 마커 인식 중... (Planning)")
            # print("-" * 70)
            log.info("READY_TO_NAV  마커 인식 중... (Planning)")

            raw_pose, reason = ndi.get_latest_valid_pose(
                api, ttool_handle, timeout_sec=10.0
            )

            if raw_pose is None:
                # print(f"[WARNING] {reason}")
                log.warning(f"{reason}")
                sel = input("  재시도(Enter) / 종료(q): ").strip().lower()
                if sel == 'q':
                    next_state = State.IDLE
                    break
                continue

            # 좌표 변환
            q0, qx, qy, qz = raw_pose['q0'], raw_pose['qx'], raw_pose['qy'], raw_pose['qz']
            tx, ty, tz = raw_pose['tx'], raw_pose['ty'], raw_pose['tz']

            result = nav.compute(q0, qx, qy, qz, tx, ty, tz)

            pose = {**raw_pose, **result}

            x = pose['x'] + x_offset
            y = pose['y'] + y_offset
            z = pose['z'] + z_offset
            u, v, w = pose['u'], pose['v'], pose['w']

            target_pose = [x, y, z, u, v, w]

            # print(f"\n[DETECTED] {pose['ts_str']} | NDI Error: {pose['err']:.3f} mm")
            log.info(f"DETECTED  {pose['ts_str']} | NDI Error: {pose['err']:.3f} mm")

            # print("\n  [NDI Raw]")
            # print(f"    Pos  x={pose['tx']:10.3f}  y={pose['ty']:10.3f}  z={pose['tz']:10.3f} (mm)")
            # print(f"    Quat w={pose['q0']:.5f}  x={pose['qx']:.5f}  y={pose['qy']:.5f}  z={pose['qz']:.5f}")
            log.info(f"NDI Raw  Pos  x={pose['tx']:10.3f}  y={pose['ty']:10.3f}  z={pose['tz']:10.3f} (mm)")
            log.info(f"NDI Raw  Quat w={pose['q0']:.5f}  x={pose['qx']:.5f}  y={pose['qy']:.5f}  z={pose['qz']:.5f}")

            # print("\n  [Navigator Result (Offset 미적용)]")
            # print(f"    x={pose['x']:10.4f}  y={pose['y']:10.4f}  z={pose['z']:10.4f} (mm)")
            # print(f"    u={pose['u']:10.4f}  v={pose['v']:10.4f}  w={pose['w']:10.4f} (deg)")
            log.info(f"Navigator Result  x={pose['x']:10.4f}  y={pose['y']:10.4f}  z={pose['z']:10.4f} (mm) | "
                     f"u={pose['u']:10.4f}  v={pose['v']:10.4f}  w={pose['w']:10.4f} (deg)")

            # print("\n  [Robot Target (Offset 적용, 실제 이동 좌표)]")
            # print(f"    x={x:10.4f}  y={y:10.4f}  z={z:10.4f} (mm)")
            # print(f"    u={u:10.4f}  v={v:10.4f}  w={w:10.4f} (deg)")
            log.info(f"Robot Target  x={x:10.4f}  y={y:10.4f}  z={z:10.4f} (mm) | "
                     f"u={u:10.4f}  v={v:10.4f}  w={w:10.4f} (deg)")

            # print(f"\n  [INDY 포맷]")
            # print(f"    [{x:.4f}, {y:.4f}, {z:.4f}, {u:.4f}, {v:.4f}, {w:.4f}]")
            # print()
            log.info(f"INDY 포맷  [{x:.4f}, {y:.4f}, {z:.4f}, {u:.4f}, {v:.4f}, {w:.4f}]")

            sel = input("  이동하려면 Enter / 재인식(r) / 종료(q): ").strip().lower()

            if sel == 'q':
                # print("[INFO] Navigation mode 종료.")
                log.info("Navigation mode 종료.")
                next_state = State.IDLE
                break

            elif sel == 'r':
                # print("[INFO] 마커 재인식합니다.")
                log.info("마커 재인식합니다.")
                continue

            # ── ROBOT_MOVING ────────────────────────────────────────────────
            # print("\n[ROBOT_MOVING] 로봇 이동 시작...")
            # print(f"  목표: {target_pose}")
            log.info(f"ROBOT_MOVING  로봇 이동 시작...  목표: {target_pose}")

            try:
                robot_controller.movel_to_pose(
                    target_pose,
                    vel_ratio=10,
                    acc_ratio=10,
                    timeout=60
                )
                # TARGET_REACHED → ON_TARGET / IDLE
                # print("[ON_TARGET] 로봇 이동 완료. IDLE 상태로 복귀합니다.")
                log.success("ON_TARGET  로봇 이동 완료. IDLE 상태로 복귀합니다.")
                next_state = State.IDLE
                break

            except Exception as e:
                # print(f"[ERROR_DETECT] 로봇 이동 실패: {e}")
                # print("[→ SAFE_STOP]")
                log.error(f"ERROR_DETECT  로봇 이동 실패: {e}  →  SAFE_STOP")
                next_state = State.SAFE_STOP
                break

    except KeyboardInterrupt:
        # print("\n[ERROR_DETECT] Navigation mode interrupted by user. → SAFE_STOP")
        log.error("ERROR_DETECT  Navigation mode interrupted by user.  →  SAFE_STOP")
        next_state = State.SAFE_STOP

    finally:
        api.stopTracking()
        # print("[INFO] Tracking stopped.")
        log.info("Tracking stopped.")

    return next_state

# ===========================
# MAIN
# ===========================
def main():
    STATE  = State.IDLE
    config = io.load_config("config.json", base_dir=Path(__file__).resolve().parent)

    hostname  = config["ndi"]["hostname"]
    tools     = config["ndi"]["tools"]
    rom_dir   = config["ndi"]["rom_dir"]
    encrypted = config["ndi"]["encrypted"]
    cipher    = config["ndi"]["cipher"]

    robot_ip = config["robot"]["ip"]

    dataset_root    = config["dataset"]["dataset_root"]
    robot_pose_file = config["dataset"]["robot_pose_file"]

    ttool    = config["navigation"]["ttool"]
    x_offset = config["navigation"]["x_offset"]
    y_offset = config["navigation"]["y_offset"]
    z_offset = config["navigation"]["z_offset"]

    paths = io.get_calibration_filepaths(robot_pose_file, dataset_root)

    while True:

        # ── IDLE ──────────────────────────────────────────────────────────
        if STATE == State.IDLE:
            # print("\n" + "=" * 40)
            # print("State: IDLE")
            # print("  1: Tracking Mode    (CMD_TRACKING)")
            # print("  2: Calibration Mode (CMD_CALIBRATION)")
            # print("  3: Navigation Mode  (CMD_NAVIGATION)")
            # print("  Q: Exit")
            # print("=" * 40)
            log.section("State: IDLE  |  1: Tracking  2: Calibration  3: Navigation  Q: Exit")
            sel = input("Select: ").strip()

            if sel == "1":
                STATE = State.TRACKING
            elif sel == "2":
                STATE = State.CALIBRATION
            elif sel == "3":
                STATE = State.READY_TO_NAV
            elif sel.lower() == "q":
                STATE = State.EXIT
            else:
                # print("[WARNING] Invalid selection.")
                log.warning("Invalid selection.")

        # ── TRACKING ──────────────────────────────────────────────────────
        elif STATE == State.TRACKING:
            # print("\n[TRACKING] 트래킹 시작... (CMD_STOP: q)")
            log.info("TRACKING  트래킹 시작... (CMD_STOP: q)")
            try:
                ndi.run_tracking(
                    hostname, tools, rom_dir, encrypted, cipher,
                    print_tracking_data=ndi.print_tracking_data
                )
                # CMD_STOP → IDLE
                # print("[TRACKING] CMD_STOP 수신 → IDLE")
                log.info("TRACKING  CMD_STOP 수신  →  IDLE")
                STATE = State.IDLE

            except Exception as e:
                # print(f"[ERROR_DETECT] Tracking error: {e} → SAFE_STOP")
                log.error(f"ERROR_DETECT  Tracking error: {e}  →  SAFE_STOP")
                STATE = State.SAFE_STOP

        # ── CALIBRATION ───────────────────────────────────────────────────
        elif STATE == State.CALIBRATION:
            robot_controller = RobotController(robot_ip=robot_ip)
            try:
                robot_controller.indy.get_control_state()
            except Exception as e:
                # print("[ERROR] Robot not connected:", e)
                log.error(f"Robot not connected: {e}")
                STATE = State.IDLE
                continue

            try:
                # Robot trajectory & Data collection & Calibration → IDLE
                run_calibration_mode(
                    robot_controller,
                    hostname, tools, rom_dir, encrypted, cipher,
                    robot_pose_file, dataset_root,
                    duration_sec=config["calibration"]["duration_sec"],
                    samples=config["calibration"]["samples"]
                )
                calib = HandEyeCalibration(csv_path=paths["csv"])
                calib.run()
                # print("[CALIBRATION] 완료 → IDLE")
                log.success("CALIBRATION  완료  →  IDLE")
                STATE = State.IDLE

            except Exception as e:
                # print(f"[ERROR_DETECT] Calibration error: {e} → SAFE_STOP")
                log.error(f"ERROR_DETECT  Calibration error: {e}  →  SAFE_STOP")
                STATE = State.SAFE_STOP

        # ── READY_TO_NAV → (ROBOT_MOVING → ON_TARGET/IDLE | SAFE_STOP) ───
        elif STATE == State.READY_TO_NAV:
            calib_json = paths["json"]

            robot_controller = RobotController(robot_ip=robot_ip)
            try:
                robot_controller.indy.get_control_state()
            except Exception as e:
                # print("[ERROR] Robot not connected:", e)
                log.error(f"Robot not connected: {e}")
                STATE = State.IDLE
                continue

            STATE = run_navigation_mode(
                robot_controller=robot_controller,
                hostname=hostname,
                ttool=ttool, rom_dir=rom_dir,
                encrypted=encrypted, cipher=cipher,
                calib_json_path=calib_json,
                x_offset=x_offset, y_offset=y_offset, z_offset=z_offset
            )

        # ── SAFE_STOP ─────────────────────────────────────────────────────
        elif STATE == State.SAFE_STOP:
            # print("\n" + "=" * 40)
            # print("State: SAFE_STOP")
            # print("  로봇/트래커가 안전하게 정지되었습니다.")
            # print("  RECOVERY_RETRY: 시스템을 점검 후 계속하려면 Enter를 누르세요.")
            # print("  종료하려면 q를 입력하세요.")
            # print("=" * 40)
            log.section("State: SAFE_STOP  |  로봇/트래커 안전 정지  |  Enter: RECOVERY_RETRY  q: Exit")
            sel = input("Select (Enter=RECOVERY_RETRY / q=Exit): ").strip().lower()

            if sel == 'q':
                STATE = State.EXIT
            else:
                # RECOVERY_RETRY → IDLE
                # print("[SAFE_STOP] RECOVERY_RETRY → IDLE")
                log.info("SAFE_STOP  RECOVERY_RETRY  →  IDLE")
                STATE = State.IDLE

        # ── EXIT ──────────────────────────────────────────────────────────
        elif STATE == State.EXIT:
            # print("Exit.")
            log.info("Exit.")
            break

if __name__ == "__main__":
    main()