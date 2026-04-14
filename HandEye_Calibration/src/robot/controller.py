"""
controller.py – 로봇 제어 클래스 (Neuromeka IndyDCP)

motion.py 의 유틸을 사용하여 RobotController 를 구현한다.
main.py 를 비롯한 상위 모듈은 이 클래스만 import 하면 된다.

단독 실행 시:
    python controller.py                          # 홈 이동만
    python controller.py --json dataset/poses/robot_pose_broad.json  # json 경로 지정
    python controller.py --pose 597 -143 225 79 -176 80              # 단일 movel
    python controller.py --jog                                        # 키보드 조그
"""

import time
import signal
import threading
import queue
import sys
import os
import argparse
from typing import List
from src.utils.logger import get_logger
log = get_logger(__name__)

from neuromeka import IndyDCP3, OpState

# ── import 경로 (단독 실행 / 패키지 실행 모두 대응) ──────────────────
try:
    from src.robot.motion import (movej_and_wait, movel_and_wait,
                                   movel_relative_and_wait, movel_relative,
                                   movel_from_json)
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.robot.motion import (movej_and_wait, movel_and_wait,
                                   movel_relative_and_wait, movel_relative,
                                   movel_from_json)

_OPSTATE_MAP: dict = {
    0:  "SYSTEM_OFF", 1:  "SYSTEM_ON", 2:  "VIOLATE",
    3:  "RECOVER_HARD", 4:  "RECOVER_SOFT", 5:  "IDLE",
    6:  "MOVING", 7:  "TEACHING", 8:  "COLLISION",
    9:  "STOP_AND_OFF", 10: "COMPLIANCE", 11: "BRAKE_CONTROL",
    12: "SYSTEM_RESET", 13: "SYSTEM_SWITCH", 15: "VIOLATE_HARD",
    16: "MANUAL_RECOVER", 17: "TELE_OP",
}

class RobotController:
    """
    Neuromeka Indy 로봇 제어 클래스.
    연결 확인, 홈 이동, movej/movel 이동을 담당한다.
    """

    # 캘리브레이션 기준 홈 포지션 (조인트각, deg)
    # EE 좌표: x=597.54, y=-143.74, z=225.75, u=79.39, v=-176.49, w=80.84
    DEFAULT_HOME_JOINT: List[float] = [
        0.00, -10.01, -100.01, 10.00, 20.00, 0.00
    ]

    # 키보드 조그 기본 이동량
    JOG_LINEAR_MM:   float = 2.0   # 1회 선형 이동량 (mm)
    JOG_ANGULAR_DEG: float = 1.0   # 1회 회전 이동량 (deg)
    # 꾹 눌렀을 때 명령 재전송 주기 (초)
    JOG_SEND_INTERVAL: float = 0.03  # ≈ 33 Hz

    def __init__(self, robot_ip: str = '192.168.0.137', index: int = 0):
        """
        Parameters
        ----------
        robot_ip : str – 로봇 IP 주소
        index    : int – IndyDCP3 인덱스

        Raises
        ------
        ConnectionError : 연결 또는 상태 조회 실패 시
        """
        self.indy = IndyDCP3(robot_ip=robot_ip, index=index)
        try:
            self.indy.get_control_state()
        except Exception as e:
            raise ConnectionError(f"Robot connection failed: {e}")
        self.home_joint = list(self.DEFAULT_HOME_JOINT)

    # ── 상태 조회 ────────────────────────────────────────────────

    def get_current_pose(self) -> List[float]:
        """현재 EE 태스크 좌표 반환 [x, y, z, u, v, w]."""
        return self.indy.get_control_state()['p']
    
    def get_opstate(self) -> str:
        state_value = self.indy.get_robot_data()['op_state']
        return _OPSTATE_MAP.get(state_value, str(state_value))

    # ── 이동 명령 ─────────────────────────────────────────────────

    def move_to_home(self, vel_ratio: int = 10, acc_ratio: int = 10, timeout: float = 60.0):
        """홈 포지션으로 movej 이동."""
        log.info("홈 포지션 이동 중...")
        movej_and_wait(self.indy, self.home_joint,
                       vel_ratio=vel_ratio, acc_ratio=acc_ratio, timeout=timeout)
        log.success("홈 포지션 도착")

    def movej_to_pose(self, offset: List[float],
                      vel_ratio: int = 10, acc_ratio: int = 10,
                      timeout: float = 60.0) -> List[float]:
        """
        홈 기준 오프셋 조인트 이동 (movej).

        Parameters
        ----------
        offset : List[float] – 홈 대비 각 조인트 오프셋 (deg)

        Returns
        -------
        현재 EE 태스크 좌표 [x, y, z, u, v, w]
        """
        target_joints = [h + o for h, o in zip(self.home_joint, offset)]
        movej_and_wait(self.indy, target_joints,
                       vel_ratio=vel_ratio, acc_ratio=acc_ratio, timeout=timeout)
        time.sleep(0.5)
        return self.get_current_pose()

    def movel_to_pose(self, target_pos: List[float],
                      vel_ratio: int = 10, acc_ratio: int = 10,
                      timeout: float = 60.0) -> List[float]:
        """
        절대 좌표로 movel 이동.

        Parameters
        ----------
        target_pos : List[float] – 목표 EE 태스크 좌표 [x, y, z, u, v, w]

        Returns
        -------
        도달 후 실제 EE 태스크 좌표 [x, y, z, u, v, w]
        """
        log.info(f"Moving to pose: {target_pos}")
        movel_and_wait(self.indy, target_pos,
                       vel_ratio=vel_ratio, acc_ratio=acc_ratio, timeout=timeout)
        log.success("Reached")
        return self.get_current_pose()

    def movel_relative_to_pose(self, target_pos: List[float],
                               vel_ratio: int = 10, acc_ratio: int = 10,
                               timeout: float = 60.0) -> List[float]:
        """
        현재 로봇 위치 기준 상대위치 ttarget만큼 movel 이동.

        Parameters
        ----------
        target_pos : List[float] – 상대 이동량 [x, y, z, u, v, w]

        Returns
        -------
        도달 후 실제 EE 태스크 좌표 [x, y, z, u, v, w]
        """
        log.info(f"Moving to pose: {target_pos}")
        movel_relative_and_wait(self.indy, target_pos,
                                vel_ratio=vel_ratio, acc_ratio=acc_ratio, timeout=timeout)
        log.success("Reached")
        return self.get_current_pose()

    def run_from_json(self, json_path: str,
                      vel_ratio: int = 10, acc_ratio: int = 10, timeout: float = 60.0):
        """
        robot_pose.json을 읽어 순서대로 movel 이동.

        Parameters
        ----------
        json_path : str – pose JSON 파일 경로
        """
        log.info(f"JSON 포즈 시퀀스 실행: {json_path}")
        movel_from_json(self.indy, json_path,
                        vel_ratio=vel_ratio, acc_ratio=acc_ratio, timeout=timeout)

    # ── 직접교시 활성화 / 비활성화 ─────────────────────────────────
    def run_direct_teaching(self):
        self.indy.set_direct_teaching(enable=True)
        
    def exit_direct_teaching(self):
        self.indy.set_direct_teaching(enable=False)

    # def run_simulation_mode(self):
    #     if not self.indy.get_robot_data()['sim_mode']:
    #         self.indy.set_simulation_mode(enable=True)
    #         log.info("시뮬레이션 모드로 변경합니다.")
    #     else:
    #         log.info("이미 시뮬레이션 모드입니다.")

    # def exit_simulation_mode(self):
    #     if self.indy.get_robot_data()['sim_mode']:
    #         self.indy.set_simulation_mode(enable=False)
    #         log.info("실제 로봇 모드로 변경합니다.")
    #     else:
    #         log.info("이미 실제 모드입니다.")

    def robot_recovery(self):
        self.indy.recover()

    # ── 키보드 조그 ─────────────────────────────────────────────────
    def keyboard_jog(self, linear_mm: float = None, angular_deg: float = None,
                    vel_ratio: int = 10, acc_ratio: int = 10) -> str:

        lm = linear_mm   if linear_mm   is not None else self.JOG_LINEAR_MM
        ad = angular_deg if angular_deg is not None else self.JOG_ANGULAR_DEG

        X, Y, Z, U, V, W = 0, 1, 2, 3, 4, 5

        KEY_MAP = {
            '\x1b[D': (Y, +lm, "← +Y (Left)"),
            '\x1b[C': (Y, -lm, "→ -Y (Right)"),
            '\x1b[A': (X, -lm, "↑ -X (Upward)"),
            '\x1b[B': (X, +lm, "↓ +X (Downward)"),

            '8': (Z, +lm, "8  +Z (Forward)"),
            '2': (Z, -lm, "2  -Z (Backward)"),

            '7': (U, -ad, "7  -Rx"),
            '9': (U, +ad, "9  +Rx"),
            '4': (V, -ad, "4  -Ry"),
            '6': (V, +ad, "6  +Ry"),
            '1': (W, -ad, "1  -Rz"),
            '3': (W, +ad, "3  +Rz"),
        }

        log.info(
            "Keyboard Jog 모드 시작\n"
            "  ⚠ NumLock ON 상태에서 사용\n"
            "  방향키 : ← +Y  → -Y  ↑ -X  ↓ +X\n"
            "  숫자키 : 8/2=±Z  7/9=±Rx  4/6=±Ry  1/3=±Rz\n"
            "  종료   : q / ESC / Ctrl+C"
        )

        try:
            import tty, termios
            _unix = True
        except ImportError:
            _unix = False

        fd = None
        old_settings = None
        _original_sigint = signal.getsignal(signal.SIGINT)

        def _sigint_handler(signum, frame):
            if _unix and old_settings is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass
            signal.signal(signal.SIGINT, _original_sigint)
            log.info("Keyboard Jog: Ctrl+C → 프로세스 종료")
            sys.exit(0)

        signal.signal(signal.SIGINT, _sigint_handler)

        _key_queue: queue.Queue = queue.Queue()
        _stop_reader = threading.Event()

        # Windows
        if not _unix:
            import msvcrt

            _WIN_ARROW_MAP = {
                'H': '\x1b[A',
                'P': '\x1b[B',
                'K': '\x1b[D',
                'M': '\x1b[C',
            }

            def _reader_win():
                while not _stop_reader.is_set():
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()

                        if ch in ('\x00', '\xe0'):
                            code = msvcrt.getwch()
                            ch = _WIN_ARROW_MAP.get(code, '')
                        if ch:
                            _key_queue.put(ch)
                    else:
                        time.sleep(0.005)

            _reader_thread = threading.Thread(target=_reader_win, daemon=True)

            def _set_raw(): pass
            def _restore(): pass

        else:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)

            def _reader_unix():
                import select as _sel
                while not _stop_reader.is_set():
                    try:
                        r, _, _ = _sel.select([sys.stdin], [], [], 0.05)
                        if not r:
                            continue

                        ch = os.read(fd, 1).decode('utf-8', errors='replace')

                        if ch == '\x1b':
                            r2, _, _ = _sel.select([sys.stdin], [], [], 0.02)
                            if r2:
                                ch += os.read(fd, 1).decode('utf-8', errors='replace')
                                r3, _, _ = _sel.select([sys.stdin], [], [], 0.02)
                                if r3:
                                    ch += os.read(fd, 1).decode('utf-8', errors='replace')

                        _key_queue.put(ch)
                    except Exception:
                        break

            _reader_thread = threading.Thread(target=_reader_unix, daemon=True)

            def _set_raw():
                tty.setraw(fd)

            def _restore():
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        current_key = ''
        last_send_t = 0.0
        result = 'quit'

        _set_raw()
        _reader_thread.start()

        try:
            while True:
                now = time.monotonic()

                while True:
                    try:
                        key = _key_queue.get_nowait()
                    except queue.Empty:
                        break

                    if key in ('q', 'Q', '\x1b'):
                        log.info("Keyboard Jog 종료.")
                        result = 'quit'
                        _stop_reader.set()
                        raise _JogExit()

                    if key in KEY_MAP:
                        log.info(f"Jog  {KEY_MAP[key][2]}")
                        current_key = key

                if current_key:
                    axis, delta, _ = KEY_MAP[current_key]
                    offset = [0.0] * 6
                    offset[axis] = delta

                    try:
                        movel_relative(self.indy, offset,
                                    vel_ratio=vel_ratio, acc_ratio=acc_ratio)
                    except Exception as e:
                        log.error(f"Jog 이동 실패: {e}")
                        _stop_reader.set()
                        raise _JogExit()

                    current_key = ''

                time.sleep(0.005)

        except _JogExit:
            pass

        finally:
            _stop_reader.set()
            _restore()
            signal.signal(signal.SIGINT, _original_sigint)

        return result

class _JogExit(Exception):
    """keyboard_jog 루프 탈출용 내부 예외."""
    pass

# ── 단독 실행 엔트리포인트 ───────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="RobotController 단독 실행",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--ip',   type=str, default='192.168.0.137',
                        help='로봇 IP 주소 (default: 192.168.0.137)')
    parser.add_argument('--json', type=str, default=None,
                        help='실행할 robot_pose JSON 파일 경로\n'
                             '예) dataset/poses/robot_pose_broad.json')
    parser.add_argument('--pose', type=float, nargs=6, default=None,
                        metavar=('X', 'Y', 'Z', 'U', 'V', 'W'),
                        help='단일 movel 목표 좌표 (6개 값)\n'
                             '예) --pose 597 -143 225 79 -176 80')
    parser.add_argument('--jog',  action='store_true',
                        help='키보드 조그 모드 실행')
    parser.add_argument('--vel',  type=int, default=10,
                        help='속도 비율 (default: 10)')
    parser.add_argument('--acc',  type=int, default=10,
                        help='가속도 비율 (default: 10)')
    return parser.parse_args()

if __name__ == "__main__":
    from src.utils.logger import configure_logging
    configure_logging()
    args = _parse_args()

    log.info(f"연결 중 – Robot IP: {args.ip}")
    robot = RobotController(robot_ip=args.ip)

    robot.move_to_home(vel_ratio=args.vel, acc_ratio=args.acc)

    if args.json:
        robot.run_from_json(args.json, vel_ratio=args.vel, acc_ratio=args.acc)
    elif args.pose:
        pose = robot.movel_to_pose(args.pose, vel_ratio=args.vel, acc_ratio=args.acc)
        log.info(f"도달 좌표: {pose}")
    elif args.jog:
        robot.keyboard_jog(vel_ratio=args.vel, acc_ratio=args.acc)

    robot.move_to_home(vel_ratio=args.vel, acc_ratio=args.acc)
    log.success("controller.py 단독 실행 종료")