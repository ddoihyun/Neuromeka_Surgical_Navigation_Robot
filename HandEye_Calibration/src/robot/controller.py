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

from neuromeka import IndyDCP3

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

    def run_direct_teaching(self):
        self.indy.set_direct_teaching(enable=True)
        
    def exit_direct_teaching(self):
        self.indy.set_direct_teaching(enable=False)

    def run_simulation_mode(self):
        if not self.indy.get_robot_data()['sim_mode']:
            self.indy.set_simulation_mode(enable=True)
            log.info("시뮬레이션 모드로 변경합니다.")
        else:
            log.info("이미 시뮬레이션 모드입니다.")

    def exit_simulation_mode(self):
        if self.indy.get_robot_data()['sim_mode']:
            self.indy.set_simulation_mode(enable=False)
            log.info("실제 로봇 모드로 변경합니다.")
        else:
            log.info("이미 실제 모드입니다.")

    def robot_recovery(self):
        self.indy.recover()
    
    # ── 키보드 조그 ───────────────────────────────────────────────

    def keyboard_jog(self,
                     linear_mm: float = None,
                     angular_deg: float = None,
                     vel_ratio: int = 10,
                     acc_ratio: int = 10) -> str:
        """
        키보드 입력으로 현재 위치 기준 연속 상대 이동.

        구조
        ----
        · stdin 읽기 전용 데몬 스레드가 blocking read 로 키를 읽어 큐에 넣는다.
          → select(0) + read(1) 방식의 시퀀스 오염 문제 없음.
        · 메인 루프는 큐를 소비해 current_key 를 갱신하고,
          JOG_SEND_INTERVAL 주기로 movel_relative(OVERRIDE, no-wait) 를 전송한다.
        · IDLE_RESET_SEC 동안 새 키 입력이 없으면 키를 뗐다고 간주해 전송을 멈춘다.

        키 맵핑
        -------
        방향키
          ←   : -Y (좌)         →   : +Y (우)
          ↑   : +X (상)         ↓   : -X (하)
        숫자키
          8   : +Z (전)         2   : -Z (후)
          7   : -Rx             9   : +Rx
          4   : -Ry             6   : +Ry
          1   : -Rz             3   : +Rz
        종료
          q / Q / ESC : 조그 종료 → 'quit' 반환
          Ctrl+C      : 터미널 복원 후 프로세스 즉시 종료 (sys.exit)

        Parameters
        ----------
        linear_mm   : float – 1회 선형 이동량 (mm).  None → JOG_LINEAR_MM
        angular_deg : float – 1회 회전 이동량 (deg). None → JOG_ANGULAR_DEG
        vel_ratio   : int   – 속도 비율
        acc_ratio   : int   – 가속도 비율

        Returns
        -------
        'quit' : q / ESC 로 정상 종료
        """
        lm = linear_mm   if linear_mm   is not None else self.JOG_LINEAR_MM
        ad = angular_deg if angular_deg is not None else self.JOG_ANGULAR_DEG

        # X, Y, Z, U, V, W = 0, 1, 2, 3, 4, 5
        Z, Y, X, U, V, W = 0, 1, 2, 3, 4, 5

        KEY_MAP = {
            # 방향키 (XY 평면 + Z)
            '\x1b[D': (Y, -lm, "← -Y"),   # 왼쪽
            '\x1b[C': (Y, +lm, "→ +Y"),   # 오른쪽
            '\x1b[A': (X, +lm, "↑ +X"),   # 위
            '\x1b[B': (X, -lm, "↓ -X"),   # 아래

            # 숫자키 (전후 이동)
            '8': (Z, +lm, "8  +Z (forward)"),
            '2': (Z, -lm, "2  -Z (backward)"),

            # 회전 (그대로 유지)
            '7':      (U, -ad, "7  -Rx"),
            '9':      (U, +ad, "9  +Rx"),
            '4':      (V, -ad, "4  -Ry"),
            '6':      (V, +ad, "6  +Ry"),
            '1':      (W, -ad, "1  -Rz"),
            '3':      (W, +ad, "3  +Rz"),
        }

        log.info(
            "Keyboard Jog 모드 시작\n"
            "  방향키 : ← -Y  → +Y  ↑ +X  ↓ -X\n"
            "  숫자키 : 8/2=±Z  7/9=±Rx  4/6=±Ry  1/3=±Rz\n"
            "  종료   : q / ESC / Ctrl+C"
        )

        # ── 플랫폼 분기 ──────────────────────────────────────────────────
        try:
            import tty, termios
            _unix = True
        except ImportError:
            _unix = False

        fd           = None
        old_settings = None
        _original_sigint = signal.getsignal(signal.SIGINT)

        # ── Ctrl+C 핸들러: raw 모드 복원 후 즉시 프로세스 종료 ──────────
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

        # ── stdin 읽기 스레드 ─────────────────────────────────────────────
        #
        # 핵심 설계
        # ---------
        # tty.setraw() 상태에서 os.read(fd, N) 을 blocking으로 호출한다.
        # VMIN=1, VTIME=0 이므로 1바이트 이상 들어오면 즉시 반환.
        # 방향키(\x1b[X) 는 3바이트가 연속으로 들어오므로
        # \x1b 를 읽은 직후 짧은 timeout(20ms) 으로 나머지를 drain 한다.
        # → select(0)+read(1) 루프와 달리 키 repeat 으로 버퍼에 쌓인
        #   여러 시퀀스를 순서대로 정확히 파싱할 수 있다.
        #
        _key_queue: queue.Queue = queue.Queue()
        _stop_reader = threading.Event()

        if _unix:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)

            def _reader_unix():
                import select as _sel
                while not _stop_reader.is_set():
                    try:
                        # blocking: 최대 50ms 대기 (루프 종료 감지용)
                        r, _, _ = _sel.select([sys.stdin], [], [], 0.05)
                        if not r:
                            continue

                        ch = os.read(fd, 1).decode('utf-8', errors='replace')

                        if ch == '\x1b':
                            # 나머지 시퀀스 바이트를 20ms 안에 읽는다
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

        else:
            import msvcrt
            _WIN_MAP = {'K': '\x1b[D', 'M': '\x1b[C',
                        'H': '\x1b[A', 'P': '\x1b[B'}

            def _reader_win():
                while not _stop_reader.is_set():
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch in ('\x00', '\xe0'):
                            ch = _WIN_MAP.get(msvcrt.getwch(), '')
                        if ch:
                            _key_queue.put(ch)
                    else:
                        time.sleep(0.005)

            _reader_thread = threading.Thread(target=_reader_win, daemon=True)

            def _set_raw():  pass
            def _restore():  pass

        # ── 메인 조그 루프 ────────────────────────────────────────────────
        #
        # · current_key    : 현재 눌려 있다고 간주하는 키 ('' = 없음)
        # · last_input_t   : 마지막으로 유효 키가 들어온 시각
        # · IDLE_RESET_SEC : 이 시간 동안 새 키 없으면 키 업으로 간주
        #                    key repeat 주기(보통 30~50ms)보다 넉넉하게 설정
        # · last_send_t    : 마지막으로 movel_relative 를 전송한 시각
        #
        IDLE_RESET_SEC = 0.08   # 150ms 무입력 → 키 업 간주

        current_key  = ''
        last_input_t = time.monotonic()
        last_send_t  = 0.0
        result       = 'quit'

        _set_raw()
        _reader_thread.start()

        try:
            while True:
                now = time.monotonic()

                # 1) 큐에서 키 소비 (쌓인 것 모두 처리)
                while True:
                    try:
                        key = _key_queue.get_nowait()
                    except queue.Empty:
                        break

                    # 종료: q / Q / ESC
                    if key in ('q', 'Q', '\x1b'):
                        log.info("Keyboard Jog 종료.")
                        result = 'quit'
                        _stop_reader.set()
                        # finally 블록으로 점프
                        raise _JogExit()

                    # 유효 이동 키만 반영
                    if key in KEY_MAP:
                        if key != current_key:
                            log.info(f"Jog  {KEY_MAP[key][2]}")
                        current_key  = key
                        last_input_t = now   # 입력 시각 갱신

                # 2) 무입력 지속 → 키 업 간주
                if current_key and (now - last_input_t) > IDLE_RESET_SEC:
                    current_key = ''

                # 3) 주기적 이동 명령 전송
                if current_key and (now - last_send_t) >= self.JOG_SEND_INTERVAL:
                    axis, delta, _ = KEY_MAP[current_key]
                    offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    offset[axis] = delta
                    try:
                        movel_relative(self.indy, offset,
                                       vel_ratio=vel_ratio, acc_ratio=acc_ratio)
                    except Exception as e:
                        log.error(f"Jog 이동 실패: {e}")
                        _stop_reader.set()
                        raise _JogExit()
                    last_send_t = now

                # 4) CPU 양보
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