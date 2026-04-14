"""
main.py – 로봇 State Machine 시스템
══════════════════════════════════════════════════════════════════════════════

[State 정의]
    NOT_READY       : 초기 로봇/NDI 연결 시도
    IDLE            : 대기 상태 (키보드 또는 action.json 명령 대기)
    TRACKING        : NDI 마커 실시간 트래킹
    CALIBRATION     : 핸드-아이 캘리브레이션 데이터 수집
    READY_TO_NAV    : 내비게이션 목표 계산 → 사용자 Enter 확인 대기
    ROBOT_MOVING    : 로봇 이동 실행 중
    DIRECT_TEACHING : 다이렉트 티칭 모드
    STOP            : 긴급/명령 정지 → recovery → IDLE 자동 복귀

[action.json 지원 명령]
    tracking        → TRACKING
    calibration     → CALIBRATION
    navigation      → READY_TO_NAV  (offset 포함)
    move            → ROBOT_MOVING  (offset 포함)
    stop            → STOP
    direct_teaching → DIRECT_TEACHING
    null            → 무시

[설계 원칙]
    - threading + Event 기반 (asyncio 미사용)
    - 모든 State는 진입/이탈 시 로그 출력
    - stop_event로 모든 blocking 작업 중단 가능
    - StateManager 클래스로 상태 전이 캡슐화
    - 각 state 핸들러 함수 분리 → 확장 용이
    - thread-safe 설계 (Lock 사용)

[수정 사항 — v2]
    1. Interrupt 정책: stop 명령만 허용 (다른 action 수행 중 다른 명령 차단)
       - ActionDispatcher.on_action_received: 비-IDLE 상태에서 stop 이외 명령 무시
       - ROBOT_MOVING 진입 전 반드시 OpState == IDLE(5) 확인
    2. OpState 기반 로봇 상태 모니터링 스레드 추가 (_RobotStateMonitor)
       - VIOLATION / COLLISION 등 외부 이상 상태 감지 시 stop_event.set() → STOP 전이
    3. 매 state transition마다 입력 버퍼 초기화
       - StateManager.transition() 호출 시 _input_flush_fn 콜백으로 버퍼 비움
       - dispatcher.pop_pending() 도 함께 호출하여 누적 명령 제거
"""

import sys
import os
import time
import json
import threading
import signal

from pathlib import Path
from typing import Optional, Callable

# ─── 프로젝트 내부 모듈 ────────────────────────────────────────────────────
# 실제 환경에서는 아래 import를 사용한다.
# 이 파일에는 mock 구현을 포함하므로, 실제 모듈이 있으면 import가 우선한다.
try:
    import src.ndi.tracker as ndi
    from src.robot.controller import RobotController
    from src.calib.calibration import HandEyeCalibration
    from src.calib.navigator import Navigator
    import src.utils.io as io_utils
    from src.utils.logger import get_logger
    _MOCK_MODE = False
except ImportError:
    # ── Mock 모드: 실제 하드웨어 없이 State Machine 동작 검증용 ──────────────
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _MOCK_MODE = True

    # ── 최소 mock 구현 ────────────────────────────────────────────────────
    class _MockLogger:
        """표준 logging을 감싸는 mock logger (success/section 메서드 추가)."""
        def __init__(self, name: str):
            self._log = logging.getLogger(name)
        def info(self, msg, *a, **kw):    self._log.info(msg, *a, **kw)
        def warning(self, msg, *a, **kw): self._log.warning(msg, *a, **kw)
        def error(self, msg, *a, **kw):   self._log.error(msg, *a, **kw)
        def debug(self, msg, *a, **kw):   self._log.debug(msg, *a, **kw)
        def success(self, msg, *a, **kw):
            self._log.info(f"✅ {msg}", *a, **kw)
        def section(self, title: str, **kw):
            sep = "─" * 60
            self._log.info(f"\n{sep}\n  {title}\n{sep}")

    def get_logger(name: str) -> _MockLogger:
        return _MockLogger(name)

    class _MockConfig:
        """config.json 없이 동작하기 위한 기본값 제공."""
        @staticmethod
        def load_config(path="config.json", base_dir=None) -> dict:
            return {
                "ndi":        {"hostname": "192.168.1.1", "tools": [], "rom_dir": None,
                               "encrypted": False, "cipher": ""},
                "robot":      {"ip": "192.168.1.2"},
                "dataset":    {"dataset_root": "./dataset",
                               "robot_pose_file": "dataset/poses/robot_pose.json"},
                "calibration":{"duration_sec": 2.0, "samples": 5},
                "navigation": {"ttool": "", "x_offset": 0.0,
                               "y_offset": 0.0, "z_offset": 0.0},
                "logging":    {"level": "DEBUG", "emoji": True},
            }
        @staticmethod
        def apply_navigation_offset_to_config(config: dict, offset: dict) -> dict:
            nav = config.setdefault("navigation", {})
            nav["x_offset"] = float(offset.get("x", nav.get("x_offset", 0.0)))
            nav["y_offset"] = float(offset.get("y", nav.get("y_offset", 0.0)))
            nav["z_offset"] = float(offset.get("z", nav.get("z_offset", 0.0)))
            return config
        @staticmethod
        def get_calibration_filepaths(robot_pose_file, dataset_root):
            return {
                "csv":  os.path.join(dataset_root, "calibration", "calibration_data.csv"),
                "json": os.path.join(dataset_root, "results",     "calibration_result.json"),
                "png":  os.path.join(dataset_root, "results",     "calibration_result.png"),
            }
        @staticmethod
        def delete_calibration_csv(robot_pose_file, dataset_root):
            return os.path.join(dataset_root, "calibration", "calibration_data.csv")
        @staticmethod
        def save_data_to_csv(*args, **kwargs):
            pass

    class _MockActionWatcher:
        def __init__(self, path, callback): pass
        def start(self):  pass
        def stop(self):   pass

    class _MockRobotController:
        def move_to_home(self):
            time.sleep(0.2)
        def movel_to_pose(self, pose, **kw):
            time.sleep(0.3)
        def movel_relative_to_pose(self, offset, **kw):
            time.sleep(0.2)
        def get_current_pose(self):
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        def run_direct_teaching(self):
            pass
        def exit_direct_teaching(self):
            pass
        def keyboard_jog(self, **kw):
            pass
        def robot_recovery(self):
            pass
        def get_opstate(self):
            # Mock: 항상 IDLE 반환
            return "IDLE"
        class indy:
            @staticmethod
            def stop_motion():   pass
            @staticmethod
            def get_control_state(): return {}

    class _MockNavigator:
        method = "mock"
        unit   = "mm"
        def __init__(self, calib_path):
            pass
        def compute(self, *args):
            return {"x": 100.0, "y": 200.0, "z": -1500.0,
                    "u": 0.0, "v": 0.0, "w": 0.0}

    class _MockApi:
        """handle_tracking 내부에서 직접 사용하는 mock NDI api."""
        def startTracking(self):
            pass
        def stopTracking(self):
            pass
        def getTrackingDataBX2(self):
            time.sleep(0.05)
            return []  # 빈 목록 -> 트래킹 루프는 정상 동작

    class _MockNDI:
        @staticmethod
        def connect_and_setup(hostname, tools, rom_dir, encrypted, cipher):
            """handle_tracking에서 직접 호출. (api, enabled_tools) 반환."""
            return _MockApi(), []

        @staticmethod
        def connect_and_setup_calibration(*args):
            return None

        @staticmethod
        def connect_and_setup_navigation(*args):
            return _MockApi(), None

        @staticmethod
        def run_tracking(*args, **kwargs):
            stop_event = kwargs.get("stop_event")
            for _ in range(100):
                if stop_event and stop_event.is_set():
                    return
                time.sleep(0.1)

        @staticmethod
        def extract_full_data_dict(td, timestamp=None):
            """handle_tracking 트래킹 루프에서 호출."""
            return {
                "position":   {"x": 0.0, "y": 0.0, "z": 0.0},
                "quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                "error_mm": 0.0, "tool_handle": "01",
                "timestamp": time.time(),
                "status_raw": "0x0000", "error_code": 0,
                "error_str": "OK", "missing": False, "face": 0,
            }

        @staticmethod
        def is_missing_frame(full_data):
            return full_data.get("missing", False)

        @staticmethod
        def print_tracking_data(data):
            pass

        @staticmethod
        def get_latest_valid_pose(*args, **kwargs):
            return {"q0": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
                    "tx": 0.0, "ty": 0.0, "tz": 0.0, "error": 0.1}, "ok"

        @staticmethod
        def collect_marker_samples(*args):
            return [{"timestamp": time.time(),
                     "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                     "quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                     "error_mm": 0.1}]

    class _MockHandEyeCalibration:
        def __init__(self, csv_path): pass
        def run(self): pass

    # ── mock을 실제 이름으로 바인딩 ─────────────────────────────────────────
    ndi              = _MockNDI()
    RobotController  = _MockRobotController
    HandEyeCalibration = _MockHandEyeCalibration
    Navigator        = _MockNavigator
    io_utils         = _MockConfig()
    io_utils.ActionWatcher = _MockActionWatcher


log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. State 정의
# ══════════════════════════════════════════════════════════════════════════════

class State:
    """전역 State 상수 모음."""
    NOT_READY       = "NOT_READY"
    IDLE            = "IDLE"
    TRACKING        = "TRACKING"
    CALIBRATION     = "CALIBRATION"
    READY_TO_NAV    = "READY_TO_NAV"
    ROBOT_MOVING    = "ROBOT_MOVING"
    DIRECT_TEACHING = "DIRECT_TEACHING"
    STOP            = "STOP"
    EXIT            = "EXIT"

    # 유효한 State 집합 (외부 검증용)
    ALL = frozenset([
        NOT_READY, IDLE, TRACKING, CALIBRATION,
        READY_TO_NAV, ROBOT_MOVING, DIRECT_TEACHING, STOP, EXIT,
    ])

    # action 수신 시 interrupt(stop만 허용)가 적용되는 active 상태 집합
    # 이 상태에서는 stop 이외의 명령이 수신되어도 무시한다.
    ACTIVE = frozenset([
        TRACKING, CALIBRATION, READY_TO_NAV, ROBOT_MOVING, DIRECT_TEACHING,
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 2. StateManager — 상태 전이 관리 (thread-safe)
# ══════════════════════════════════════════════════════════════════════════════

class StateManager:
    """
    State 전이를 캡슐화하는 클래스.

    - 모든 state 변경은 transition()을 통해야 한다.
    - 진입/이탈 로그를 자동으로 출력한다.
    - on_enter / on_exit 콜백 등록을 지원한다 (확장 포인트).
    - [수정] transition() 호출 시 _input_flush_fn 을 실행해 입력 버퍼를 비운다.
    """

    def __init__(self, initial_state: str = State.NOT_READY) -> None:
        self._state      = initial_state
        self._prev_state = initial_state
        self._lock       = threading.Lock()

        # 콜백 등록 테이블: {state: {"enter": fn, "exit": fn}}
        self._callbacks: dict = {}

        # [수정] state 전이 시 입력 버퍼를 비우기 위한 콜백
        # main()에서 dispatcher.pop_pending 등을 등록한다.
        self._input_flush_fn: Optional[Callable] = None

        log.info(f"[StateManager] 초기화 완료. 초기 state: {self._state}")

    # ── 속성 접근자 ─────────────────────────────────────────────────────────

    @property
    def current(self) -> str:
        with self._lock:
            return self._state

    @property
    def previous(self) -> str:
        with self._lock:
            return self._prev_state

    # ── 입력 플러시 콜백 등록 ────────────────────────────────────────────────

    def set_input_flush_fn(self, fn: Callable) -> None:
        """
        state 전이 시마다 호출될 입력 버퍼 초기화 함수 등록.

        Parameters
        ----------
        fn : Callable – 인수 없는 함수. dispatcher.pop_pending 등을 wrapping해 전달.
        """
        self._input_flush_fn = fn

    # ── 전이 ────────────────────────────────────────────────────────────────

    def transition(self, new_state: str) -> bool:
        """
        State를 전이한다.

        Parameters
        ----------
        new_state : str  – 전이 대상 State

        Returns
        -------
        True  : 전이 성공 (상태가 실제로 변경됨)
        False : 동일 상태 → 전이 없음

        [수정]
        - 전이 성공 시 _input_flush_fn 호출 → 입력 버퍼 초기화
        """
        if new_state not in State.ALL:
            log.error(f"[StateManager] 알 수 없는 state: '{new_state}' — 무시")
            return False

        with self._lock:
            if self._state == new_state:
                return False
            old_state        = self._state
            self._prev_state = old_state
            self._state      = new_state

        # ── 이탈 로그 + 콜백 ────────────────────────────────────────────
        log.info(f"[STATE] ─── {old_state}  →  {new_state} ───")
        self._fire_callback(old_state, "exit")

        # ── [수정] 입력 버퍼 초기화 ─────────────────────────────────────
        # state 전이마다 누적된 음성/키보드 입력을 제거한다.
        if self._input_flush_fn is not None:
            try:
                self._input_flush_fn()
            except Exception as e:
                log.warning(f"[StateManager] 입력 버퍼 초기화 중 오류: {e}")

        # ── 진입 콜백 ────────────────────────────────────────────────────
        self._fire_callback(new_state, "enter")

        return True

    # ── 콜백 등록 / 실행 ────────────────────────────────────────────────────

    def register_on_enter(self, state: str, fn: Callable) -> None:
        """특정 state에 진입할 때 호출될 콜백 등록."""
        self._callbacks.setdefault(state, {})["enter"] = fn

    def register_on_exit(self, state: str, fn: Callable) -> None:
        """특정 state를 이탈할 때 호출될 콜백 등록."""
        self._callbacks.setdefault(state, {})["exit"] = fn

    def _fire_callback(self, state: str, event: str) -> None:
        fn = self._callbacks.get(state, {}).get(event)
        if fn:
            try:
                fn()
            except Exception as e:
                log.warning(f"[StateManager] 콜백 오류 ({state}/{event}): {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. NonBlockingInput — 메인 루프 블로킹 방지
# ══════════════════════════════════════════════════════════════════════════════

class NonBlockingInput:
    """
    별도 daemon 스레드에서 input()을 실행해
    메인 루프가 블로킹되지 않도록 하는 헬퍼.

    사용 패턴:
        nbi = NonBlockingInput("Select: ")
        nbi.start()
        while True:
            val = nbi.get()   # None이면 아직 입력 없음
            if val is not None:
                ...

    [수정]
    - flush() 메서드 추가: state 전이 시 버퍼를 즉시 소비해 누적 입력 방지.
    """

    def __init__(self, prompt: str = "") -> None:
        self._prompt = prompt
        self._result: Optional[str] = None
        self._ready  = False
        self._lock   = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """입력 대기 스레드 시작."""
        with self._lock:
            self._result = None
            self._ready  = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            val = input(self._prompt)
        except (EOFError, OSError):
            val = ""
        with self._lock:
            self._result = val
            self._ready  = True

    def get(self) -> Optional[str]:
        """
        입력이 완료됐으면 문자열을, 아직이면 None을 반환.
        한 번 반환한 이후에도 동일 값을 계속 반환하므로
        읽은 뒤 새 start()를 호출해야 한다.
        """
        with self._lock:
            return self._result if self._ready else None

    def flush(self) -> None:
        """
        [수정] 버퍼에 누적된 입력을 버린다.
        state 전이 직후 호출해 이전 state의 입력이 다음 state로 흘러들지 않도록 한다.
        """
        with self._lock:
            self._result = None
            self._ready  = False


# ══════════════════════════════════════════════════════════════════════════════
# 4. ActionDispatcher — action.json 이벤트 → State 매핑
# ══════════════════════════════════════════════════════════════════════════════

class ActionDispatcher:
    """
    action.json으로 수신된 명령 데이터를 파싱해 전이할 State를 결정한다.

    - thread-safe pending_action 관리
    - stop 명령 수신 시 stop_event 즉시 set
    - navigation/move 명령의 offset을 config에 반영

    [수정]
    - on_action_received: 현재 state가 ACTIVE 집합에 속하면 stop 이외 명령을 무시한다.
      → interrupt 정책: 동작 중에는 stop 명령만 허용
    - _current_state_getter: 현재 state를 조회하는 callable (main에서 주입)
    """

    # 단순 1:1 매핑
    _ACTION_MAP: dict = {
        "tracking":        State.TRACKING,
        "calibration":     State.CALIBRATION,
        "navigation":      State.READY_TO_NAV,
        "move":            State.ROBOT_MOVING,
        "stop":            State.STOP,
        "direct_teaching": State.DIRECT_TEACHING,
    }

    def __init__(
        self,
        config: dict,
        stop_event: threading.Event,
        robot_controller_getter: Callable,
        current_state_getter: Callable,
    ) -> None:
        """
        Parameters
        ----------
        config                 : 공유 config dict (navigation offset 반영용)
        stop_event             : 전역 중단 이벤트
        robot_controller_getter: 현재 RobotController 인스턴스를 반환하는 callable
        current_state_getter   : 현재 State 문자열을 반환하는 callable
                                 [수정] interrupt 정책 적용을 위해 추가
        """
        self._config       = config
        self._stop_event   = stop_event
        self._get_rc       = robot_controller_getter
        self._get_state    = current_state_getter  # [수정]

        self._lock                              = threading.Lock()
        self._pending_action: Optional[dict]    = None
        self._pending_move:   Optional[dict]    = None

    # ── ActionWatcher 콜백 (별도 스레드에서 호출) ───────────────────────────

    def on_action_received(self, action_data: dict) -> None:
        """
        io.ActionWatcher 콜백.
        action.json 변경 시마다 호출된다.

        [수정] interrupt 정책:
        - 현재 state가 ACTIVE 집합에 속할 때는 stop 이외의 명령을 무시한다.
        - stop 명령은 어떤 state에서도 즉시 처리한다.
        """
        action = action_data.get("action")
        desc   = action_data.get("description", "")

        log.info(f"[ActionWatcher] 수신: action={action}  desc={desc!r}")

        # stop 명령이면 blocking 루프를 즉시 깨운다 (항상 허용)
        if action == "stop":
            log.warning(f"[ActionWatcher] STOP 수신 — stop_event.set()  desc={desc!r}")
            self._stop_event.set()
            with self._lock:
                self._pending_action = action_data
            return

        # [수정] interrupt 정책: ACTIVE 상태에서는 stop 이외 명령 무시
        current_state = self._get_state()
        if current_state in State.ACTIVE:
            log.warning(
                f"[ActionWatcher] 현재 state={current_state} (ACTIVE) — "
                f"stop 이외 명령 '{action}' 무시 (interrupt 정책)"
            )
            return

        with self._lock:
            self._pending_action = action_data

    # ── 메인 루프에서 호출 ──────────────────────────────────────────────────

    def pop_pending(self) -> Optional[dict]:
        """pending_action을 꺼내고 내부 값을 None으로 초기화."""
        with self._lock:
            action          = self._pending_action
            self._pending_action = None
            return action

    def pop_pending_move(self) -> Optional[dict]:
        """pending_move를 꺼내고 내부 값을 None으로 초기화."""
        with self._lock:
            move           = self._pending_move
            self._pending_move = None
            return move

    def resolve_state(self, action_data: dict) -> Optional[str]:
        """
        action_data를 파싱해 전이할 State 문자열을 반환한다.

        Returns
        -------
        str  : 전이 대상 State
        None : 무시 (null / 인식 불가 명령)
        """
        action = action_data.get("action")

        if action is None or action == "null":
            log.warning(
                f"[Dispatcher] 인식 불가/null 명령 무시: "
                f"{action_data.get('description', '')!r}"
            )
            return None

        # navigation: offset → config에 반영 후 READY_TO_NAV
        if action == "navigation":
            offset = action_data.get("offset", {})
            nav = self._config.setdefault("navigation", {})

            # action에서 들어온 offset이 모두 0인 경우: config 기존값 유지
            # action에서 들어온 offset에 0이 아닌 값이 있으면: config에 덮어씀
            ax = float(offset.get("x", 0.0))
            ay = float(offset.get("y", 0.0))
            az = float(offset.get("z", 0.0))

            if ax != 0.0 or ay != 0.0 or az != 0.0:
                nav["x_offset"] = ax
                nav["y_offset"] = ay
                nav["z_offset"] = az
                log.info(
                    f"[Dispatcher] Navigation offset (action) → "
                    f"x={ax}, y={ay}, z={az}"
                )
            else:
                log.info(
                    f"[Dispatcher] Navigation offset 모두 0 → config 기존값 유지 "
                    f"x={nav.get('x_offset', 0.0)}, "
                    f"y={nav.get('y_offset', 0.0)}, "
                    f"z={nav.get('z_offset', 0.0)}"
                )

            return State.READY_TO_NAV

        # move: pending_move 저장 후 ROBOT_MOVING
        if action == "move":
            rc = self._get_rc()
            if rc is None:
                log.warning("[Dispatcher] 'move' 수신했으나 robot_controller 없음 — 무시")
                return None
            with self._lock:
                self._pending_move = action_data
            return State.ROBOT_MOVING

        return self._ACTION_MAP.get(action)

# ══════════════════════════════════════════════════════════════════════════════
# 4-B. _RobotStateMonitor — OpState 기반 이상 상태 감시 스레드
# ══════════════════════════════════════════════════════════════════════════════

# [수정] OpState 이상 감지 대상 문자열 집합
# controller.py get_opstate()가 OpState(value).name 문자열을 반환하므로 이름 기반 비교
_OPSTATE_ERROR_NAMES = frozenset([
    "VIOLATE",        # 2
    "COLLISION",      # 8
    "VIOLATE_HARD",   # 15
    "STOP_AND_OFF",   # 9  – 비상 정지 버튼 등 하드웨어 정지
])

class _RobotStateMonitor:
    """
    [수정] OpState 기반 로봇 이상 상태 모니터링 스레드.

    RobotController.get_opstate() 를 주기적으로 폴링하여
    VIOLATION / COLLISION 등 이상 상태가 감지되면 stop_event.set() 을 호출한다.
    → 메인 루프가 STOP state로 전이해 recovery를 수행한다.

    사용 패턴:
        monitor = _RobotStateMonitor(get_robot_controller, stop_event)
        monitor.start()
        ...
        monitor.stop()
    """

    POLL_INTERVAL: float = 0.2   # 폴링 주기 (초)

    def __init__(
        self,
        robot_controller_getter: Callable,
        stop_event: threading.Event,
        sm_current_getter: Callable,
    ) -> None:
        self._get_rc    = robot_controller_getter
        self._stop_evt  = stop_event
        self._get_state = sm_current_getter
        self._running   = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """모니터링 스레드 시작."""
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="robot-state-monitor"
        )
        self._thread.start()
        log.info("[Monitor] 로봇 OpState 모니터링 시작.")

    def stop(self) -> None:
        """모니터링 스레드 정지."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        log.info("[Monitor] 로봇 OpState 모니터링 종료.")

    def _run(self) -> None:
        while self._running.is_set():
            # STOP / EXIT / NOT_READY 상태에서는 모니터링 비활성
            current = self._get_state()
            if current in (State.STOP, State.EXIT, State.NOT_READY, State.IDLE):
                time.sleep(self.POLL_INTERVAL)
                continue

            rc = self._get_rc()
            if rc is None:
                time.sleep(self.POLL_INTERVAL)
                continue

            try:
                op_name = rc.get_opstate()   # e.g. "IDLE", "MOVING", "COLLISION"
                if op_name in _OPSTATE_ERROR_NAMES:
                    log.error(
                        f"[Monitor] 이상 OpState 감지: {op_name} "
                        f"(현재 state={current}) → stop_event.set()"
                    )
                    self._stop_evt.set()
            except Exception as e:
                # 폴링 중 일시적 통신 오류는 무시하고 계속 감시
                log.warning(f"[Monitor] get_opstate() 오류 (무시): {e}")

            time.sleep(self.POLL_INTERVAL)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Cleanup / Recovery 유틸
# ══════════════════════════════════════════════════════════════════════════════

# ── 전역 NDI API 레퍼런스 ──────────────────────────────────────────────────────
# handle_tracking / handle_navigation 이 api 객체를 여기에 등록한다.
# do_stop_recovery() 는 이 레퍼런스를 통해 stopTracking()을 즉시 호출할 수 있다.
_g_ndi_api_lock = threading.Lock()
_g_ndi_api      = None          # 현재 활성 NDI api 객체 (없으면 None)


def _register_ndi_api(api) -> None:
    """현재 활성 NDI api 객체를 전역에 등록한다."""
    global _g_ndi_api
    with _g_ndi_api_lock:
        _g_ndi_api = api


def _unregister_ndi_api() -> None:
    """전역 NDI api 레퍼런스를 해제한다."""
    global _g_ndi_api
    with _g_ndi_api_lock:
        _g_ndi_api = None


def _stop_active_ndi() -> None:
    """
    전역에 등록된 NDI api가 있으면 api.stopTracking()을 호출한다.

    STOP 명령·Ctrl+C 수신 시 handle_tracking/handle_navigation 내부 루프가
    종료를 감지하기 전에 NDI 트래킹을 즉시 중단하기 위해 사용한다.
    tracker.py 의 api.stopTracking() 을 경유한다.
    """
    with _g_ndi_api_lock:
        api = _g_ndi_api
    if api is None:
        return
    try:
        api.stopTracking()
        log.info("[Cleanup] 전역 NDI stopTracking() 완료.")
    except Exception as e:
        log.warning(f"[Cleanup] 전역 NDI stopTracking() 중 오류: {e}")


def cleanup_ndi(api=None) -> None:
    """
    NDI 트래킹 API를 안전하게 종료하고 전역 레퍼런스도 해제한다.
    api=None 이면 전역에 등록된 api를 사용한다.
    """
    with _g_ndi_api_lock:
        target = api if api is not None else _g_ndi_api

    if target is None:
        return
    try:
        target.stopTracking()
        log.info("[Cleanup] NDI 트래킹 중지 완료.")
    except Exception as e:
        log.warning(f"[Cleanup] NDI 중지 중 오류: {e}")
    finally:
        _unregister_ndi_api()



def stop_robot_motion(robot_controller) -> None:
    """로봇 모션을 즉시 정지한다."""
    if robot_controller is None:
        return
    try:
        robot_controller.indy.stop_motion()
        log.info("[Cleanup] 로봇 모션 정지 완료.")
    except Exception as e:
        log.warning(f"[Cleanup] 로봇 모션 정지 중 오류: {e}")


def do_stop_recovery(robot_controller, prev_state: str) -> None:
    """
    STOP state 진입 시 실행하는 복구 루틴.

    실행 순서
    ---------
    1) [NDI]   활성 트래킹 즉시 중단 (api.stopTracking)
               tracker.py 의 api.stopTracking() 을 전역 레퍼런스로 호출.
               TRACKING / READY_TO_NAV / CALIBRATION 상태 대응.
    2) [DT]    직접교시 비활성화 (RobotController.exit_direct_teaching)
               -> indy.set_direct_teaching(enable=False)
               DIRECT_TEACHING 상태 대응.
    3) [ROBOT] 로봇 모션 즉시 정지 (indy.stop_motion)
               ROBOT_MOVING / CALIBRATION / READY_TO_NAV 등 대응.
    4) [ROBOT] 에러 복구 (RobotController.robot_recovery -> indy.recover())
               어떤 state에서 왔더라도 항상 실행.

    각 단계는 독립 try/except 로 보호하여
    한 단계 실패가 다음 단계를 막지 않는다.
    """
    log.warning(f"[STOP] Recovery 시작 — 직전 state: {prev_state}")

    # ── Step 1: NDI stopTracking ─────────────────────────────────────────────
    if prev_state in (State.TRACKING, State.READY_TO_NAV, State.CALIBRATION):
        log.info("[STOP] NDI stopTracking 시도...")
        _stop_active_ndi()

    # ── Step 2: 직접교시 비활성화 ────────────────────────────────────────────
    if prev_state == State.DIRECT_TEACHING:
        log.info("[STOP] 직접교시 비활성화 시도 (exit_direct_teaching)...")
        try:
            if robot_controller is not None:
                robot_controller.exit_direct_teaching()
                log.success("[STOP] 직접교시 비활성화 완료.")
        except Exception as e:
            log.warning(f"[STOP] 직접교시 비활성화 중 오류: {e}")

    # ── Step 3: 로봇 모션 즉시 정지 ─────────────────────────────────────────
    if prev_state in (State.ROBOT_MOVING, State.CALIBRATION,
                      State.READY_TO_NAV, State.TRACKING):
        log.info("[STOP] 로봇 모션 즉시 정지 시도 (stop_motion)...")
        stop_robot_motion(robot_controller)

    # ── Step 4: 에러 복구 ───────────────────────────────────────────────────
    log.info("[STOP] 로봇 에러 복구 시도 (robot_recovery = indy.recover())...")
    try:
        if robot_controller is not None:
            robot_controller.robot_recovery()
            log.success("[STOP] Robot recovery 완료.")
        else:
            log.warning("[STOP] robot_controller 없음 — recovery 생략.")
    except Exception as e:
        log.warning(f"[STOP] Robot recovery 중 오류: {e}")

    log.info("[STOP] Recovery 시퀀스 완료.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. State 핸들러 함수
# ══════════════════════════════════════════════════════════════════════════════

# ── 6-1. TRACKING ─────────────────────────────────────────────────────────────

def handle_tracking(stop_event: threading.Event,
                    hostname: str, tools: list,
                    rom_dir: str, encrypted: bool, cipher: str) -> str:
    """
    NDI 트래킹을 daemon 스레드에서 실행하고 stop_event로 종료를 제어한다.

    Returns
    -------
    State.IDLE : 정상 종료
    State.STOP : stop_event 세팅 또는 내부 오류
    """
    log.info("[TRACKING] NDI 트래킹 시작... (stop 명령 또는 Ctrl+C로 종료)")

    result_holder = {"state": State.IDLE}
    tracking_done = threading.Event()

    def _thread():
        api = None
        try:
            api, _ = ndi.connect_and_setup(
                hostname, tools, rom_dir, encrypted, cipher
            )
            api.startTracking()
            log.success("[TRACKING] NDI startTracking 완료.")

            _register_ndi_api(api)

            while not stop_event.is_set():
                tool_data_list = api.getTrackingDataBX2()
                for td in tool_data_list:
                    full_data = ndi.extract_full_data_dict(td)
                    if not ndi.is_missing_frame(full_data):
                        ndi.print_tracking_data(full_data)
                time.sleep(0.05)

            log.info("[TRACKING] stop_event 감지 → 트래킹 루프 종료.")

        except Exception as e:
            log.error(f"[TRACKING] 내부 오류: {e}")
            result_holder["state"] = State.STOP
        finally:
            if api is not None:
                try:
                    api.stopTracking()
                    log.info("[TRACKING] NDI stopTracking 완료.")
                except Exception as e:
                    log.warning(f"[TRACKING] stopTracking 중 오류: {e}")
            _unregister_ndi_api()
            tracking_done.set()

    t = threading.Thread(target=_thread, daemon=True, name="tracking-worker")
    t.start()

    while not tracking_done.is_set():
        if stop_event.is_set():
            log.warning("[TRACKING] stop_event 감지 (메인) → 스레드 종료 대기 중...")
            result_holder["state"] = State.STOP
            tracking_done.wait(timeout=5.0)
            break
        time.sleep(0.05)

    log.info(f"[TRACKING] 종료. 다음 state → {result_holder['state']}")
    return result_holder["state"]


# ── 6-2. CALIBRATION ──────────────────────────────────────────────────────────

def handle_calibration(
    robot_controller,
    hostname: str, tools: list, rom_dir: str,
    encrypted: bool, cipher: str,
    robot_pose_path: str,
    dataset_root: str,
    duration_sec: float,
    samples: int,
    stop_event: threading.Event,
    calib_paths: dict,
) -> str:
    """
    캘리브레이션 데이터 수집 → HandEyeCalibration 실행.

    Returns
    -------
    State.IDLE : 정상 완료
    State.STOP : stop_event 또는 오류
    """
    log.info("[CALIBRATION] 캘리브레이션 시작.")

    csv_file = io_utils.delete_calibration_csv(robot_pose_path, dataset_root)

    try:
        log.info("[CALIBRATION] 홈 포지션 이동 중...")
        robot_controller.move_to_home()

        with open(robot_pose_path, "r", encoding="utf-8") as f:
            pose_list = sorted(json.load(f), key=lambda x: x["sample_number"])

        log.info(f"[CALIBRATION] 총 {len(pose_list)}개 포즈 수집 시작.")

        api = ndi.connect_and_setup_calibration(
            hostname, tools, rom_dir, encrypted, cipher
        )

        try:
            for pose in pose_list:
                if stop_event.is_set():
                    log.warning("[CALIBRATION] STOP 신호 수신 — 조기 종료")
                    return State.STOP

                pose_id    = pose["sample_number"]
                target_pos = pose["pose"]

                log.info(
                    f"[CALIBRATION] Pose {pose_id}/{len(pose_list)}: "
                    f"로봇 이동 중... target={target_pos}"
                )
                robot_controller.movel_to_pose(
                    target_pos, vel_ratio=10, acc_ratio=10, timeout=60
                )
                time.sleep(1.0)

                log.info(
                    f"[CALIBRATION] Pose {pose_id}: "
                    f"{samples}샘플 수집 (timeout: {duration_sec}s)..."
                )

                def on_sample(full_data, _pid=pose_id):
                    pos = full_data["position"]
                    q   = full_data["quaternion"]
                    tool_data = {
                        "q0": q["w"], "qx": q["x"],
                        "qy": q["y"], "qz": q["z"],
                        "tx": pos["x"], "ty": pos["y"], "tz": pos["z"],
                        "error": full_data["error_mm"],
                    }
                    pose_state = robot_controller.get_current_pose()
                    robot_data = {
                        "x": pose_state[0], "y": pose_state[1], "z": pose_state[2],
                        "u": pose_state[3], "v": pose_state[4], "w": pose_state[5],
                    }
                    io_utils.save_data_to_csv(
                        csv_file, full_data["timestamp"], _pid,
                        tool_data, robot_data=robot_data,
                    )

                collected = ndi.collect_marker_samples(
                    api, samples, duration_sec, pose_id, on_sample
                )
                n = len(collected)
                if   n == 0:       log.error(f"[CALIBRATION] Pose {pose_id}: 유효 데이터 없음!")
                elif n < samples:  log.warning(f"[CALIBRATION] Pose {pose_id}: {n}/{samples} 샘플만 수집.")
                else:              log.info(f"[CALIBRATION] Pose {pose_id}: {n}샘플 저장 완료.")

        finally:
            cleanup_ndi(api)
            robot_controller.move_to_home()
            log.info("[CALIBRATION] NDI 종료 + 홈 복귀 완료.")

        log.info("[CALIBRATION] HandEyeCalibration 실행 중...")
        calib = HandEyeCalibration(csv_path=calib_paths["csv"])
        calib.run()
        log.success("[CALIBRATION] 완료 → IDLE")
        return State.IDLE

    except Exception as e:
        log.error(f"[CALIBRATION] 오류: {e}")
        return State.STOP


# ── 6-3. READY_TO_NAV ─────────────────────────────────────────────────────────

def _wait_for_input_nonblocking(
    prompt: str,
    stop_event: threading.Event,
    dispatcher: "ActionDispatcher",
    nbi: Optional["NonBlockingInput"] = None,
    poll_interval: float = 0.05,
) -> Optional[str]:
    """
    NonBlockingInput을 이용해 사용자 입력을 기다리되,
    stop_event 세팅 또는 action.json 명령 수신 시 즉시 None을 반환한다.

    [수정]
    - nbi 파라미터: 외부에서 생성된 NonBlockingInput 인스턴스를 재사용 가능.
      None이면 내부에서 새로 생성한다.

    Parameters
    ----------
    prompt        : input() 프롬프트 문자열
    stop_event    : 전역 중단 이벤트
    dispatcher    : action.json 감시용 Dispatcher
    nbi           : 재사용할 NonBlockingInput 인스턴스 (None이면 내부 생성)
    poll_interval : polling 주기 (초)

    Returns
    -------
    str  : 사용자가 입력한 문자열 (strip 적용)
    None : stop_event 세팅 또는 외부 명령 수신으로 인해 중단됨
    """
    if nbi is None:
        nbi = NonBlockingInput(prompt)
        nbi.start()

    while True:
        if stop_event.is_set():
            return None

        with dispatcher._lock:
            has_pending = dispatcher._pending_action is not None
        if has_pending:
            return None

        val = nbi.get()
        if val is not None:
            return val.strip()

        time.sleep(poll_interval)


def handle_navigation(
    sm: StateManager,
    robot_controller,
    dispatcher: "ActionDispatcher",
    hostname: str, ttool: str, rom_dir: str,
    encrypted: bool, cipher: str,
    calib_json_path: str,
    config: dict,
    stop_event: threading.Event,
) -> str:
    log.section(
        "State: READY_TO_NAV\n"
        "  마커 인식 → 목표 확인 → Enter: 이동 / j: 조그 / r: 재인식 / q: 종료"
    )

    if not os.path.exists(calib_json_path):
        log.error(f"[NAV] 캘리브레이션 결과 파일 없음: {calib_json_path}")
        return State.IDLE

    try:
        nav = Navigator(calib_path=calib_json_path)
        log.info(f"[NAV] Navigator 로드 완료 (method={nav.method}, unit={nav.unit})")
    except Exception as e:
        log.error(f"[NAV] Navigator 초기화 실패: {e}")
        return State.IDLE

    try:
        api, ttool_handle = ndi.connect_and_setup_navigation(
            hostname, ttool, rom_dir, encrypted, cipher
        )
    except Exception as e:
        log.error(f"[NAV] NDI 연결 실패: {e}")
        return State.IDLE

    if ttool_handle is None and not _MOCK_MODE:
        log.error("[NAV] ttool_handle 없음 → IDLE")
        return State.IDLE

    _register_ndi_api(api)
    next_state = State.IDLE

    try:
        nbi = None
        while True:
            # ── 긴급 정지 감시 ─────────────────────────────────────────
            if stop_event.is_set():
                log.warning("[NAV] stop_event 감지 → STOP")
                next_state = State.STOP
                break

            # ── 외부 명령(action.json) 감시 ───────────────────────────
            pending = dispatcher.pop_pending()
            if pending is not None:
                new_state = dispatcher.resolve_state(pending)
                if new_state is not None:
                    log.info(f"[NAV] 외부 명령 수신 → {new_state}")
                    next_state = new_state
                    break

            # ── 마커 인식 ──────────────────────────────────────────────
            log.info("[NAV] 마커 인식 중...")
            raw_pose, reason = ndi.get_latest_valid_pose(
                api, ttool_handle, timeout_sec=10.0
            )

            if raw_pose is None:
                log.warning(f"[NAV] 마커 인식 실패: {reason}")
                print("  재시도(Enter) / 키보드조그(j) / 종료(q): ", end="", flush=True)
                sel = _wait_for_input_nonblocking("", stop_event, dispatcher, nbi=nbi)
                if sel is None:
                    continue
                if sel.lower() == "q":
                    log.info("[NAV] 사용자 취소 → IDLE")
                    next_state = State.IDLE
                    break
                if sel.lower() == "j" and robot_controller:
                    robot_controller.keyboard_jog(vel_ratio=10, acc_ratio=10)
                continue

            # ── Navigator 좌표 계산 ────────────────────────────────────
            q0, qx, qy, qz = (
                raw_pose["q0"], raw_pose["qx"],
                raw_pose["qy"], raw_pose["qz"],
            )
            tx, ty, tz = raw_pose["tx"], raw_pose["ty"], raw_pose["tz"]
            result = nav.compute(q0, qx, qy, qz, tx, ty, tz)

            rx, ry, rz = result["x"], result["y"], result["z"]
            u,  v,  w  = result["u"], result["v"], result["w"]

            # ── ★ offset 결정 로직 (핵심 수정) ────────────────────────
            # action의 offset이 모두 0  → config navigation offset을 보정값으로 더함
            # action의 offset에 값 있음 → Navigator 결과만 사용 (config 무시)
            nav_cfg = config.get("navigation", {})
            use_config_offset = (
                float(nav_cfg.get("x_offset", 0.0)) != 0.0
                or float(nav_cfg.get("y_offset", 0.0)) != 0.0
                or float(nav_cfg.get("z_offset", 0.0)) != 0.0
            )

            # NOTE: resolve_state()에서 action offset이 모두 0이면 config를 유지,
            #       하나라도 값이 있으면 config를 덮어쓰므로,
            #       여기서는 현재 config 값만 보면 됩니다.
            if use_config_offset:
                cx = float(nav_cfg.get("x_offset", 0.0))
                cy = float(nav_cfg.get("y_offset", 0.0))
                cz = float(nav_cfg.get("z_offset", 0.0))
                x, y, z = rx + cx, ry + cy, rz + cz
                log.info(
                    f"[NAV] config offset 적용: "
                    f"Δx={cx}, Δy={cy}, Δz={cz} → "
                    f"({rx:.4f}+{cx}, {ry:.4f}+{cy}, {rz:.4f}+{cz})"
                )
            else:
                x, y, z = rx, ry, rz
                log.info("[NAV] config offset = 0 → Navigator 결과 그대로 사용")

            target_pose = [x, y, z, u, v, w]

            log.info(
                f"[NAV] NDI Raw   Pos  "
                f"tx={tx:10.3f}  ty={ty:10.3f}  tz={tz:10.3f} (mm)"
            )
            log.info(
                f"[NAV] Navigator Res  "
                f"x={result['x']:10.4f}  y={result['y']:10.4f}  "
                f"z={result['z']:10.4f} (mm)  "
                f"u={u:8.4f}  v={v:8.4f}  w={w:8.4f} (deg)"
            )
            log.info(
                f"[NAV] Robot Target   "
                f"[{x:.4f}, {y:.4f}, {z:.4f}, {u:.4f}, {v:.4f}, {w:.4f}]"
            )

            # ── 사용자 확인 ────────────────────────────────────────────
            print("  이동(Enter) / 키보드조그(j) / 재인식(r) / 취소(q): ",
                  end="", flush=True)
            sel = _wait_for_input_nonblocking("", stop_event, dispatcher, nbi=nbi)

            if sel is None:
                continue

            sel = sel.lower()

            if sel == "q":
                log.info("[NAV] 사용자 취소 → IDLE")
                next_state = State.IDLE
                break

            if sel == "r":
                log.info("[NAV] 재인식 요청...")
                continue

            if sel == "j" and robot_controller:
                log.info("[NAV] 키보드 조그 진입...")
                robot_controller.keyboard_jog(vel_ratio=10, acc_ratio=10)
                log.info("[NAV] 조그 종료 → 재인식")
                continue

            # ── Enter → 이동 실행 ──────────────────────────────────────
            log.info(f"[NAV] 이동 시작 → target={target_pose}")
            sm.transition(State.ROBOT_MOVING)

            move_done  = threading.Event()
            move_error = {"exc": None}

            def _do_move():
                try:
                    robot_controller.movel_to_pose(
                        target_pose, vel_ratio=10, acc_ratio=10, timeout=60
                    )
                except Exception as exc:
                    move_error["exc"] = exc
                finally:
                    move_done.set()

            threading.Thread(target=_do_move, daemon=True, name="nav-move-worker").start()

            while not move_done.is_set():
                if stop_event.is_set():
                    log.warning("[NAV] 이동 중 stop_event → 모션 중지")
                    try:
                        robot_controller.indy.stop_motion()
                    except Exception:
                        pass
                    move_done.wait(timeout=3.0)
                    next_state = State.STOP
                    break
                time.sleep(0.05)
            else:
                if move_error["exc"] is not None:
                    log.error(f"[NAV] 이동 실패: {move_error['exc']} → STOP")
                    next_state = State.STOP
                else:
                    log.success("[NAV] 이동 완료 → IDLE")
                    next_state = State.IDLE
            break

    except KeyboardInterrupt:
        log.error("[NAV] KeyboardInterrupt → STOP")
        next_state = State.STOP
    finally:
        cleanup_ndi(api)

    return next_state
# ── 6-4. ROBOT_MOVING (move 명령) ─────────────────────────────────────────────

def handle_move(robot_controller, move_data: dict, prev_state: str) -> str:
    """
    action.json 'move' 명령으로 진입한 ROBOT_MOVING 처리.

    [수정]
    - 이동 실행 전 controller.get_opstate() 로 OpState == IDLE(5) 확인.
      IDLE이 아니면 STOP으로 전이한다.

    Returns
    -------
    State.IDLE : 이동 완료
    State.STOP : 오류 또는 OpState 비정상
    """
    offset = move_data.get("offset", {})
    dx, dy, dz = offset.get("x", 0.0), offset.get("y", 0.0), offset.get("z", 0.0)
    du, dv, dw = offset.get("u", 0.0), offset.get("v", 0.0), offset.get("w", 0.0)

    log.info(
        f"[ROBOT_MOVING] {move_data.get('description', '')}\n"
        f"  offset=[{dx}, {dy}, {dz}, {du}, {dv}, {dw}]"
    )

    # [수정] 이동 전 OpState == IDLE 확인
    if not _assert_robot_idle(robot_controller, context="[ROBOT_MOVING/move]"):
        return State.STOP

    try:
        robot_controller.movel_relative_to_pose(
            [dx, dy, dz, du, dv, dw], vel_ratio=10, acc_ratio=10, timeout=60
        )
        log.success("[ROBOT_MOVING] 이동 완료 → IDLE")
        return State.IDLE
    except Exception as e:
        log.error(f"[ROBOT_MOVING] 이동 실패: {e} → STOP")
        return State.STOP


# ── 6-5. DIRECT_TEACHING ──────────────────────────────────────────────────────

def handle_direct_teaching(
    robot_controller,
    stop_event: threading.Event,
    dispatcher: "ActionDispatcher",
) -> str:
    """
    DIRECT_TEACHING 상태 처리.

    - run_direct_teaching() 으로 DT 모드 활성화
    - stop_event 세팅(음성 "중지") 또는 Enter 입력 시 exit_direct_teaching() 호출
    - stop_event 로 종료 → State.STOP
    - Enter 로 종료    → State.IDLE

    Returns
    -------
    State.IDLE  : 사용자 Enter로 정상 완료
    State.STOP  : stop_event (외부 "중지" 명령) 로 종료
    """
    if robot_controller is None:
        log.warning("[DIRECT_TEACHING] robot_controller 없음 → IDLE")
        return State.IDLE

    try:
        robot_controller.run_direct_teaching()
    except Exception as e:
        log.error(f"[DIRECT_TEACHING] run_direct_teaching 실패: {e}")
        return State.IDLE

    log.info("[DIRECT_TEACHING] DT 모드 활성화. 완료 후 Enter / 중지 명령으로 종료.")

    nbi = NonBlockingInput("[DIRECT_TEACHING] 티칭 완료 후 Enter를 누르세요...\n")
    nbi.start()

    next_state = State.IDLE

    try:
        while True:
            # ── stop_event 감지: 음성 "중지" 명령 ──────────────────────
            if stop_event.is_set():
                log.warning("[DIRECT_TEACHING] stop_event 감지 → DT 종료 (STOP)")
                next_state = State.STOP
                break

            # ── action.json 외부 명령 감시 ─────────────────────────────
            with dispatcher._lock:                          # ← _lock (private)
                has_pending = dispatcher._pending_action is not None
            if has_pending:
                log.info("[DIRECT_TEACHING] 외부 명령 수신 → DT 종료")
                next_state = State.IDLE
                break

            # ── Enter 입력 감지 ─────────────────────────────────────────
            val = nbi.get()
            if val is not None:
                log.info("[DIRECT_TEACHING] Enter 입력 → DT 종료 (IDLE)")
                next_state = State.IDLE
                break

            time.sleep(0.05)

    finally:
        try:
            robot_controller.exit_direct_teaching()
            log.success(f"[DIRECT_TEACHING] 완료 → {next_state}")
        except Exception as e:
            log.warning(f"[DIRECT_TEACHING] exit_direct_teaching 실패: {e}")

    return next_state
# ── 6-6. IDLE 메뉴 ────────────────────────────────────────────────────────────

def handle_idle(
    sm: StateManager,
    dispatcher: ActionDispatcher,
    config: dict,
    robot_controller,
    stop_event: threading.Event,
) -> str:
    """
    IDLE 상태에서 키보드 입력 또는 action.json 명령 대기.

    [수정]
    - 진입 시 stop_event / 입력 버퍼 소비 (StateManager.transition이 이미 처리하나
      IDLE 자체 재진입 방어로 한 번 더 수행)

    Returns
    -------
    전이할 다음 State 문자열
    """
    log.section(
        "State: IDLE\n"
        "  1: Tracking  2: Calibration  3: Navigation 4: Direct Teaching\n"
        "  j: 키보드 조그  Q: Exit\n"
        "  (키보드 입력 또는 action.json 명령 대기 중...)"
    )

    # IDLE 진입 시 잔여 stop 신호 소비
    if stop_event.is_set():
        stop_event.clear()
        dispatcher.pop_pending()
        log.info("[IDLE] 잔여 STOP 신호 소비 (IDLE 유지)")

    nbi = NonBlockingInput("Select: ")
    nbi.start()

    while True:
        # ── action.json 감시 ──────────────────────────────────────────────
        pending = dispatcher.pop_pending()
        if pending is not None:
            new_state = dispatcher.resolve_state(pending)
            if new_state is not None:
                log.info(f"[IDLE] 명령 수신 → {new_state}")
                return new_state
            log.warning(
                f"[IDLE] 알 수 없는 명령 무시: {pending.get('description', '')!r}"
            )

        # ── stop_event 감시 (IDLE에서 재진입 방지) ────────────────────────
        if stop_event.is_set():
            stop_event.clear()
            dispatcher.pop_pending()
            log.info("[IDLE] stop_event 소비 (IDLE 유지)")

        # ── 키보드 입력 감시 ──────────────────────────────────────────────
        sel = nbi.get()
        if sel is not None:
            sel = sel.strip()
            if   sel == "1":         return State.TRACKING
            elif sel == "2":         return State.CALIBRATION
            elif sel == "3":         return State.READY_TO_NAV
            elif sel == "4":         return State.DIRECT_TEACHING
            elif sel.lower() == "q": return State.EXIT
            elif sel.lower() == "j":
                if robot_controller:
                    log.info("[IDLE] 키보드 조그 진입. (종료 후 IDLE 복귀)")
                    robot_controller.keyboard_jog(vel_ratio=10, acc_ratio=10)
                    log.info("[IDLE] 키보드 조그 종료 → IDLE 유지")
                else:
                    log.warning("[IDLE] robot_controller 없음 — 조그 불가.")
                nbi = NonBlockingInput("Select: ")
                nbi.start()
                continue
            else:
                log.warning(f"[IDLE] 알 수 없는 입력: '{sel}' — 무시.")
                nbi = NonBlockingInput("Select: ")
                nbi.start()
                continue

        time.sleep(0.05)


# ══════════════════════════════════════════════════════════════════════════════
# 6-B. 공통 유틸: ROBOT_MOVING 진입 전 OpState 사전 확인
# ══════════════════════════════════════════════════════════════════════════════

def _assert_robot_idle(robot_controller, context: str = "") -> bool:
    """
    [수정] ROBOT_MOVING 전환 직전에 반드시 OpState == IDLE(5)인지 확인한다.
    controller.get_opstate() 를 이용해 이름 문자열로 비교한다.

    Parameters
    ----------
    robot_controller : RobotController 인스턴스 (None이면 False 반환)
    context          : 로그 출력 시 사용할 컨텍스트 문자열

    Returns
    -------
    True  : OpState == IDLE → 이동 명령 실행 가능
    False : 비정상 상태 → 이동 명령 실행 금지
    """
    if robot_controller is None:
        log.error(f"{context} robot_controller 없음 → 이동 거부")
        return False

    try:
        op_name = robot_controller.get_opstate()   # e.g. "IDLE", "MOVING", ...
    except Exception as e:
        log.error(f"{context} get_opstate() 실패: {e} → 이동 거부")
        return False

    if op_name != "IDLE":
        log.error(
            f"{context} OpState={op_name} (IDLE이 아님) → 이동 거부. "
            f"STOP으로 전이 필요."
        )
        return False

    log.debug(f"{context} OpState=IDLE 확인 완료 → 이동 허용")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN — State Machine 루프
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    메인 State Machine 루프.

    초기화 → NOT_READY → IDLE → (명령 대기) → 각 state 핸들러 → IDLE 복귀
    Ctrl+C / stop 명령 → STOP → recovery → IDLE
    """
    # ── 설정 로드 ────────────────────────────────────────────────────────────
    config = io_utils.load_config(
        "config.json", base_dir=Path(__file__).resolve().parent
    )

    # NDI 파라미터
    hostname  = config["ndi"]["hostname"]
    tools     = config["ndi"]["tools"]
    rom_dir   = config["ndi"]["rom_dir"]
    encrypted = config["ndi"]["encrypted"]
    cipher    = config["ndi"]["cipher"]

    # 로봇 파라미터
    robot_ip = config["robot"]["ip"]

    # 데이터셋 경로
    dataset_root    = config["dataset"]["dataset_root"]
    robot_pose_file = config["dataset"]["robot_pose_file"]

    # navigation 파라미터
    ttool = config["navigation"].get("ttool", "")

    # robot_pose 절대 경로 결정
    if os.path.isabs(robot_pose_file):
        robot_pose_path = robot_pose_file
    else:
        robot_pose_path = os.path.join(dataset_root, robot_pose_file)
        if not os.path.exists(robot_pose_path):
            robot_pose_path = os.path.join(
                Path(__file__).resolve().parent, robot_pose_file
            )
    log.info(f"[INIT] robot_pose_path: {robot_pose_path}")

    calib_paths = io_utils.get_calibration_filepaths(robot_pose_file, dataset_root)

    # ── 전역 중단 이벤트 ─────────────────────────────────────────────────────
    stop_event = threading.Event()

    # ── StateManager 초기화 ──────────────────────────────────────────────────
    sm = StateManager(initial_state=State.NOT_READY)

    # ── RobotController (공유) ───────────────────────────────────────────────
    robot_controller: Optional[RobotController] = None

    def get_robot_controller():
        return robot_controller

    def ensure_robot() -> bool:
        nonlocal robot_controller
        try:
            if robot_controller is None:
                robot_controller = RobotController(robot_ip=robot_ip)
            robot_controller.indy.get_control_state()
            return True
        except Exception as e:
            log.error(f"[Robot] 연결 실패: {e}")
            robot_controller = None
            return False

    # ── ActionDispatcher 초기화 ──────────────────────────────────────────────
    # [수정] current_state_getter 추가: interrupt 정책 적용을 위해 현재 state 주입
    dispatcher = ActionDispatcher(
        config=config,
        stop_event=stop_event,
        robot_controller_getter=get_robot_controller,
        current_state_getter=sm.__class__.current.fget.__get__(sm, type(sm))
        if False else (lambda: sm.current),  # property를 callable로 전달
    )

    # ── [수정] StateManager 입력 버퍼 플러시 콜백 등록 ──────────────────────
    # state 전이마다 dispatcher pending 소비 + 입력 큐 초기화
    def _flush_input_buffers():
        """
        state 전이 시마다 호출: 누적 입력 제거.
        - dispatcher.pop_pending(): action.json 미처리 명령 소비
        - stop 이외 pending이 있더라도 전이 후 새 state에서 재평가하므로
          여기서는 무조건 소비한다.
        """
        flushed = dispatcher.pop_pending()
        if flushed is not None:
            log.debug(
                f"[InputFlush] state 전이로 인해 pending 명령 소비: "
                f"action={flushed.get('action')}"
            )

    sm.set_input_flush_fn(_flush_input_buffers)

    # ── [수정] 로봇 OpState 모니터링 스레드 시작 ─────────────────────────────
    robot_monitor = _RobotStateMonitor(
        robot_controller_getter=get_robot_controller,
        stop_event=stop_event,
        sm_current_getter=lambda: sm.current,
    )
    robot_monitor.start()

    # ── ActionWatcher 시작 ───────────────────────────────────────────────────
    action_file = os.path.join(
        Path(__file__).resolve().parent.parent,
        "shared", "action.json",
    )
    watcher = io_utils.ActionWatcher(action_file, dispatcher.on_action_received)
    watcher.start()
    log.info(f"[INIT] ActionWatcher 시작: {action_file}")

    # ── SIGINT (Ctrl+C) 핸들러 등록 ─────────────────────────────────────────
    def _sigint_handler(signum, frame):
        log.warning("[MAIN] Ctrl+C (SIGINT) 수신 → stop_event.set()")
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    # ══════════════════════════════════════════════════════════════════════════
    # State Machine 메인 루프
    # ══════════════════════════════════════════════════════════════════════════
    log.info(f"[MAIN] State Machine 시작. 초기 state: {sm.current}")

    try:
        while True:
            current = sm.current

            # ────────────────────────────────────────────────────────────────
            # NOT_READY : 로봇 / NDI 초기 연결 시도
            # ────────────────────────────────────────────────────────────────
            if current == State.NOT_READY:
                stop_event.clear()
                log.info("[NOT_READY] 로봇 및 NDI 연결 시도 중...")

                if ensure_robot():
                    log.success("[NOT_READY] 로봇 연결 성공.")
                else:
                    log.warning(
                        "[NOT_READY] 로봇 연결 실패 — "
                        "move/jog 기능 비활성화 상태로 IDLE 진입."
                    )
                log.info("[NOT_READY] NDI는 각 모드 진입 시 연결합니다.")
                sm.transition(State.IDLE)

            # ────────────────────────────────────────────────────────────────
            # IDLE : 사용자 입력 / action.json 명령 대기
            # ────────────────────────────────────────────────────────────────
            elif current == State.IDLE:
                stop_event.clear()
                dispatcher.pop_pending()          # 잔여 pending 소비

                if robot_controller is None:
                    ensure_robot()

                next_state = handle_idle(
                    sm=sm,
                    dispatcher=dispatcher,
                    config=config,
                    robot_controller=robot_controller,
                    stop_event=stop_event,
                )
                sm.transition(next_state)

            # ────────────────────────────────────────────────────────────────
            # TRACKING : NDI 마커 실시간 트래킹
            # ────────────────────────────────────────────────────────────────
            elif current == State.TRACKING:
                stop_event.clear()
                next_state = handle_tracking(
                    stop_event=stop_event,
                    hostname=hostname, tools=tools,
                    rom_dir=rom_dir, encrypted=encrypted, cipher=cipher,
                )
                sm.transition(next_state)

            # ────────────────────────────────────────────────────────────────
            # CALIBRATION : 핸드-아이 캘리브레이션
            # ────────────────────────────────────────────────────────────────
            elif current == State.CALIBRATION:
                stop_event.clear()

                if not ensure_robot():
                    log.error("[CALIBRATION] 로봇 연결 없음 → IDLE")
                    sm.transition(State.IDLE)
                    continue

                next_state = handle_calibration(
                    robot_controller=robot_controller,
                    hostname=hostname, tools=tools,
                    rom_dir=rom_dir, encrypted=encrypted, cipher=cipher,
                    robot_pose_path=robot_pose_path,
                    dataset_root=dataset_root,
                    duration_sec=config["calibration"]["duration_sec"],
                    samples=config["calibration"]["samples"],
                    stop_event=stop_event,
                    calib_paths=calib_paths,
                )
                sm.transition(next_state)

            # ────────────────────────────────────────────────────────────────
            # READY_TO_NAV : 마커 인식 → 목표 계산 → 사용자 확인
            # ────────────────────────────────────────────────────────────────
            elif current == State.READY_TO_NAV:
                stop_event.clear()

                if not ensure_robot():
                    log.error("[READY_TO_NAV] 로봇 연결 없음 → IDLE")
                    sm.transition(State.IDLE)
                    continue

                next_state = handle_navigation(
                    sm=sm,
                    robot_controller=robot_controller,
                    dispatcher=dispatcher,
                    hostname=hostname, ttool=ttool,
                    rom_dir=rom_dir, encrypted=encrypted, cipher=cipher,
                    calib_json_path=calib_paths["json"],
                    config=config,
                    stop_event=stop_event,
                )
                # handle_navigation 내부에서 ROBOT_MOVING 전이가 이미 수행된 경우,
                # 이동은 완료됐으므로 반환된 next_state(IDLE/STOP)로만 전이한다.
                sm.transition(next_state)

            # ────────────────────────────────────────────────────────────────
            # ROBOT_MOVING : move 명령으로 진입한 상대 이동
            # ────────────────────────────────────────────────────────────────
            elif current == State.ROBOT_MOVING:
                stop_event.clear()

                # [수정] ROBOT_MOVING 진입 직후 OpState == IDLE(5) 강제 확인
                # controller.get_opstate()를 이용하며, IDLE이 아니면 STOP 전이
                if not _assert_robot_idle(robot_controller, context="[ROBOT_MOVING 진입]"):
                    log.error(
                        "[ROBOT_MOVING] OpState != IDLE → 이동 명령 차단, STOP 전이"
                    )
                    sm.transition(State.STOP)
                    continue

                move_data = dispatcher.pop_pending_move()
                if move_data is None:
                    prev = sm.previous
                    if prev == State.READY_TO_NAV:
                        log.info("[ROBOT_MOVING] Navigation 이동 완료 → IDLE")
                        sm.transition(State.IDLE)
                    else:
                        log.warning(
                            "[ROBOT_MOVING] 이동 데이터 없음 "
                            f"→ 이전 state({prev}) 복귀"
                        )
                        back = prev if prev != State.ROBOT_MOVING else State.IDLE
                        sm.transition(back)
                    continue

                if not ensure_robot():
                    log.error("[ROBOT_MOVING] 로봇 연결 없음 → STOP")
                    sm.transition(State.STOP)
                    continue

                next_state = handle_move(
                    robot_controller=robot_controller,
                    move_data=move_data,
                    prev_state=sm.previous,
                )
                sm.transition(next_state)

            # ────────────────────────────────────────────────────────────────
            # DIRECT_TEACHING : 다이렉트 티칭
            # ────────────────────────────────────────────────────────────────
            elif current == State.DIRECT_TEACHING:
                stop_event.clear()
                next_state = handle_direct_teaching(
                    robot_controller,
                    stop_event=stop_event,
                    dispatcher=dispatcher,
                )
                sm.transition(next_state)

            # ────────────────────────────────────────────────────────────────
            # STOP : 긴급 정지 → recovery → IDLE 복귀
            # ────────────────────────────────────────────────────────────────
            elif current == State.STOP:
                do_stop_recovery(
                    robot_controller=robot_controller,
                    prev_state=sm.previous,
                )
                stop_event.clear()
                dispatcher.pop_pending()
                log.info("[STOP] Recovery 완료 → IDLE 자동 복귀")
                sm.transition(State.IDLE)

            # ────────────────────────────────────────────────────────────────
            # EXIT : 프로그램 종료
            # ────────────────────────────────────────────────────────────────
            elif current == State.EXIT:
                log.info("[EXIT] 프로그램 종료.")
                break

            # ────────────────────────────────────────────────────────────────
            # 알 수 없는 state (버그 방어)
            # ────────────────────────────────────────────────────────────────
            else:
                log.error(f"[MAIN] 알 수 없는 state: {current} → NOT_READY")
                sm.transition(State.NOT_READY)

    except KeyboardInterrupt:
        log.warning("[MAIN] KeyboardInterrupt → STOP recovery 후 종료")
        sm.transition(State.STOP)
        do_stop_recovery(robot_controller=robot_controller, prev_state=sm.previous)

    finally:
        robot_monitor.stop()
        watcher.stop()
        log.info("[MAIN] 모니터링/ActionWatcher 종료. 프로그램 완전 종료.")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()