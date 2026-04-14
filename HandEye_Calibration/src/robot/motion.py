"""
motion.py – 로봇 동작 유틸 함수 모음 (Neuromeka IndyDCP)

IndyDCP3 인스턴스(indy)를 직접 받아 동작하는 함수들만 정의한다.
클래스 수준의 제어 로직은 controller.py 의 RobotController 를 사용할 것.
"""

import time
import json

from neuromeka import IndyDCP3, TaskBaseType, BlendingType, OpState
from typing import List
from src.utils.logger import get_logger
log = get_logger(__name__)

def wait_until_idle(indy: IndyDCP3, timeout: float = 10.0):
    start = time.time()
    while True:
        op_state = indy.get_robot_data().get('op_state')

        if op_state == OpState.IDLE:
            return True

        if time.time() - start > timeout:
            raise TimeoutError("Robot not in IDLE state")

        time.sleep(0.1)


def wait_until_reached(indy: IndyDCP3, timeout: float = 30.0, poll_interval: float = 0.1) -> bool:
    """
    로봇이 목표 위치에 도달할 때까지 폴링 대기.

    Parameters
    ----------
    indy          : IndyDCP3
    timeout       : float  – 최대 대기 시간 (초)
    poll_interval : float  – 폴링 간격 (초)

    Returns
    -------
    True : 도달 성공

    Raises
    ------
    TimeoutError : timeout 초과 시
    """
    start = time.time()

    while True:
        motion = indy.get_motion_data()
        robot  = indy.get_robot_data()

        op_state   = robot.get('op_state')

        # ✅ 정상 종료 조건
        if not motion['is_in_motion'] and op_state == OpState.IDLE:
            return True

        # ❗ 이상 상태 감지 (추천)
        if op_state in [OpState.COLLISION, OpState.VIOLATE, OpState.VIOLATE_HARD]:
            raise RuntimeError(f"Robot error state detected: {op_state}")
        
        if time.time() - start > timeout:
            raise TimeoutError("Motion timeout exceeded")
        time.sleep(poll_interval)


def movej_and_wait(indy: IndyDCP3, target_joint: List[float],
                   vel_ratio: int = 10, acc_ratio: int = 10, timeout: float = 60.0):
    """movej 명령 실행 후 도달 대기."""
    wait_until_idle(indy)
    indy.movej(jtarget=target_joint, vel_ratio=vel_ratio, acc_ratio=acc_ratio)
    wait_until_reached(indy, timeout=timeout)


def movel_and_wait(indy: IndyDCP3, target_pos: List[float],
                   vel_ratio: int = 10, acc_ratio: int = 10, timeout: float = 60.0):
    """movel 명령 실행 후 도달 대기. robot_pose_json용(절대위치)"""
    wait_until_idle(indy)
    indy.movel(ttarget=target_pos, vel_ratio=vel_ratio, acc_ratio=acc_ratio,
               base_type=TaskBaseType.ABSOLUTE, bypass_singular=True)
    wait_until_reached(indy, timeout=timeout)


def movel_relative_and_wait(indy: IndyDCP3, target_pos: List[float],
                            vel_ratio: int = 10, acc_ratio: int = 10, timeout: float = 60.0):
    """movel 상대 이동 명령 실행 후 도달 대기."""
    wait_until_idle(indy)
    indy.movel(ttarget=target_pos, vel_ratio=vel_ratio, acc_ratio=acc_ratio,
               blending_type=BlendingType.OVERRIDE,
               base_type=TaskBaseType.TCP, bypass_singular=True)
    wait_until_reached(indy, timeout=timeout)


def movel_relative(indy: IndyDCP3, target_pos: List[float],
                   vel_ratio: int = 10, acc_ratio: int = 10):
    wait_until_idle(indy)
    indy.movel(ttarget=target_pos, vel_ratio=vel_ratio, acc_ratio=acc_ratio,
               blending_type=BlendingType.OVERRIDE,
               base_type=TaskBaseType.TCP, bypass_singular=True)


def movel_from_json(indy: IndyDCP3, json_path: str,
                    vel_ratio: int = 10, acc_ratio: int = 10, timeout: float = 60):
    """
    robot_pose.json 을 읽어 sample_number 순서대로 movel 이동.

    JSON 포맷 예시:
        [{"sample_number": 1, "pose": [x, y, z, u, v, w]}, ...]
    """
    with open(json_path, 'r') as f:
        pose_list = sorted(json.load(f), key=lambda x: x['sample_number'])

    for item in pose_list:
        wait_until_idle(indy)
        sample_no  = item['sample_number']
        target_pos = item['pose']
        log.info(f"[{sample_no}] Moving to pose: {target_pos}")
        movel_and_wait(indy, target_pos, vel_ratio=vel_ratio, acc_ratio=acc_ratio, timeout=timeout)
        log.success(f"Reached sample {sample_no}")