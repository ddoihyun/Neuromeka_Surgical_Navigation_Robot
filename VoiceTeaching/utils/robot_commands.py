# utils/robot_commands.py
"""
로봇 명령 실행 함수
Global 변수 기반으로 단순화
"""
from configs import globals as g
from utils.logger import Logger


def execute_command(action: str) -> str:
    """
    로봇 명령 실행
    Args:
        action: "tracking" | "calibration" | "navigation" | "move" | "stop" | "direct_teaching
    Returns:
        성공 메시지 또는 None
    """
    if action == "tracking":
        g.set_robot_mode("tracking")
        return "타겟 추적 모드 시작"

    elif action == "calibration":
        g.set_robot_mode("calibration")
        return "캘리브레이션 시작"

    elif action == "navigation":
        g.set_robot_mode("navigation")
        return "내비게이션 준비"

    elif action == "move":
        g.set_robot_mode("move")
        return "목표 위치로 이동 실행"

    elif action == "stop":
        g.set_robot_mode("stop")
        return "정지"
    
    elif action == "direct_teaching":
        g.set_robot_mode("direct_teaching")
        return "직접교시 활성화"

    else:
        Logger.warning(f"알 수 없는 명령: {action}")
        return None


def get_current_mode() -> str:
    """현재 로봇 모드 반환"""
    return g.get_robot_mode()