# configs/globals.py
"""
전역 상태 관리 모듈
Blackboard 패턴 대신 전역 변수 사용
"""
import threading


# 스레드 안전을 위한 Lock
_lock = threading.Lock()


# ==========================================
# 음성 제어 관련 전역 변수
# ==========================================
voice_command = None          # LLM이 반환한 명령 JSON (딕셔너리)
voice_command_raw = None      # LLM이 반환한 원본 JSON 문자열 (원본 코드 호환용)
speak_text = None             # TTS로 말할 텍스트
speak_flag = False            # TTS 요청 플래그
is_speaking = False           # TTS 재생 중 여부

# ==========================================
# 로봇 상태 관련 전역 변수
# ==========================================
robot_mode = "stop"           # 현재 로봇 모드: "tracking", "calibration", "navigation", "move", "stop"

# ==========================================
# 스레드 안전 접근 함수 (필요 시 사용)
# ==========================================
def set_voice_command(command, raw_json=None):
    """
    음성 명령 설정 (스레드 안전)
    
    Args:
        command: 파싱된 JSON 딕셔너리
        raw_json: 원본 JSON 문자열 (선택)
    """
    global voice_command, voice_command_raw
    with _lock:
        voice_command = command
        if raw_json:
            voice_command_raw = raw_json

def get_voice_command():
    """음성 명령 조회 - 딕셔너리 (스레드 안전)"""
    with _lock:
        return voice_command

def get_voice_command_raw():
    """음성 명령 조회 - JSON 문자열 (스레드 안전)"""
    with _lock:
        return voice_command_raw

def set_speak_request(text):
    """TTS 요청 (스레드 안전)"""
    global speak_text, speak_flag
    with _lock:
        speak_text = text
        speak_flag = True

def clear_speak_flag():
    """TTS 플래그 초기화 (스레드 안전)"""
    global speak_flag
    with _lock:
        speak_flag = False

def set_speaking_state(state: bool):
    """TTS 재생 상태 설정 (스레드 안전)"""
    global is_speaking
    with _lock:
        is_speaking = state

def get_speaking_state():
    """TTS 재생 상태 조회 (스레드 안전)"""
    with _lock:
        return is_speaking

def set_robot_mode(mode: str):
    """로봇 모드 설정 (스레드 안전)"""
    global robot_mode
    with _lock:
        robot_mode = mode

def get_robot_mode():
    """로봇 모드 조회 (스레드 안전)"""
    with _lock:
        return robot_mode
