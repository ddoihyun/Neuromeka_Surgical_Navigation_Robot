# config.py
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# [모델 선택] 원하는 모델로 이름을 변경하세요.
STT_MODEL = "whisper"   # "whisper" 또는 "google"
LLM_MODEL = "gpt"    # "gpt" 또는 "gemini"
TTS_MODEL = "openai"    # "google(gtts)" 또는 "clova" 또는 "openai"
INPUT_MODE = "manual"     # "vad" (자동 감지) 또는 "manual" (엔터키) 또는 "wakeword" (호출어)

# [API 키 설정]
OPENAI_API_KEY = "sk-xxxxxx-x-xxxxxxxxxxxxxxxx-xxxxxxxxxxxx-x" # "YOUR_OPENAI_API_KEY"
GEMINI_API_KEY = "AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # "YOUR_GEMINI_API_KEY"
CLOVA_CLIENT_ID = "xxxxxxxxxx" # "YOUR_CLOVA_CLIENT_ID"
CLOVA_CLIENT_SECRET = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # "YOUR_CLOVA_CLIENT_SECRET"
GOOGLE_KEY_PATH = os.path.join(PROJECT_ROOT, "configs", "gen-lang-client-xxxxxxxxxx-xxxxxxxxxxxx.json") # google-service-account.json"
# [Realtime API 설정]
SERVER_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"

# [오디오 설정]
SAMPLE_RATE = 16000
CHUNK_SIZE = 160 # VAD를 위한 10ms 단위

# [Wake Word 설정]
WAKE_WORD_ENABLED = True
WAKE_WORD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "NamiEir.tflite") # Teachable Machine 모델
WAKE_WORD_NAMES = {
    0: "Nami",
    1: "Eir",
    2: "Background"  # 사용 안 함
}
WAKE_WORD_TARGETS = [1]  # None = 모두 감지, [0] = "Nami"만, [1] = "Eir"만
WAKE_WORD_THRESHOLD = 0.55 # 0.3~0.7 사이로 조절 가능
WAKE_WORD_TIMEOUT = None

# [VAD 설정]
VAD_ENABLED = True
VAD_SILENCE_DURATION = 3.0  # 1.5초 침묵 시 녹음 중지
VAD_ENERGY_THRESHOLD = 300  # 음성 에너지 임계값

# [로그 설정]
LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
LOG_USE_EMOJI = True         # 이모지 사용 여부
LOG_USE_COLOR = True         # 터미널 색상 사용 여부
LOG_TO_FILE = True           # 로그 파일 저장 여부 (main.py에서만)
LOG_OUTPUT_DIR = "logs"   # 로그 출력 디렉토리
SUPPRESS_TF_WARNINGS = True
SUPPRESS_PYGAME_HELLO = True

# [TTS 설정]
TTS_KEEP_FILES = True  # True: 파일 유지 (디버깅), False: 자동 삭제 (운영)

# [파일 경로 설정]
OUTPUT_DIR = "assets"
TTS_OUTPUT_FILE = "tts_output.mp3"
RECORDING_OUTPUT_FILE = "user_voice.wav"
ACTION_JSON_FILE = "action.json"  # LLM 결과 저장 파일

# 전체 경로
TTS_PATH = os.path.join(OUTPUT_DIR, TTS_OUTPUT_FILE)
RECORDING_PATH = os.path.join(OUTPUT_DIR, RECORDING_OUTPUT_FILE)
ACTION_JSON_PATH = os.path.join(OUTPUT_DIR, ACTION_JSON_FILE)

# [DOCKER를 위한 파일 경로 설정]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACTION_JSON_PATH = os.path.join(BASE_DIR, "../..", "shared", ACTION_JSON_FILE)