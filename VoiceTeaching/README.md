# 🤖 VoiceModular: Robot Voice Control System

VoiceModular는 STT, LLM, TTS 엔진을 자유롭게 조합하여  
로봇을 음성으로 제어할 수 있도록 설계된 모듈형 음성 제어 시스템입니다.

음성 입력부터 명령 해석, 로봇 제어, 음성 피드백까지의 전체 파이프라인을  
유연하고 확장 가능하게 구성하는 것을 목표로 합니다.


## 🏗 시스템 아키텍처
```
[Mic / Text Input]
    ↓
[Wake Word / VAD / Manual]
    ↓
STT Module [Whisper / Google / Clova]
    ↓
LLM Module [Gemini / GPT]
    ↓
Global 변수 저장
    ↓
Robot Controller
    ↓
TTS Module [Google / OpenAI / Clova]
    ↓
Speaker
```

- 음성 처리부와 제어부를 블랙보드 시스템으로 분리
- 각 단계는 독립 모듈로 교체 및 확장 가능


## 🛠 주요 특징

### 모듈화 구조
- STT: Whisper, Google Speech 등
- LLM: GPT, Gemini 등
- TTS: OpenAI, Google, Naver Clova 등
- 엔진 교체 시 핵심 로직 수정 불필요

### 입력 방식 유연성
- Wake Word 대기 모드
- VAD 기반 마이크 입력
- 텍스트 입력 (디버깅/테스트용)


## 🚀 시작하기

### 1️⃣ 환경 설정
Python 3.9 이상 환경에서 의존성 설치:

```bash
pip install -r requirements.txt
```

### 2️⃣ API 키 설정

다음 파일 중 하나에 API 키를 설정합니다:
- configs/config.py
- .env

필요한 API 키:
- OpenAI API Key
- Google Cloud API Key
- Naver Clova API Key

⚠️ 해당 파일들은 Git에 업로드되지 않도록 .gitignore 설정을 권장합니다.

### 3️⃣ 실행 방법
#### 기본 명령어 구조
```bash
python main.py --input <INPUT_MODE> --mode <VOICE_MODE> [--stt <STT_ENGINE>] [--llm <LLM_ENGINE>] [--tts <TTS_ENGINE>]
```

#### 📋 옵션 설명
```
| `--input` | **입력 방식** | `mic`, `text` | `mic` |
| `--stt` | **STT 엔진** | `whisper`, `google`, `clova` | `config.py` 설정값 |
| `--llm` | **LLM 엔진** | `gemini`, `gpt` | `config.py` 설정값 |
| `--tts` | **TTS 엔진** | `google`, `openai`, `clova` | `config.py` 설정값 |
| `--mode` | **녹음 모드** | `wakeword`, `vad`, `manual` | `vad` |

※ `--stt`, `--llm`, `--tts`를 지정하지 않으면 `configs/config.py`의 기본값 사용
```

##### 1. 음성 입력 모드 (기본)
기본 엔진 사용
```bash
python main.py
```
![Main default](docs/main/main_default.png)
Whisper STT + GPT LLM + OpenAI TTS
```bash
python main.py --stt whisper --llm gemini --tts google
```
Whisper STT + GPT-4o + OpenAI TTS
```bash
python main.py --stt whisper --llm gpt --tts openai
```
Google STT + Gemini + Clova TTS
```bash
python main.py --stt google --llm gemini --tts clova
```

##### 2. Wake Word(호출어) 대기 모드
기본 엔진 사용
```bash
python main.py --mode wakeword
```
![Wakeword](docs/main/main_wakeword.png)
특정 엔진 조합
```bash
python main.py --mode wakeword --stt whisper --llm gpt --tts openai
```

##### 3. 텍스트 입력 모드 (디버깅용)
음성 입력 없이 텍스트로 테스트
```bash
python main.py --input text --llm gemini --tts google
```
STT 건너뛰고 LLM만 테스트
```bash
python main.py --input text --llm gpt
```
![Text](docs/main/main_text.png)

![Debug](docs/main/main_debug.png)
debug 모드는 config.py에서 설정 가능합니다.

#### [⚙️ 설정 파일 사용](./configs/README.md)
`configs/config.py`에서 기본값을 설정하면 명령어를 간단하게 사용할 수 있습니다:


### 🎶 명령어 도움말
전체 옵션을 확인하려면:
```bash
python main.py --help
```
![Help 화면](docs/main/main_help.png)


## 📂 폴더 구조
```bash
VoiceModular/
├── core/        # 핵심 로직 (레코더, 컨트롤러, 팩토리 등)
├── models/      # WakeWord를 위한 FTlite 모델
├── modules/     # 외부 API 연동 모듈 (STT, LLM, TTS)
├── utils/       # 유틸리티 (로거 등)
├── tests/       # 단위 / 통합 테스트 코드
├── configs/     # 설정 파일 (API Key, 모델 설정 등)
├── assets/      # 생성되는 오디오 및 json 파일
├── logs/     # 로그 출력 저장
└── main.py
```
Last Updated: 2026-01-09
