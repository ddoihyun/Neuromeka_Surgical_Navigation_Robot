# Neuromeka Surgical Navigation Robot

음성 기반 명령과 마커 인식을 결합하여 로봇을 제어하는 프로젝트입니다.
두 개의 독립적인 환경(Voice / Navigation)이 action.json을 통해 상호 연동됩니다.

## 📌 시스템 구조
---
```
[VoiceModular 3.11 venv]          [Navigation 3.7 venv]
        │                                   │
   음성 입력 처리                      마커 인식
   (STT → LLM)                  (Hand-eye calibration)
        │                                   │
        ▼                                   ▼
   action.json  ───────────────▶  action.json 읽기
   (공유 파일)                    → 로봇 위치 이동 실행
```

## ⚙️ 실험 세팅 방법

### 1. 로봇 앤드툴 세팅
- Marker-Flange 부착
- 사용 부품:
    - M5 Nut 1개
    - Bolt 1개
    - M6 Bolt 4개

### 2. 로봇 TCP 설정
- 로봇 접속 후 Conty에서 TCP 위치 변경
- 경로: `[설정] → [프로그램 환경] → [툴 좌표계]`
- 사용자 툴 좌표계 추가 후 값 확인: `tool_frame: [40.45, 0.0, 30.44, 0.0, 0.0, 0.0]`

### 3. 환경 구성
- Pyramid Volume 내부에 다음 배치:
    - 로봇
    - NDI VEGA 센서

### 4. 로봇 Pose 확인
필요 시 위치 보정 수행

### 5. Hand-Eye Calibration 실행

---

## 🛠️ 설치 방법

### Navigation (Python 3.7)
```
.\.venv37\Scripts\python.exe -m pip install --upgrade pip
.\.venv37\Scripts\python.exe -m pip install -r requirements.txt
```

### ▶️ 실행 방법
### 1️⃣ Navigation 실행 (Terminal 1)
``` 
cd Neuromeka_Surgical_Navigation_Robot\HandEye_Calibration
.\.venv37\Scripts\Activate
python main.py
```

### 2️⃣ Voice Module 실행 (Terminal 2)
```
cd Neuromeka_Surgical_Navigation_Robot\VoiceTeaching
.\.venv311\Scripts\Activate
python main.py
```

### 텍스트 입력 모드 실행
```
python main.py --input text
```

## 🔄 동작 흐름
1. Voice Module에서 음성 입력 처리
2. STT → LLM을 통해 명령 생성
3. action.json 파일 생성 (공유 파일)
4. Navigation Module에서 해당 파일 읽기
5. 로봇이 목표 위치로 이동

## 📁 주요 구성 요소
### VoiceTeaching
- 음성 입력 처리 (STT)
- LLM 기반 명령 생성

### HandEye_Calibration
- 마커 인식
좌표 변환 및 로봇 제어
action.json
두 모듈 간 데이터 전달을 위한 공유 파일
⚠️ 주의 사항
두 프로그램은 반드시 동시에 실행되어야 합니다.
action.json 파일 경로가 양쪽에서 동일해야 합니다.
Python 버전이 서로 다르므로 각 환경을 정확히 활성화해야 합니다.
Voice: Python 3.11
Navigation: Python 3.7