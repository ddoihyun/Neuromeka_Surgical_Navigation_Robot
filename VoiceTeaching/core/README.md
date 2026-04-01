# core/ - 핵심 기능 모듈

VoiceModular 시스템의 핵심 기능을 제공하는 모듈입니다.

## 📁 파일 구성

- `factory.py` - STT/LLM/TTS 엔진 팩토리
- `recorder.py` - 오디오 녹음 (VAD/Manual)
- `wakeword.py` - Wake word 감지

---

## 📄 factory.py - 엔진 선택 팩토리
STT, LLM, TTS 엔진을 선택하여 반환하는 팩토리 패턴 구현입니다.

### get_stt_fn(name)
STT 엔진 함수 반환 ("whisper" | "google" | "clova")
```python
from core.factory import get_stt_fn

stt = get_stt_fn("whisper")
text = stt("assets/user_voice.wav")
```

### get_llm_fn(name)
LLM 엔진 함수 반환 ("gpt" | "gemini")
```python
from core.factory import get_llm_fn

llm = get_llm_fn("gemini")
result = await llm("로봇 이동 모드로 변경")
# {"action": "mode", "aux0": "free", "aux1": "..."}
```

### get_tts_fn(name)
TTS 엔진 함수 반환 ("google" | "openai" | "clova")
```python
from core.factory import get_tts_fn

tts = get_tts_fn("clova")
tts("명령을 수행합니다")
```

---
## 📄 recorder.py - 오디오 녹음

### AudioRecorder.record_to_file(output_path, mode) → str
: 오디오 녹음 후 파일 저장
- mode: "vad" (자동 감지) | "manual" (엔터키)
```python
from core.recorder import AudioRecorder

recorder = AudioRecorder()

# VAD 자동 녹음
path = recorder.record_to_file("user_voice.wav", mode="vad")

# Manual 엔터키 녹음
path = recorder.record_to_file("manual.wav", mode="manual")
```
### AudioRecorder.calculate_rms(frame) → float
: 오디오 프레임의 RMS 에너지 계산 (VAD 용도)

---

## 📄 wakeword.py - Wake Word 감지

### WakeWordDetector.detect_from_stream(audio_chunk) → bool
: 실시간 오디오 청크에서 wake word 감지
```python
import sounddevice as sd
from core.wakeword import WakeWordDetector

detector = WakeWordDetector(
    model_path="models/soundclassifier_with_metadata.tflite",
    threshold=0.7,
    target_wake_words=[4],
    wake_word_names={1: "Hey"}
)

with sd.InputStream(samplerate=16000, channels=1, dtype="int16") as stream:
    while True:
        chunk, _ = stream.read(160)
        if detector.detect_from_stream(chunk):
            print("Wake word 감지!")
            break
```
### WakeWordDetector.reset() → None
: 버퍼 초기화 (다음 감지 준비)
```python
detector.reset()
```