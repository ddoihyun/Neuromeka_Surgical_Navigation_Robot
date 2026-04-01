# core/factory.py
"""
STT/LLM/TTS 엔진 팩토리
Lazy Initialization으로 필요할 때만 인스턴스 생성
"""
import modules.stt_modules as stt
import modules.llm_modules as llm
import modules.tts_modules as tts

# 싱글톤 인스턴스 캐시
_stt_instances = {}
_llm_instances = {}
_tts_instances = {}

class STT:
    """STT 지연 초기화 래퍼"""
    def __init__(self, name):
        self.name = name
        self._instance = None
    
    def _get_instance(self):
        if self._instance is None:
            if self.name == "whisper":
                self._instance = stt.WhisperSTT()
            elif self.name == "google":
                self._instance = stt.GoogleSTT()
            elif self.name == "clova":
                self._instance = stt.ClovaSTT()
            else:
                self._instance = stt.WhisperSTT()
        return self._instance
    
    def __call__(self, audio_path):
        return self._get_instance().transcribe(audio_path)

class LLM:
    """LLM 지연 초기화 래퍼"""
    def __init__(self, name):
        self.name = name
        self._instance = None
    
    def _get_instance(self):
        if self._instance is None:
            if self.name == "gpt":
                self._instance = llm.GPT4oLLM()
            elif self.name == "gemini":
                self._instance = llm.GeminiLLM()
            else:
                self._instance = llm.GeminiLLM()
        return self._instance
    
    async def __call__(self, text):
        return await self._get_instance().ask(text)

class TTS:
    """TTS 지연 초기화 래퍼"""
    def __init__(self, name):
        self.name = name
        self._instance = None
    
    def _get_instance(self):
        if self._instance is None:
            if self.name == "google":
                self._instance = tts.GoogleTTS()
            elif self.name == "openai":
                self._instance = tts.OpenAITTS()
            elif self.name == "clova":
                self._instance = tts.ClovaTTS()
            else:
                self._instance = tts.GoogleTTS()
        return self._instance
    
    def __call__(self, text):
        return self._get_instance().speak(text)

def get_stt_fn(name):
    """STT 함수 반환 (지연 초기화)"""
    if name not in _stt_instances:
        _stt_instances[name] = STT(name)
    return _stt_instances[name]

def get_llm_fn(name):
    """LLM 함수 반환 (지연 초기화)"""
    if name not in _llm_instances:
        _llm_instances[name] = LLM(name)
    return _llm_instances[name]

def get_tts_fn(name):
    """TTS 함수 반환 (지연 초기화)"""
    if name not in _tts_instances:
        _tts_instances[name] = TTS(name)
    return _tts_instances[name]
