# modules/stt_modules.py
"""
STT (Speech-to-Text) 엔진 통합 모듈
각 엔진을 클래스로 관리하여 독립적 사용 가능
"""
import os
from typing import Optional
import requests
from openai import OpenAI
from google.cloud import speech
from configs import OPENAI_API_KEY, CLOVA_CLIENT_ID, CLOVA_CLIENT_SECRET, GOOGLE_KEY_PATH
from utils.logger import Logger

# ==========================================
# 1. Whisper STT (OpenAI)
# ==========================================
class WhisperSTT:
    """OpenAI Whisper API를 사용한 STT"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)
    
    def transcribe(self, audio_path: str) -> Optional[str]:
        """
        오디오 파일을 텍스트로 변환
        Args:
            audio_path: 오디오 파일 경로 (.wav)
        Returns:
            인식된 텍스트 (실패 시 None)
        """
        if not os.path.exists(audio_path):
            Logger.warning(f"파일 없음: {audio_path}")
            return None
        
        try:
            Logger.debug(f"Whisper STT 처리 중: {audio_path}")
            
            with open(audio_path, "rb") as f:
                result = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text",
                    language="ko"
                )
            
            Logger.debug(f"Whisper 결과: {result}")
            return result
        
        except Exception as e:
            Logger.error(f"Whisper STT 오류: {e}")
            return None

# ==========================================
# 2. Google Cloud STT
# ==========================================
class GoogleSTT:
    """Google Cloud Speech-to-Text API"""
    
    def __init__(self, key_path: str = None):
        self.key_path = key_path or GOOGLE_KEY_PATH
        self.client = speech.SpeechClient.from_service_account_json(self.key_path)
    
    def transcribe(self, audio_path: str) -> Optional[str]:
        """
        오디오 파일을 텍스트로 변환
        Args:
            audio_path: 오디오 파일 경로 (.wav)
        Returns:
            인식된 텍스트 (실패 시 None)
        """
        if not os.path.exists(audio_path):
            Logger.warning(f"파일 없음: {audio_path}")
            return None
        
        try:
            Logger.debug(f"Google STT 처리 중: {audio_path}")
            
            with open(audio_path, "rb") as f:
                content = f.read()
            
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="ko-KR"
            )
            
            response = self.client.recognize(config=config, audio=audio)
            
            for result in response.results:
                text = result.alternatives[0].transcript
                Logger.debug(f"Google STT 결과: {text}")
                return text
            
            Logger.warning("Google STT 결과 없음")
            return None
        
        except Exception as e:
            Logger.error(f"Google STT 오류: {e}")
            return None

# ==========================================
# 3. Clova STT (Naver)
# ==========================================
class ClovaSTT:
    """Naver Clova Speech Recognition API"""
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        self.client_id = client_id or CLOVA_CLIENT_ID
        self.client_secret = client_secret or CLOVA_CLIENT_SECRET
        self.url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=Kor"
    
    def transcribe(self, audio_path: str) -> Optional[str]:
        """
        오디오 파일을 텍스트로 변환
        Args:
            audio_path: 오디오 파일 경로 (.wav)
        Returns:
            인식된 텍스트 (실패 시 None)
        """
        if not os.path.exists(audio_path):
            Logger.warning(f"파일 없음: {audio_path}")
            return None
        
        headers = {
            "X-NCP-APIGW-API-KEY-ID": self.client_id,
            "X-NCP-APIGW-API-KEY": self.client_secret,
            "Content-Type": "application/octet-stream"
        }
        
        try:
            Logger.debug(f"Clova STT 처리 중: {audio_path}")
            
            with open(audio_path, "rb") as f:
                data = f.read()
            
            response = requests.post(self.url, data=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json().get("text")
                Logger.debug(f"Clova STT 결과: {result}")
                return result
            else:
                Logger.error(f"Clova STT HTTP 오류: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            Logger.error(f"Clova STT 예외: {e}")
            return None
