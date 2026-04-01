# modules/tts_modules.py
"""
TTS (Text-to-Speech) 엔진 통합 모듈
각 TTS 엔진을 클래스로 관리 + 백그라운드 TTS Manager 포함
"""
import os
import time
import threading
import requests
import pygame
from gtts import gTTS
from openai import OpenAI
from typing import Optional
from configs import OPENAI_API_KEY, CLOVA_CLIENT_ID, CLOVA_CLIENT_SECRET, OUTPUT_DIR, TTS_PATH, TTS_KEEP_FILES
from configs import globals as g
from utils.logger import Logger

# assets 폴더 자동 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pygame 초기화
pygame.mixer.init()

# ==========================================
# 유틸리티 함수
# ==========================================
class AudioPlayer:
    """오디오 재생 및 파일 관리 유틸리티"""
    
    def __init__(self):
        self._should_stop = False
    
    def stop(self):
        """재생 강제 중단"""
        self._should_stop = True
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except:
            pass
    
    def play(self, file_path: str) -> None:
        """오디오 파일 재생"""
        if not os.path.exists(file_path):
            Logger.warning(f"재생할 파일이 없습니다: {file_path}")
            return
        
        try:
            Logger.debug(f"오디오 재생 시작: {file_path}")
            
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                if self._should_stop:
                    pygame.mixer.music.stop()
                    break
                pygame.time.Clock().tick(10)
            
            pygame.mixer.music.unload()
            Logger.debug("오디오 재생 완료")
        
        except Exception as e:
            Logger.error(f"오디오 재생 오류: {e}")
    
    @staticmethod
    def safe_remove(filepath, max_retries=3):
        """안전하게 파일 삭제 (재시도 포함)"""
        for i in range(max_retries):
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    Logger.debug(f"TTS 파일 삭제: {filepath}")
                return True
            except PermissionError:
                if i < max_retries - 1:
                    time.sleep(0.1)
                else:
                    Logger.warning(f"파일 삭제 실패: {filepath}")
                    return False
            except Exception as e:
                Logger.warning(f"파일 삭제 오류: {e}")
                return False
        return False

# ==========================================
# 1. Google TTS (gTTS)
# ==========================================
class GoogleTTS:
    """Google Text-to-Speech (gTTS)"""
    
    def __init__(self, lang: str = "ko", output_path: str = None):
        self.lang = lang
        self.output_path = output_path or TTS_PATH
        self.player = AudioPlayer()
    
    def speak(self, text: str) -> None:
        """텍스트를 음성으로 출력"""
        if not text:
            return
        
        try:
            Logger.debug(f"Google TTS 생성 중: {text}")
            
            # 기존 파일 삭제 (TTS_KEEP_FILES와 무관하게 항상 삭제)
            AudioPlayer.safe_remove(self.output_path)
            
            tts = gTTS(text=text, lang=self.lang)
            tts.save(self.output_path)
            Logger.debug(f"TTS 파일 저장: {self.output_path}")
            
            self.player.play(self.output_path)
        
        except KeyboardInterrupt:
            Logger.info("TTS 사용자 중단")
            self.player.stop()
            raise
        except Exception as e:
            Logger.error(f"Google TTS 실패: {e}")
        finally:
            # 조건부 삭제: TTS_KEEP_FILES=False일 때만 삭제
            if not TTS_KEEP_FILES:
                AudioPlayer.safe_remove(self.output_path)
            else:
                Logger.debug(f"TTS 파일 유지: {self.output_path}")

# ==========================================
# 2. OpenAI TTS
# ==========================================
class OpenAITTS:
    """OpenAI Text-to-Speech API"""
    
    def __init__(self, api_key: str = None, model: str = "tts-1", voice: str = "alloy", output_path: str = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.voice = voice
        self.output_path = output_path or TTS_PATH
        self.client = OpenAI(api_key=self.api_key)
        self.player = AudioPlayer()
    
    def speak(self, text: str) -> None:
        """텍스트를 음성으로 출력"""
        if not text:
            return
        
        try:
            Logger.debug(f"OpenAI TTS 생성 중: {text}")
            
            AudioPlayer.safe_remove(self.output_path)
            
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text
            )
            response.stream_to_file(self.output_path)
            Logger.debug(f"TTS 파일 저장: {self.output_path}")
            
            self.player.play(self.output_path)
        
        except KeyboardInterrupt:
            Logger.info("TTS 사용자 중단")
            self.player.stop()
            raise
        except Exception as e:
            Logger.error(f"OpenAI TTS 실패: {e}")
        finally:
            if not TTS_KEEP_FILES:
                AudioPlayer.safe_remove(self.output_path)
            else:
                Logger.debug(f"TTS 파일 유지: {self.output_path}")

# ==========================================
# 3. Clova TTS (Naver)
# ==========================================
class ClovaTTS:
    """Naver Clova Premium TTS"""
    
    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        speaker: str = "jinho",
        volume: int = 0,
        speed: int = -1,
        pitch: int = 1,
        alpha: int = 0,
        end_pitch: int = 0,
        output_path: str = None
    ):
        self.url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
        self.client_id = client_id or CLOVA_CLIENT_ID
        self.client_secret = client_secret or CLOVA_CLIENT_SECRET
        self.speaker = speaker
        self.volume = volume
        self.speed = speed
        self.pitch = pitch
        self.alpha = alpha
        self.end_pitch = end_pitch
        self.output_path = output_path or TTS_PATH
        self.player = AudioPlayer()
    
    def speak(self, text: str) -> None:
        """텍스트를 음성으로 출력"""
        if not text:
            return
        
        try:
            Logger.debug(f"Clova TTS 생성 중: {text}")
            
            AudioPlayer.safe_remove(self.output_path)
            
            headers = {
                "X-NCP-APIGW-API-KEY-ID": self.client_id,
                "X-NCP-APIGW-API-KEY": self.client_secret,
                "Content-Type": "application/x-www-form-urlencoded",
            }
            
            data = {
                "speaker": self.speaker,
                "text": text,
                "volume": str(self.volume),
                "speed": str(self.speed),
                "pitch": str(self.pitch),
                "alpha": str(self.alpha),
                "end_pitch": str(self.end_pitch),
                "format": "mp3",
            }
            
            response = requests.post(self.url, headers=headers, data=data)
            
            if response.status_code == 200:
                with open(self.output_path, "wb") as f:
                    f.write(response.content)
                Logger.debug(f"TTS 파일 저장: {self.output_path}")
                
                self.player.play(self.output_path)
            else:
                Logger.error(f"Clova TTS HTTP 오류: {response.status_code} - {response.text}")
        
        except KeyboardInterrupt:
            Logger.info("TTS 사용자 중단")
            self.player.stop()
            raise
        except Exception as e:
            Logger.error(f"Clova TTS 실패: {e}")
        finally:
            if not TTS_KEEP_FILES:
                AudioPlayer.safe_remove(self.output_path)
            else:
                Logger.debug(f"TTS 파일 유지: {self.output_path}")

# ==========================================
# 4. TTS Manager (Global 변수 감시)
# ==========================================
class TTSManager:
    """
    globals.speak_flag를 감시하며
    TTS 요청이 들어오면 음성 출력을 담당
    """
    
    def __init__(self, tts_engine="google"):
        self.running = False
        self.thread = None
        self.speaking_thread = None
        self.speaking_active = False
        
        # TTS 엔진 초기화
        if tts_engine == "google":
            self.tts = GoogleTTS()
        elif tts_engine == "openai":
            self.tts = OpenAITTS()
        elif tts_engine == "clova":
            self.tts = ClovaTTS()
        else:
            Logger.warning(f"알 수 없는 TTS 엔진: {tts_engine}, Google TTS 사용")
            self.tts = GoogleTTS()
        
        Logger.debug(f"TTS 매니저 초기화: {tts_engine}")
    
    def start(self):
        """TTS 감시 루프 시작"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True, name="TTSManager")
            self.thread.start()
            Logger.success("TTS 매니저 시작")
    
    def stop(self):
        """TTS 매니저 종료"""
        if self.running:
            self.running = False
            
            # TTS 즉시 중단
            if hasattr(self.tts, 'player'):
                self.tts.player.stop()
            
            if self.thread:
                self.thread.join(timeout=2)
            if self.speaking_thread and self.speaking_thread.is_alive():
                self.speaking_thread.join(timeout=1)
            
            Logger.success("TTS 매니저 종료")
    
    def _monitor_loop(self):
        """Global 변수 감시 루프"""
        while self.running:
            try:
                # Global 변수에서 TTS 요청 확인
                if g.speak_flag:
                    g.clear_speak_flag()
                    
                    if not self.speaking_active:
                        tts_text = g.speak_text
                        
                        if tts_text:
                            self.speaking_thread = threading.Thread(
                                target=self._speak,
                                args=(tts_text,),
                                daemon=True,
                                name="SpeakingThread"
                            )
                            self.speaking_thread.start()
                
                time.sleep(0.01)
                
            except Exception as e:
                Logger.error(f"TTS 매니저 오류: {e}")
    
    def _speak(self, text):
        """TTS 실행"""
        self.speaking_active = True
        g.set_speaking_state(True)
        
        try:
            Logger.info(f"로봇: {text}", emoji="🤖")
            self.tts.speak(text)
        except KeyboardInterrupt:
            Logger.info("TTS 사용자 중단")
        except Exception as e:
            Logger.error(f"TTS 오류: {e}")
        finally:
            self.speaking_active = False
            g.set_speaking_state(False)
