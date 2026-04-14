# main.py
"""
음성 제어 시스템 통합 진입점

역할:
    - STT, LLM, TTS 엔진을 통합하여 음성 제어 시스템 구동
    - 3가지 입력 방식: 마이크(VAD/Manual/Wakeword), 텍스트
    - 명령행 인자로 엔진 선택 가능 (기본값은 config.py 사용)

사용법:
    # 기본 실행 (config.py 설정 사용)
    python main.py
    
    # 입력 방식 선택
    python main.py --input mic --mode vad          # 마이크 + 자동 감지
    python main.py --input mic --mode manual       # 마이크 + 엔터키
    python main.py --input mic --mode wakeword     # Wake word 감지
    python main.py --input text                    # 텍스트 입력
    
    # 엔진 선택
    python main.py --stt google --llm gemini --tts clova
    python main.py --stt whisper --llm gpt --tts openai
    
    # 조합 예시
    python main.py --input mic --mode wakeword --stt google --llm gemini --tts clova
"""
import os
import sys
import warnings

# Warning 필터링
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='tensorflow')
warnings.filterwarnings('ignore', module='keras')
warnings.filterwarnings('ignore', message='.*tf.lite.Interpreter is deprecated.*')
warnings.filterwarnings('ignore', message='.*np.object.*')

import asyncio
import argparse
import time

from core import get_stt_fn, get_llm_fn, AudioRecorder
from modules.tts_modules import TTSManager
from utils.robot_commands import execute_command
from configs import globals as g
from utils import Logger
from configs import STT_MODEL, LLM_MODEL, TTS_MODEL, INPUT_MODE


class VoiceControlSystem:
    """음성 제어 시스템 메인 클래스"""
    
    def __init__(self, args):
        self.args = args
        self.stt_fn = None
        self.llm_fn = None
        self.tts_manager = None
        self.recorder = None
    
    async def initialize(self):
        """시스템 초기화"""
        self.stt_fn = get_stt_fn(self.args.stt)
        self.llm_fn = get_llm_fn(self.args.llm)
        
        self.tts_manager = TTSManager(tts_engine=self.args.tts)
        self.tts_manager.start()
        
        self.recorder = AudioRecorder()
        
        self._print_startup_info()
    
    def _print_startup_info(self):
        """시스템 시작 정보 출력"""
        Logger.section("음성 제어 시스템 시작")
        Logger.info(f"입력 소스: {self.args.input}")
        
        if self.args.input == "mic":
            Logger.info(f"마이크 모드: {self.args.mode}")
        
        Logger.info(f"STT 엔진: {self.args.stt}")
        Logger.info(f"LLM 엔진: {self.args.llm}")
        Logger.info(f"TTS 엔진: {self.args.tts}")
        Logger.separator()
    
    async def run(self):
        """메인 실행 루프"""
        if self.args.input == "mic":
            if self.args.mode == "wakeword":
                await self._wakeword_loop()
            else:
                await self._mic_loop()
        else:
            await self._text_loop()
    
    def cleanup(self):
        """시스템 종료"""
        if self.tts_manager:
            self.tts_manager.stop()
        Logger.success("시스템 종료 완료")
    
    async def _wakeword_loop(self):
        """Wake word 감지 루프"""
        from core import WakeWordDetector
        import sounddevice as sd
        from configs import SAMPLE_RATE, WAKE_WORD_NAMES, WAKE_WORD_TARGETS
        
        # config.py 설정을 자동으로 사용 (인자 없이 생성)
        detector = WakeWordDetector()
        
        # 동적 청크 크기 사용
        chunk_size = detector.chunk_size
        
        # 활성화된 wake word 이름 출력
        if WAKE_WORD_TARGETS:
            wake_names = [WAKE_WORD_NAMES.get(i, f'Class{i}') for i in WAKE_WORD_TARGETS]
            wake_display = ', '.join(wake_names)
        else:
            wake_display = 'any wake word'
        
        Logger.info(f"Wake word 대기 중... '{wake_display}'라고 말해보세요", emoji="🎧")
        
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
                while True:
                    # 동적 청크 크기로 읽기
                    audio_chunk, _ = stream.read(chunk_size)
                    
                    if detector.detect_from_stream(audio_chunk):
                        Logger.info("Wake word 감지!", emoji="✨")
                        
                        g.set_speak_request("네")
                        await asyncio.sleep(0.5)
                        
                        audio_path = self.recorder.record_to_file("user_voice.wav", mode="manual")
                        await self._process_voice_command(audio_path)
                        
                        detector.reset()
                        Logger.info(f"Wake word 대기 중... '{wake_display}'라고 말해보세요", emoji="🎧")
                        Logger.separator()
                        
        except Exception as e:
            Logger.error(f"Wake word 루프 오류: {e}")
            
    async def _mic_loop(self):
        """마이크 입력 루프 (VAD 또는 Manual)"""
        while True:
            try:
                audio_path = self.recorder.record_to_file("user_voice.wav", mode=self.args.mode)
                await self._process_voice_command(audio_path)
                Logger.separator()
                
            except KeyboardInterrupt:
                Logger.info("\n사용자 종료 요청")
                break
            except Exception as e:
                Logger.error(f"마이크 처리 오류: {e}")
                await asyncio.sleep(1)

    async def _text_loop(self):
        """텍스트 입력 루프"""
        while True:
            try:
                user_text = input("\n👤 텍스트 입력: ")
                
                if not user_text:
                    continue
                
                Logger.info(f"사용자: {user_text}", emoji="🎤")
                
                # LLM 처리
                res_data = await self.llm_fn(user_text)
                
                # LLM JSON 응답 출력
                Logger.info(f"LLM 응답: {res_data}", emoji="🤖")
                
                # Global 변수에 저장
                g.set_voice_command(res_data)
                
                # 로봇 명령 실행
                action = res_data.get("action")
                if action:
                    result = execute_command(action)
                    if result:
                        Logger.success(result)
                        Logger.info(f"현재 로봇 모드: {g.get_robot_mode()}", emoji="🔧")
                else:
                    Logger.warning("실행 가능한 명령이 없습니다")
                
                # TTS 요청 및 완료 대기
                feedback = res_data.get("description", "")
                if feedback:
                    g.set_speak_request(feedback)
                    await self._wait_for_tts()
                
                Logger.separator()
                
            except EOFError:
                Logger.info("\n입력 종료")
                break
            except Exception as e:
                Logger.error(f"텍스트 처리 오류: {e}")
                await asyncio.sleep(1)

    async def _process_voice_command(self, audio_path):
        """공통 음성 명령 처리 로직"""
        user_text = await asyncio.to_thread(self.stt_fn, audio_path)
        
        if not user_text:
            return
        
        Logger.info(f"사용자: {user_text}", emoji="🎤")
        
        # LLM 처리
        res_data = await self.llm_fn(user_text)
        
        # LLM JSON 응답 출력
        Logger.info(f"LLM 응답: {res_data}", emoji="🤖")
        
        # Global 변수에 저장
        g.set_voice_command(res_data)
        
        # 로봇 명령 실행
        action = res_data.get("action")
        if action:
            result = execute_command(action)
            if result:
                Logger.success(result)
                Logger.info(f"현재 로봇 모드: {g.get_robot_mode()}", emoji="🔧")
        else:
            Logger.warning("실행 가능한 명령이 없습니다")
        
        # TTS 요청 및 완료 대기
        feedback = res_data.get("description", "")
        if feedback:
            g.set_speak_request(feedback)
            await self._wait_for_tts()

    async def _wait_for_tts(self, timeout=10):
        """TTS 완료 대기"""
        start_time = time.time()
        
        for _ in range(10):
            if g.get_speaking_state():
                break
            await asyncio.sleep(0.1)
        
        while g.get_speaking_state():
            await asyncio.sleep(0.1)
            if time.time() - start_time > timeout:
                Logger.warning("TTS 타임아웃")
                break
        
        await asyncio.sleep(0.2)


def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="모듈화된 음성 제어 시스템",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input", 
        choices=["mic", "text"], 
        default="mic",
        help="입력 소스: mic (마이크) / text (키보드)"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["vad", "manual", "wakeword"], 
        default=INPUT_MODE,
        help="마이크 모드: vad (자동) / manual (엔터키) / wakeword (Wake word)"
    )
    
    parser.add_argument(
        "--stt", 
        choices=["whisper", "google", "clova"], 
        default=STT_MODEL,
        help="STT 엔진 선택"
    )
    
    parser.add_argument(
        "--llm", 
        choices=["gemini", "gpt"], 
        default=LLM_MODEL,
        help="LLM 엔진 선택"
    )
    
    parser.add_argument(
        "--tts", 
        choices=["google", "openai", "clova"], 
        default=TTS_MODEL,
        help="TTS 엔진 선택"
    )
    
    return parser.parse_args()


async def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    # 로그 파일 초기화 (main에서만)
    from configs import LOG_TO_FILE, LOG_OUTPUT_DIR
    
    log_path = None
    if LOG_TO_FILE:
        log_path = Logger.init_file_logging(LOG_OUTPUT_DIR)
        if log_path:
            Logger.debug(f"로그 파일: {log_path}", emoji="📝")
    
    system = VoiceControlSystem(args)
    
    try:
        await system.initialize()
        await system.run()
    except KeyboardInterrupt:
        Logger.info("\n사용자 종료 요청")
    finally:
        system.cleanup()
        # 로그 파일 닫기
        Logger.close_file_logging()


if __name__ == "__main__":
    asyncio.run(main())
