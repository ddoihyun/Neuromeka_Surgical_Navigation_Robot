# core/recorder.py
"""
오디오 녹음 모듈
- VAD (Voice Activity Detection) 기반 자동 녹음
- Manual 모드 (엔터키 제어)
"""
import os
import sounddevice as sd
import numpy as np
import webrtcvad
import collections
import wave
import threading
from configs import SAMPLE_RATE, CHUNK_SIZE, OUTPUT_DIR
from utils.logger import Logger


class AudioRecorder:
    """오디오 녹음 관리 클래스"""
    
    def __init__(self):
        self.vad = webrtcvad.Vad(3)
        self.ENERGY_THRESHOLD = 500
        
        # assets 폴더 자동 생성
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        Logger.debug("AudioRecorder 초기화 완료")
    
    def calculate_rms(self, frame):
        """
        오디오 프레임의 에너지(크기)를 계산
        Args:
            frame: 오디오 프레임 (numpy array)
        Returns:
            RMS 값
        """
        return np.sqrt(np.mean(frame.astype(np.float32)**2))
    
    def record_to_file(self, output_path="user_voice.wav", mode="vad"):
        """
        오디오 녹음 (VAD 또는 Manual 모드)
        Args:
            output_path: 출력 파일명 (상대 경로면 assets/에 저장)
            mode: "vad" 또는 "manual"
        Returns:
            저장된 파일의 전체 경로
        """
        # 상대 경로면 OUTPUT_DIR에 저장
        if not os.path.isabs(output_path):
            output_path = os.path.join(OUTPUT_DIR, output_path)
        
        if mode == "manual":
            return self._record_manual(output_path)
        else:
            return self._record_vad(output_path)
    
    def _record_manual(self, output_path):
        """
        수동 녹음 모드 (엔터키로 시작/종료)
        Args:
            output_path: 출력 파일 전체 경로
        Returns:
            저장된 파일 경로
        """
        input("\n🔔 [Manual] 엔터(Enter)를 누르면 '녹음 시작'합니다...")
        recording_data = []
        stop_recording = False
        
        def wait_for_stop():
            nonlocal stop_recording
            input("🔴 녹음 중... 종료하려면 엔터(Enter)를 누르세요.\n")
            stop_recording = True
        
        stop_thread = threading.Thread(target=wait_for_stop)
        stop_thread.start()
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
            while not stop_recording:
                frame, _ = stream.read(CHUNK_SIZE)
                recording_data.append(frame)
        
        stop_thread.join()
        Logger.info("녹음 완료 및 저장 중...", emoji="⏹️")
        audio_data = np.concatenate(recording_data)
        self._save_wav(audio_data, output_path)
        return output_path
    
    def _record_vad(self, output_path):
        """
        VAD 모드 녹음 (자동 침묵 감지)
        Args:
            output_path: 출력 파일 전체 경로
        Returns:
            저장된 파일 경로
        """
        Logger.info("음성 대기 중...", emoji="🎤")
        
        ring_buffer = collections.deque(maxlen=30)
        speech_buffer = []
        triggered = False
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
            while True:
                frame, _ = stream.read(CHUNK_SIZE)
                rms = self.calculate_rms(frame)
                
                is_speech = False
                if rms > self.ENERGY_THRESHOLD:
                    is_speech = self.vad.is_speech(frame.tobytes(), SAMPLE_RATE)
                
                if not triggered:
                    ring_buffer.append((frame, is_speech))
                    if len([f for f, s in ring_buffer if s]) > 0.8 * ring_buffer.maxlen:
                        triggered = True
                        Logger.success("음성 감지, 녹음 시작")
                        speech_buffer.extend([f for f, s in ring_buffer])
                        ring_buffer.clear()
                else:
                    speech_buffer.append(frame)
                    ring_buffer.append((frame, is_speech))
                    if len([f for f, s in ring_buffer if not s]) > 0.9 * ring_buffer.maxlen:
                        Logger.info("침묵 감지, 녹음 종료")
                        break
        
        audio_data = np.concatenate(speech_buffer)
        
        # 너무 짧은 녹음은 재시도
        if len(audio_data) < SAMPLE_RATE * 0.5:
            Logger.warning("녹음이 너무 짧습니다, 재시도합니다")
            return self._record_vad(output_path)
        
        self._save_wav(audio_data, output_path)
        return output_path
    
    def _save_wav(self, audio_data, output_path):
        """
        오디오 데이터를 WAV 파일로 저장
        Args:
            audio_data: numpy 오디오 배열
            output_path: 출력 파일 전체 경로
        """
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        Logger.debug(f"오디오 저장 완료: {output_path}")
