# core/wakeword.py
import os
import sys
import warnings
import contextlib


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


@contextlib.contextmanager
def suppress_stderr():
    stderr_fd = sys.stderr.fileno()
    stderr_dup = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(stderr_dup, stderr_fd)
        os.close(devnull_fd)
        os.close(stderr_dup)


with suppress_stderr():
    import tensorflow as tf


warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')


import numpy as np
from utils.logger import Logger


class WakeWordDetector:
    def __init__(self, model_path=None, threshold=None, target_wake_words=None, wake_word_names=None):
        """
        Wake Word 감지기 - config.py의 설정을 기본값으로 사용
        
        Args:
            model_path: .tflite 모델 파일 경로 (None이면 config.py에서 로드)
            threshold: 감지 임계값 0.0~1.0 (None이면 config.py에서 로드)
            target_wake_words: 감지할 wake word 인덱스 리스트 (None이면 config.py에서 로드)
            wake_word_names: wake word 이름 매핑 dict (None이면 config.py에서 로드)
        """
        # config.py에서 기본값 로드
        from configs import (
            WAKE_WORD_MODEL_PATH,
            WAKE_WORD_THRESHOLD,
            WAKE_WORD_TARGETS,
            WAKE_WORD_NAMES
        )
        
        # 인자가 제공되지 않으면 config 값 사용
        self.model_path = model_path or WAKE_WORD_MODEL_PATH
        self.threshold = threshold if threshold is not None else WAKE_WORD_THRESHOLD
        self.target_wake_words = target_wake_words if target_wake_words is not None else WAKE_WORD_TARGETS
        self.wake_word_names = wake_word_names if wake_word_names is not None else WAKE_WORD_NAMES
        
        # Interpreter 생성
        with suppress_stderr():
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 입력 shape 자동 추출
        self.input_shape = self.input_details[0]['shape']
        self.sample_rate = 16000
        
        # ===== 핵심: 모델 입력 타입 자동 감지 =====
        self._detect_model_type()
        
        Logger.debug(f"TFLite 모델 로드 완료: {self.model_path}")
        Logger.debug(f"입력 shape: {self.input_shape}")
        Logger.debug(f"모델 타입: {self.model_type}")
        Logger.debug(f"청크 크기: {self.chunk_size} 샘플 ({self.chunk_size/self.sample_rate:.3f}초)")
        Logger.debug(f"임계값: {self.threshold}")
        
        if self.target_wake_words:
            active_names = [self.wake_word_names.get(i, f"Class{i}") for i in self.target_wake_words]
            Logger.debug(f"활성 wake words: {active_names}")
        else:
            Logger.debug("모든 wake words 활성화")
    
    def _detect_model_type(self):
        """
        모델 입력 shape을 분석하여 모델 타입과 처리 방식 결정
        
        지원 타입:
        1. Raw Audio: [1, samples] - 원시 오디오 샘플
        2. Mel Spectrogram: [1, time_steps, mel_bins] - 오디오 특징
        3. MFCC/Spectrogram: [1, time_steps, features, channels] - 4D 텐서
        """
        shape = self.input_shape
        
        # 타입 1: Raw Audio [1, samples]
        if len(shape) == 2:
            self.model_type = "raw_audio"
            self.buffer_size = shape[1]
            self.chunk_size = 160  # 10ms @ 16kHz
            self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
            Logger.debug(f"Raw Audio 모델 감지 - 버퍼: {self.buffer_size} 샘플")
        
        # 타입 2: Mel Spectrogram [1, time_steps, mel_bins]
        elif len(shape) == 3:
            self.model_type = "mel_spectrogram"
            self.time_steps = shape[1]
            self.mel_bins = shape[2]
            
            # Mel Spectrogram 파라미터 추정
            # 일반적으로 96 mel bins, hop_length=160 (10ms)
            self.hop_length = 160
            self.n_fft = 512
            self.chunk_size = 160
            
            # 필요한 오디오 샘플 수 계산
            self.buffer_size = (self.time_steps - 1) * self.hop_length + self.n_fft
            self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
            
            Logger.debug(f"Mel Spectrogram 모델 감지")
            Logger.debug(f"  - Time steps: {self.time_steps}, Mel bins: {self.mel_bins}")
            Logger.debug(f"  - 오디오 버퍼: {self.buffer_size} 샘플")
        
        # 타입 3: MFCC/4D Tensor [1, time_steps, features, channels]
        elif len(shape) == 4:
            self.model_type = "mfcc_4d"
            self.time_steps = shape[1]
            self.features = shape[2]
            self.channels = shape[3]
            
            # ===== 수정: mel_bins 설정 추가 =====
            # Teachable Machine 오디오 모델은 보통 features가 MFCC 또는 Mel bins
            # features가 mel bins 역할을 함
            self.mel_bins = self.features
            
            # 일반적인 설정
            self.hop_length = 160
            self.n_fft = 512
            self.chunk_size = 160
            
            self.buffer_size = (self.time_steps - 1) * self.hop_length + self.n_fft
            self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
            
            Logger.debug(f"MFCC/4D 모델 감지")
            Logger.debug(f"  - Time steps: {self.time_steps}, Features: {self.features}, Channels: {self.channels}")
            Logger.debug(f"  - Mel bins (features): {self.mel_bins}")
            Logger.debug(f"  - 오디오 버퍼: {self.buffer_size} 샘플")
        
        else:
            raise ValueError(f"지원하지 않는 입력 shape: {shape}")
        
    def _compute_mel_spectrogram(self, audio):
        """
        오디오를 Mel Spectrogram으로 변환
        Args:
            audio: 오디오 샘플 (numpy array)
        Returns:
            Mel spectrogram [time_steps, mel_bins]
        """
        try:
            import librosa
            
            # Mel Spectrogram 계산
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.mel_bins,
                fmin=0,
                fmax=self.sample_rate // 2
            )
            
            # Log scale 변환
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0, 1]
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
            
            # Transpose to [time_steps, mel_bins]
            return mel_spec_norm.T
            
        except ImportError:
            Logger.error("librosa가 설치되지 않았습니다: pip install librosa")
            # Fallback: 간단한 스펙트로그램
            return self._simple_spectrogram(audio)
    
    def _simple_spectrogram(self, audio):
        """
        librosa 없이 간단한 스펙트로그램 계산
        """
        # STFT 계산
        from scipy import signal
        f, t, Zxx = signal.stft(
            audio, 
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length
        )
        
        # Magnitude
        mag = np.abs(Zxx)
        
        # Mel 필터가 없으므로 간단히 리샘플링
        if hasattr(self, 'mel_bins'):
            # 주파수 축을 mel_bins로 리샘플
            from scipy.interpolate import interp1d
            f_interp = interp1d(np.arange(mag.shape[0]), mag, axis=0, kind='linear')
            mag_resampled = f_interp(np.linspace(0, mag.shape[0]-1, self.mel_bins))
            mag = mag_resampled
        
        # Transpose
        return mag.T
    
    def detect_from_stream(self, audio_chunk):
        """오디오 청크로부터 wake word 감지"""
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()
        
        # int16 → float32 변환 (-1.0 ~ 1.0)
        audio_float = audio_chunk.astype(np.float32) / 32768.0
        
        # 순환 버퍼에 추가
        self.audio_buffer = np.roll(self.audio_buffer, -len(audio_float))
        self.audio_buffer[-len(audio_float):] = audio_float
        
        # ===== 모델 타입별 입력 데이터 생성 =====
        try:
            if self.model_type == "raw_audio":
                # Raw audio 그대로 사용
                input_data = self.audio_buffer.reshape(1, -1).astype(np.float32)
            
            elif self.model_type == "mel_spectrogram":
                # Mel Spectrogram 계산
                mel_spec = self._compute_mel_spectrogram(self.audio_buffer)
                
                # Time steps 맞추기
                if mel_spec.shape[0] < self.time_steps:
                    # Padding
                    pad_width = self.time_steps - mel_spec.shape[0]
                    mel_spec = np.pad(mel_spec, ((0, pad_width), (0, 0)), mode='constant')
                elif mel_spec.shape[0] > self.time_steps:
                    # Truncate
                    mel_spec = mel_spec[:self.time_steps, :]
                
                input_data = mel_spec.reshape(1, self.time_steps, self.mel_bins).astype(np.float32)
            
            elif self.model_type == "mfcc_4d":
                # Mel Spectrogram 계산 후 4D로 변환
                mel_spec = self._compute_mel_spectrogram(self.audio_buffer)
                
                # Time steps 맞추기
                if mel_spec.shape[0] < self.time_steps:
                    pad_width = self.time_steps - mel_spec.shape[0]
                    mel_spec = np.pad(mel_spec, ((0, pad_width), (0, 0)), mode='constant')
                elif mel_spec.shape[0] > self.time_steps:
                    mel_spec = mel_spec[:self.time_steps, :]
                
                # Features 차원 맞추기 (필요시 보간)
                if mel_spec.shape[1] != self.features:
                    from scipy.interpolate import interp1d
                    f_interp = interp1d(np.arange(mel_spec.shape[1]), mel_spec, axis=1, kind='linear')
                    mel_spec = f_interp(np.linspace(0, mel_spec.shape[1]-1, self.features))
                
                input_data = mel_spec.reshape(1, self.time_steps, self.features, self.channels).astype(np.float32)
            
            # 추론 수행
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Wake word 점수 추출
            if self.target_wake_words is None:
                wake_word_scores = output_data[0][:-1] if output_data[0].shape[0] > 1 else output_data[0]
                if len(wake_word_scores) == 0:
                    return False
                max_score = np.max(wake_word_scores)
                max_index = np.argmax(wake_word_scores)
            else:
                wake_word_scores = [(i, output_data[0][i]) for i in self.target_wake_words]
                if not wake_word_scores:
                    return False
                max_index, max_score = max(wake_word_scores, key=lambda x: x[1])
            
            # 디버그 로그 (임계값 30% 이상일 때만)
            if max_score > 0.3:
                wake_name = self.wake_word_names.get(max_index, f"Class{max_index}")
                Logger.debug(f"WakeWord {wake_name}: {max_score:.3f}")
            
            # Wake word 감지
            if max_score > self.threshold:
                wake_name = self.wake_word_names.get(max_index, f"Class{max_index}")
                Logger.success(f"'{wake_name}' 감지! (신뢰도: {max_score:.3f})")
                return True
                
        except Exception as e:
            Logger.error(f"WakeWord 추론 실패: {e}")
            import traceback
            Logger.debug(traceback.format_exc())
        
        return False
    
    def reset(self):
        """버퍼 초기화"""
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
