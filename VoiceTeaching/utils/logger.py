# utils/logger.py
"""통합 로깅 시스템"""

import datetime
import sys
import os
from enum import Enum


class LogLevel(Enum):
    """로그 레벨 정의"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class Logger:
    """중앙 집중식 로거"""
    
    CURRENT_LEVEL = LogLevel.INFO
    USE_EMOJI = True
    USE_COLOR = True
    LOG_FILE = None  # 로그 파일 핸들러
    LOG_TO_FILE = False  # 파일 저장 여부
    
    # ANSI 색상 코드
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'RESET': '\033[0m'
    }
    
    @staticmethod
    def _get_time():
        """YYYY-MM-DD HH:MM:SS 형식 타임스탬프"""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def _should_log(level: LogLevel):
        """현재 로그 레벨에서 출력 여부 결정"""
        return level.value >= Logger.CURRENT_LEVEL.value
    
    @staticmethod
    def _format_message(level: str, msg: str, emoji: str = ""):
        """로그 메시지 포맷팅"""
        timestamp = Logger._get_time()
        
        # 이모지 처리
        prefix = f"{emoji} " if emoji and Logger.USE_EMOJI else ""
        
        # 색상 처리 (터미널 지원 시)
        if Logger.USE_COLOR and sys.stdout.isatty():
            color = Logger.COLORS.get(level, '')
            reset = Logger.COLORS['RESET']
            return f"{color}[{timestamp}] [{level:<7}]{reset} {prefix}{msg}"
        else:
            return f"[{timestamp}] [{level:<7}] {prefix}{msg}"
    
    @staticmethod
    def _write_to_file(message: str):
        """로그 파일에 기록"""
        if Logger.LOG_TO_FILE and Logger.LOG_FILE:
            try:
                # 색상 코드 제거
                clean_message = message
                for code in Logger.COLORS.values():
                    clean_message = clean_message.replace(code, '')
                
                Logger.LOG_FILE.write(clean_message + '\n')
                Logger.LOG_FILE.flush()
            except Exception as e:
                print(f"로그 파일 쓰기 오류: {e}")
    
    @staticmethod
    def debug(msg: str, emoji: str = "🔍"):
        """디버그 로그"""
        if Logger._should_log(LogLevel.DEBUG):
            formatted = Logger._format_message("DEBUG", msg, emoji)
            print(formatted)
            Logger._write_to_file(formatted)
    
    @staticmethod
    def info(msg: str, emoji: str = ""):
        """정보 로그"""
        if Logger._should_log(LogLevel.INFO):
            formatted = Logger._format_message("INFO", msg, emoji)
            print(formatted)
            Logger._write_to_file(formatted)
    
    @staticmethod
    def success(msg: str):
        """성공 로그"""
        if Logger._should_log(LogLevel.INFO):
            formatted = Logger._format_message("INFO", msg, "✓")
            print(formatted)
            Logger._write_to_file(formatted)
    
    @staticmethod
    def warning(msg: str):
        """경고 로그"""
        if Logger._should_log(LogLevel.WARNING):
            formatted = Logger._format_message("WARNING", msg, "⚠")
            print(formatted)
            Logger._write_to_file(formatted)
    
    @staticmethod
    def error(msg: str):
        """에러 로그"""
        if Logger._should_log(LogLevel.ERROR):
            formatted = Logger._format_message("ERROR", msg, "✗")
            print(formatted)
            Logger._write_to_file(formatted)
    
    @staticmethod
    def separator(char: str = "─", length: int = 60):
        """구분선 출력"""
        if Logger._should_log(LogLevel.INFO):
            line = char * length
            print(line)
            Logger._write_to_file(line)
    
    @staticmethod
    def section(title: str):
        """섹션 헤더 출력"""
        if Logger._should_log(LogLevel.INFO):
            sep = "─" * 60
            print(f"\n{sep}")
            print(f"  {title}")
            print(f"{sep}")
            Logger._write_to_file(f"\n{sep}")
            Logger._write_to_file(f"  {title}")
            Logger._write_to_file(f"{sep}")
        
    @staticmethod
    def init_file_logging(log_dir: str = "outputs"):
        """
        로그 파일 초기화
        Args:
            log_dir: 로그 디렉토리 (기본값: outputs)
        Returns:
            로그 파일 경로
        """
        try:
            Logger.LOG_TO_FILE = True # 파일 저장 활성화
            
            # YYYY-MM-DD 폴더 생성
            now = datetime.datetime.now()
            date_folder = now.strftime("%Y-%m-%d")
            log_folder = os.path.join(log_dir, date_folder)
            os.makedirs(log_folder, exist_ok=True)
            
            # VoiceModular/outputs/YYYY-MM-DD/[YYYY-MM-DD HH:MM:SS]voice_system.log로 저장
            timestamp = now.strftime("%Y-%m-%d %H-%M-%S") 
            log_filename = f"[{timestamp}]voice_system.log"
            log_path = os.path.join(log_folder, log_filename)
            
            # 파일 핸들러 열기
            Logger.LOG_FILE = open(log_path, 'w', encoding='utf-8')
            
            return log_path
        except Exception as e:
            print(f"로그 파일 초기화 실패: {e}")
            Logger.LOG_TO_FILE = False
            return None
    
    @staticmethod
    def close_file_logging():
        """로그 파일 닫기"""
        if Logger.LOG_FILE:
            try:
                Logger.LOG_FILE.close()
                Logger.LOG_FILE = None
            except Exception as e:
                print(f"로그 파일 닫기 오류: {e}")


# 전역 설정 함수
def set_log_level(level: str):
    """로그 레벨 설정"""
    level_map = {
        "DEBUG": LogLevel.DEBUG,
        "INFO": LogLevel.INFO,
        "WARNING": LogLevel.WARNING,
        "ERROR": LogLevel.ERROR
    }
    Logger.CURRENT_LEVEL = level_map.get(level.upper(), LogLevel.INFO)


def configure_logger(use_emoji: bool = True, use_color: bool = True, log_to_file: bool = False):
    """로거 설정"""
    Logger.USE_EMOJI = use_emoji
    Logger.USE_COLOR = use_color
    Logger.LOG_TO_FILE = log_to_file
