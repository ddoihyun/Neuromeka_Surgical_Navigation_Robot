# utils/__init__.py
"""유틸리티 모듈"""

from utils.logger import Logger, set_log_level, configure_logger

# config.py 로드하여 로그 설정 자동 적용
try:
    from configs import LOG_LEVEL, LOG_USE_EMOJI, LOG_USE_COLOR
    set_log_level(LOG_LEVEL)
    configure_logger(use_emoji=LOG_USE_EMOJI, use_color=LOG_USE_COLOR)
except ImportError:
    # config.py가 없거나 순환 import 발생 시 기본값 사용
    pass

__all__ = ['Logger']
