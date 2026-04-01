# utils/ - 유틸리티 모듈

로깅, 로봇 명령 실행 등 공통 기능

---

## 📄 logger.py - 로깅 시스템

### Logger.debug(msg, emoji) → None
: 디버그 로그 출력 (LOG_LEVEL=DEBUG일 때만)
```python
from utils.bridge import Logger

Logger.debug("STT 처리 시작")
```
### Logger.info(msg, emoji) → None
: 정보 로그 출력
```python
Logger.info("시스템 시작", emoji="🚀")
```
### Logger.success(msg) → None
: 성공 로그 (✓ 자동 추가)
```python
Logger.success("녹음 완료")
```
### Logger.warning(msg) → None
: 경고 로그 (⚠ 자동 추가)
```python
Logger.warning("파일 없음")
```
### Logger.error(msg) → None
: 에러 로그 (✗ 자동 추가)
```python
Logger.error("STT 실패")
```
### Logger.separator(char, length) → None
: 구분선 출력
```python
Logger.separator()  # ────────────────...
```
### Logger.section(title) → None
: 섹션 헤더 출력
```python
Logger.section("음성 제어 시작")
```

---
## 📄 robot_commands.py - 로봇 명령 실행

### set_mode(mode) → str | None
: 로봇 모드 변경 ("fix" | "free" | "point" | "line")
```python
from utils.robot_commands import set_mode

result = set_mode("point")
print(result)  # "점 고정 모드로 전환"
```
### get_current_mode() → str
: 현재 로봇 모드 조회
```python
from utils.robot_commands import get_current_mode

mode = get_current_mode()
```
### move_robot(dx, dy) → str
: move_robot(dx, dy) → str
```python
from utils.robot_commands import move_robot

result = move_robot(1.0, 0.5)
```
### stop_robot() → str
: 로봇 정지
``` python
from utils.robot_commands import stop_robot

stop_robot()
```