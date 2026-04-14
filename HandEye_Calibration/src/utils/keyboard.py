import sys

def read_key():
    try:
        import tty, termios  # Unix (Linux, macOS)
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == '\x1b':  # 방향키 같은 escape sequence
                ch += sys.stdin.read(2)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    except ImportError:
        import msvcrt  # Windows
        ch = msvcrt.getwch()
        if ch in ('\x00', '\xe0'):  # 특수키 (방향키 등)
            ch += msvcrt.getwch()
        return ch

print("⚠ NumLock을 켜고 사용하세요")
print("키를 누르면 출력됩니다. (종료: q)")

while True:
    key = read_key()
    print(f"입력: {repr(key)}")

    if key in ('q', 'Q'):
        print("종료")
        break