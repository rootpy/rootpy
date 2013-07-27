import os

def check_tty(stream):
    if not hasattr(stream, 'fileno'):
        return False
    try:
        fileno = stream.fileno()
        return os.isatty(fileno)
    except (OSError, IOError):
        return False
