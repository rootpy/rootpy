try:
    from .byteplay import *
except (ImportError, SyntaxError):
    from .byteplay3 import *
