from sys import version_info

if version_info.major < 3:
    raise ImportError("byteplay3 requires python 3")

if version_info.minor < 6:
    from .byteplay import *
else:
    from .wbyteplay import *
