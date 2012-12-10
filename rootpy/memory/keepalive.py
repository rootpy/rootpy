import weakref

KEEPALIVE = weakref.WeakKeyDictionary()

def keepalive(nurse, *patients):
    KEEPALIVE.setdefault(nurse, set()).update(patients)
