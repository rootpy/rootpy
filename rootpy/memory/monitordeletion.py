# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License

from weakref import ref

def monitordeletion():
    """
    Function for checking for correct deletion of weakref-able objects.
    
    Example usage:
    
        monitor, is_alive = deletionmonitor()
        
        obj = set()
        
        monitor(obj, "obj")
        
        assert is_alive("obj") # True because there is a ref to `obj` is_alive
        
        del obj
        
        assert not is_alive("obj") # True because there `obj` is deleted
    """
    
    monitors = {}

    def set_deleted(x):
        def _(weakref):
            del monitors[x]
        return _
        
    def monitor(item, name):
        monitors[name] = ref(item, set_deleted(name))
        
    def is_alive(name):
        return monitors.get(name, None) is not None
        
    return monitor, is_alive

