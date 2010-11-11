""" Locate a file in a PATH """
import os

def find_file(filename, search_path_var='PATH', include_working=True):
    """
    find filename in PATH with the option of including
    the current working directory in the search
    """
    if not os.environ.has_key(search_path_var):
        return None
    search_path = os.environ[search_path_var]
    paths = search_path.split(os.pathsep)
    if include_working:
        paths = ['.'] + paths
    for path in paths:
        fullpath = os.path.join(path, filename)
        if os.path.exists(fullpath):
            return os.path.abspath(fullpath)
    return None
