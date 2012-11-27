from pkg_resources import resource_filename
from .io import File

filename = resource_filename('rootpy', 'etc/test_file.root')
testfile = File(filename, 'read')
