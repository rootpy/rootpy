from pkg_resources import resource_filename
from .io import File

filename = resource_filename('rootpy', 'etc/testfile.root')
testfile = File(filename, 'read')
