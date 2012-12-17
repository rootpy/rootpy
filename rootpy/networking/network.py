# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import socket
from .. import log; log = log[__name__]

def my_ip():

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('google.com', 0))
    return s.getsockname()[0]


def scan(subnet='192.168.1', port=50000):

    s = socket.socket()
    s.settimeout(0.1)
    for ip in xrange(1, 256):
        addr = "%s.%s" % (subnet, ip)
        try:
            s.connect((addr, port))
            yield addr
        except:
            import sys
            exc_type, _, _ = sys.exc_info()
            log.error("BUG: overly broad exception catch. "
                      "Please report this: '{0}'".format(exc_type))
            pass
    s.close()


if __name__ == "__main__":

    ip = my_ip()
    print "My ip is: %s" % ip
    print "Scanning local network for other devices..."
    for addr in scan('.'.join(ip.split('.')[0:-1])):
        print "Found device at %s" % addr
