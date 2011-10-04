import socket


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
            pass
    s.close()


if __name__ == "__main__":

    ip = my_ip()
    print "My ip is: %s" % ip
    print "Scanning local network for other devices..."
    for addr in scan('.'.join(ip.split('.')[0:-1])):
        print "Found device at %s" % addr
