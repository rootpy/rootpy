import socket

def my_ip():

    return [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][0]

def scan(subnet = '192.168.1', port = 50000):
    
    s = socket.socket()
    s.settimeout(0.1)
    for ip in xrange(1,256):
        addr = "%s.%s"% (subnet, ip)
        try:
            s.connect((addr,port))
            yield addr
        except: pass
    s.close()

if __name__ == "__main__":
    ip = my_ip()
    for addr in scan('.'.join(ip.split('.')[0:-1])):
        print addr
