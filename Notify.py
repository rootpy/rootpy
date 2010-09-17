import subprocess

def send(hosts,title,message,icon="python"):

    if type(hosts) is not list:
        hosts = [hosts]
    
    cmd = "ssh -X %s 'notify-send -i %s \"%s\" \"%s\"'"
    children = []
    for host in hosts:
        children.append(subprocess.Popen(args=cmd%(host,icon,title,message),shell=True))
    for child in children:
        child.wait()
