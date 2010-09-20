import cPickle
import sys
import os
import commands

display = os.environ["DISPLAY"]
path = os.path.join(os.environ["PYROOT_CONFIG_ROOT"],'env')

def define(name):

    if not os.path.exists(path):
        os.makedirs(path)
    env = {"System":os.environ,"Python":sys.path}
    file = open((os.path.join(path,"%s.env"))%name,'wb')
    cPickle.dump(env, file)
    file.close()

def load(name):
    
    envfile = (os.path.join(path,"%s.env"))%name
    if not os.path.exists(envfile):
        print "Environment %s is not defined."%name
        return False
    file = open(envfile,'rb')
    newEnv = cPickle.load(file)
    file.close()
    currentPython = commands.getoutput("which python")
    os.environ.clear()
    os.environ.update(newEnv["System"])
    if os.environ.has_key("RELEASE"):
        print "Athena %s environment is ready"%os.environ["RELEASE"]
        if os.environ.has_key("TESTAREA"):
            print "using testarea %s"%os.environ["TESTAREA"]
    os.environ["DISPLAY"] = display
    newPython = commands.getoutput("which python")
    sys.path = newEnv["Python"]
    print "Environment %s has been loaded."%name
    if newPython != currentPython:
        print "Warning! This environment uses a different version of Python."
        return newPython
    return True
