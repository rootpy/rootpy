"""
Pickle your current shell environment for later use.
"""
import cPickle
import sys
import os
import commands

DISPLAY = os.environ["DISPLAY"]
PATH = os.path.join(os.environ["ROOTPY_CONFIG_ROOT"], 'env')

def define(name):
    """
    Define an environment
    """
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    env = {"System":os.environ, "Python":sys.path}
    env_file = open((os.path.join(PATH, "%s.env"))% name,'wb')
    cPickle.dump(env, env_file)
    env_file.close()

def load(name):
    """
    Load a previously pickled environment
    """
    env_filename = (os.path.join(PATH, "%s.env"))% name
    if not os.path.exists(env_filename):
        print "Environment %s is not defined."% name
        return False
    env_file = open(env_filename, 'rb')
    new_env = cPickle.load(file)
    env_file.close()
    current_python = commands.getoutput("which python")
    os.environ.clear()
    os.environ.update(new_env["System"])
    os.environ["DISPLAY"] = DISPLAY
    new_python = commands.getoutput("which python")
    sys.path = new_env["Python"]
    print "Environment %s has been loaded."% name
    if new_python != current_python:
        print "Warning! This environment uses a different version of Python."
        return new_python
    return True
