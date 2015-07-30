# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import, print_function

import os
import sys
import readline
import cmd
import subprocess
import re
from fnmatch import fnmatch
from glob import glob
import traceback
# Try and get the termcolor module - pip install termcolor
try:
    from termcolor import colored
except ImportError:
    # Make a dummy function which does not color the text
    def colored(text, *args, **kwargs):
        return text

from .extern.argparse import ArgumentParser

from . import ROOT
from . import log; log = log[__name__]
log.basic_config_colorized()
from . import __version__
from .io import root_open, DoesNotExist
from .io.file import _DirectoryBase
from .userdata import DATA_ROOT
from .plotting import Canvas
from .plotting.style import set_style
from .logger.utils import check_tty

__all__ = [
    'ROOSH',
]

EXEC_CMD = re.compile('(?P<name>\w+)\.(?P<call>\S+)')
ASSIGN_CMD = re.compile('\w+\s*((\+=)|(-=)|(=))\s*\w+')
GET_CMD = re.compile('^(?P<name>\S+)(:?\s+as\s+(?P<alias>\S+))?$')

_COLOR_MATCHER = [
    (re.compile('^TH[123][CSIDF]'), 'red'),
    (re.compile('^TTree'), 'green'),
    (re.compile('^TChain'), 'green'),
    (re.compile('^TDirectory'), 'blue'),
]


def color_key(tkey):
    """
    Function which returns a colorized TKey name given its type
    """
    name = tkey.GetName()
    classname = tkey.GetClassName()
    for class_regex, color in _COLOR_MATCHER:
        if class_regex.match(classname):
            return colored(name, color=color)
    return name


def prompt(vars, message):
    prompt_message = message
    try:
        from IPython.Shell import IPShellEmbed
        ipshell = IPShellEmbed(
            argv=[''],
            banner=prompt_message, exit_msg="Goodbye")
        return ipshell
    except ImportError:
        # this doesn't quite work right, in that it doesn't go to the right env
        # so we just fail.
        import code
        import rlcompleter
        readline.parse_and_bind("tab: complete")
        # calling this with globals ensures we can see the environment
        if prompt_message:
            print(prompt_message)
        shell = code.InteractiveConsole(vars)
        return shell.interact


def ioctl_GWINSZ(fd):
    try:
        import fcntl
        import termios
        import struct
        cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
    except:
        return None
    return cr


def get_terminal_size():
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        try:
            cr = (env['LINES'], env['COLUMNS'])
        except:
            cr = (25, 80)
    return int(cr[1]), int(cr[0])


def make_identifier(name):
    # Replace invalid characters with '_'
    name = re.sub('[^0-9a-zA-Z_]', '_', name)
    # Remove leading characters until we find a letter or underscore
    return re.sub('^[^a-zA-Z_]+', '', name)


def is_valid_identifier(name):
    return name == make_identifier(name)


class shell_cmd(cmd.Cmd, object):

    def do_shell(self, s):
        subprocess.call(s, shell=True)

    def help_shell(self):
        print("execute commands in your $SHELL (i.e. bash)")


class empty_cmd(cmd.Cmd, object):

    def emptyline(self):
        pass


class exit_cmd(cmd.Cmd, object):

    def can_exit(self):
        return True

    def onecmd(self, line):
        r = super(exit_cmd, self).onecmd(line)
        if (r and (self.can_exit() or
                   raw_input('exit anyway ? (yes/no):') == 'yes')):
            return True
        return False

    def do_exit(self, s):
        return True

    def help_exit(self):
        print("Exit the interpreter.")
        print("You can also use the Ctrl-D shortcut.")

    def do_EOF(self, s):
        if not self.script:
            print()
        return True

    help_EOF = help_exit


def root_glob(directory, pattern):

    matches = []
    for dirpath, dirnames, filenames in \
            directory.walk(maxdepth=pattern.count(os.path.sep)):
        for dirname in dirnames:
            dirname = os.path.join(dirpath, dirname)
            if fnmatch(dirname, pattern):
                matches.append(dirname)
        for filename in filenames:
            filename = os.path.join(dirpath, filename)
            if fnmatch(filename, pattern):
                matches.append(filename)
    return matches


def show_exception(e, debug=False, show_type=False):

    if debug:
        traceback.print_exception(*sys.exc_info())
    elif show_type:
        print("{0}: {1}".format(e.__class__.__name__, e))
    else:
        print(e)


class LazyNamespace(dict):

    def __init__(self, roosh):
        self.roosh = roosh
        super(LazyNamespace, self).__init__()

    def __getitem__(self, key):
        if key in self.roosh.pwd:
            value = self.roosh.pwd[key]
            self.__setitem__(key, value)
            return value
        try:
            return super(LazyNamespace, self).__getitem__(key)
        except KeyError as e:
            if key == 'P':
                pad = ROOT.gPad.func()
                if pad:
                    return pad
                raise
            elif key == 'C':
                pad = ROOT.gPad.func()
                if pad:
                    return pad.GetCanvas()
                raise
            elif key == 'D':
                return self.roosh.pwd
            if key in __builtins__:
                return __builtins__[key]
            try:
                return getattr(ROOT, key)
            except AttributeError:
                pass
            raise e


class ROOSH(exit_cmd, shell_cmd, empty_cmd):

    ls_parser = ArgumentParser()
    ls_parser.add_argument('-l', action='store_true',
                           dest='showtype', default=False)
    ls_parser.add_argument('files', nargs='*')

    mkdir_parser = ArgumentParser()
    mkdir_parser.add_argument('-p', action='store_true',
                              dest='recurse', default=False,
                              help="create parent directories as required")
    mkdir_parser.add_argument('paths', nargs='*')

    def __init__(self, filename, mode='READ',
                 stdin=None, stdout=None,
                 script=False,
                 debug=False):
        if stdin is None:
            stdin = sys.stdin
        if stdout is None:
            stdout = sys.stdout
        super(ROOSH, self).__init__(stdin=stdin, stdout=stdout)

        self.script = script
        self.debug = debug

        root_file = root_open(filename, mode)
        self.files = {}
        self.files[filename] = root_file
        self.pwd = root_file
        self.prev_pwd = root_file
        self.current_file = root_file

        self.namespace = LazyNamespace(self)
        if script:
            self.prompt = ''
        else:
            self.__update_prompt()

    def __update_prompt(self):
        if self.script:
            return
        dirname = os.path.basename(self.pwd._path)
        if len(dirname) > 20:
            dirname = (dirname[:10] + '..' + dirname[-10:])
        self.prompt = '{0} > '.format(dirname)

    def do_env(self, s):
        for name, value in self.namespace.items():
            if name == '__builtins__':
                continue
            print("{0}\t{1}".format(name, value))

    def help_env(self):
        print("print all variable names and values in current environment")
        print("object (excluding directories) contained within the")
        print("current directory are automatically included by name")

    def do_get(self, name):
        try:
            match = re.match(GET_CMD, name)
            if match:
                name = match.group('name')
                alias = match.group('alias')
                if alias is None:
                    alias = make_identifier(os.path.basename(name))
                # check that alias is a valid identifier
                elif not is_valid_identifier(alias):
                    print("{0} is not a valid identifier".format(alias))
                    return
                self.namespace[alias] = self.pwd.Get(name)
            else:
                self.default(name)
        except DoesNotExist as e:
            show_exception(e, debug=self.debug)

    def complete_get(self, text, line, begidx, endidx):
        return self.completion_helper(text, line, begidx, endidx)

    def help_get(self):
        print(
            "load the specified object into the current namespace\n"
            "Use 'get foo as bar' to alias the object named foo as bar")

    def do_cd(self, path):
        prev_pwd = self.pwd
        if path == '.':
            return
            self.prev_pwd = self.pwd
        try:
            if not path:
                self.pwd = self.current_file
            elif path == '-':
                self.pwd = self.prev_pwd
                self.do_pwd()
            else:
                self.pwd = self.pwd.GetDirectory(path)
            self.pwd.cd()
            self.__update_prompt()
            self.prev_pwd = prev_pwd
        except DoesNotExist as e:
            show_exception(e, debug=self.debug)

    def complete_cd(self, text, line, begidx, endidx):
        return self.completion_helper(
            text, line, begidx, endidx, 'TDirectoryFile')

    def help_cd(self):
        print(
            "change the current directory\n"
            "'cd -' will change to the previous directory\n"
            "'cd' (with no path) will change to the root directory\n")

    def do_ls(self, args=None):
        if args is None:
            args = ''
        args = ROOSH.ls_parser.parse_args(args.split())
        if not args.files:
            args.files = ['']
        for i, path in enumerate(args.files):
            if '*' in path:
                paths = root_glob(self.pwd, path)
                if not paths:
                    paths = [path]
            else:
                paths = [path]
            for path in paths:
                _dir = self.pwd
                if path:
                    try:
                        _dir = self.pwd.Get(path)
                    except DoesNotExist as e:
                        show_exception(e, debug=self.debug)
                        continue
                if isinstance(_dir, _DirectoryBase):
                    if len(args.files) > 1:
                        if i > 0:
                            print()
                        print("{0}:".format(_dir.GetName()))
                    keys = _dir.keys(latest=True)
                    keys.sort(key=lambda key: key.GetName())
                    things = [color_key(key) for key in keys]
                    if things:
                        self.columnize(things)
                else:
                    print(path)

    def complete_ls(self, text, line, begidx, endidx):
        return self.completion_helper(text, line, begidx, endidx)

    def help_ls(self):
        print("list items contained in a directory")

    def do_mkdir(self, args=None):
        if args is None:
            args = ''
        args = ROOSH.mkdir_parser.parse_args(args.split())

        for path in args.paths:
            try:
                self.pwd.mkdir(path, recurse=args.recurse)
            except Exception as e:
                show_exception(e, debug=self.debug)

    def complete_mkdir(self, text, line, begidx, endidx):
        return self.completion_helper(text, line, begidx, endidx,
                                      typename='TDirectoryFile')

    def do_rm(self, path):
        try:
            self.pwd.rm(path)
        except Exception as e:
            show_exception(e, debug=self.debug)

    def complete_rm(self, text, line, begidx, endidx):
        return self.completion_helper(text, line, begidx, endidx)

    def do_cp(self, args):
        try:
            thing, dest = args.split()
            self.pwd.copytree(dest, src=thing)
        except Exception as e:
            show_exception(e, debug=self.debug)

    def complete_cp(self, text, line, begidx, endidx):
        return self.completion_helper(text, line, begidx, endidx)

    def completion_helper(self, text, line, begidx, endidx, typename=None):
        things = []
        directory = self.pwd
        head = ''
        if begidx != endidx:
            prefix = line[begidx: endidx]
            head, prefix = os.path.split(prefix)
            if head:
                try:
                    directory = directory.GetDirectory(head)
                except DoesNotExist:
                    return []
        else:
            prefix = ''
        for key in directory.GetListOfKeys():
            if typename is not None:
                if key.GetClassName() != typename:
                    continue
            name = key.GetName()
            if prefix and not name.startswith(prefix):
                continue
            if key.GetClassName() == 'TDirectoryFile':
                things.append(os.path.join(head, '{0}/'.format(name)))
            else:
                things.append(os.path.join(head, name))
        return things

    def do_pwd(self, s=None):
        print(self.pwd._path)

    def help_pwd(self):
        print("print the current directory")

    def help_help(self):
        print("'help CMD' will print help for a command")
        print("'help' will print all available commands")

    def do_python(self, s=None):
        prompt(self.namespace, '')()

    def help_python(self):
        print("drop into an interactive Python shell")
        print("anything loaded into your current namespace")
        print("will be handed over to Python")

    @property
    def current_pad(self):
        pad = ROOT.gPad.func()
        if pad:
            return pad
        return None

    @property
    def current_canvas(self):
        pad = self.current_pad
        if pad:
            return pad.GetCanvas()
        return None

    @property
    def canvases(self):
        return ROOT.gROOT.GetListOfCanvases()

    def do_canvas(self, name=None):
        current_pad = self.current_pad
        current_canvas = self.current_canvas
        canvases = self.canvases

        if not name:
            # print list of existing canvases
            if not current_pad:
                print("no canvases exist, create a new one by "
                      "specifying name: canvas mycanvas")
                return
            for c in canvases:
                if c is current_canvas:
                    print("* {0}".format(c.GetName()))
                else:
                    print("  {0}".format(c.GetName()))
            return

        for c in canvases:
            if c.GetName() == name:
                c.cd()
                print("switching to previous canvas '{0}'".format(name))
                return

        print("switching to new canvas '{0}'".format(name))
        canvas = Canvas(name=name, title=name)
        canvas.cd()

    def help_canvas(self):
        print("switch to a new or previous canvas")

    def complete_canvas(self, text, line, begidx, endidx):
        names = []
        if begidx != endidx:
            prefix = line[begidx: endidx]
        else:
            prefix = ''
        for c in self.canvases:
            name = c.GetName()
            if prefix and not name.startswith(prefix):
                continue
            names.append(name)
        return names

    def do_clear(self, *args):
        canvas = self.current_canvas
        if canvas is not None:
            canvas.Clear()
            canvas.Update()

    def help_clear(self):
        print("clear the current canvas")

    @property
    def styles(self):
        return ROOT.gROOT.GetListOfStyles()

    @property
    def current_style(self):
        return ROOT.gStyle

    def do_style(self, name):

        current_style = self.current_style
        styles = self.styles
        if not name:
            # print list of existing styles
            for s in styles:
                if s.GetName() == current_style.GetName():
                    print("* {0}".format(s.GetName()))
                else:
                    print("  {0}".format(s.GetName()))
            return
        try:
            set_style(name)
        except ValueError as e:
            show_exception(e)
        else:
            canvas = self.current_canvas
            if canvas is not None:
                canvas.UseCurrentStyle()
                canvas.Modified()
                canvas.Update()
                canvas.Modified()
                canvas.Update()

    def complete_style(self, text, line, begidx, endidx):
        names = []
        if begidx != endidx:
            prefix = line[begidx: endidx]
        else:
            prefix = ''
        if not prefix:
            return names
        for s in self.styles:
            name = s.GetName()
            if name.startswith(prefix) or name.lower().startswith(prefix):
                names.append(name)
        return names

    def help_style(self):
        print("set the current style")

    def do_roosh(self, filename=None):
        if not filename:
            for name, rfile in self.files.items():
                if rfile is self.current_file:
                    print("* {0}".format(name))
                else:
                    print("  {0}".format(name))
            return
        if not os.path.isfile(filename):
            print("file '{0}' does not exist".format(filename))
            return
        prev_pwd = self.pwd
        if filename not in self.files:
            print("switching to new file {0}".format(filename))
            rfile = root_open(filename)
            self.files[filename] = rfile
        else:
            print("switching to previous file {0}".format(filename))
            rfile = self.files[filename]
        self.pwd = rfile
        self.current_file = rfile
        self.pwd.cd()
        self.__update_prompt()
        self.prev_pwd = prev_pwd

    def help_roosh(self):
        print("switch to a new or previous file")

    def complete_roosh(self, text, line, begidx, endidx):
        names = []
        if begidx != endidx:
            prefix = line[begidx: endidx]
        else:
            prefix = ''
        for name in glob(prefix + '*'):
            if os.path.isdir(name) or fnmatch(name, '*.root*'):
                names.append(name)
        if len(names) == 1 and os.path.isdir(names[0]):
            names[0] = os.path.normpath(names[0]) + os.path.sep
        return names

    def completenames(self, text, *ignored):
        dotext = 'do_' + text
        cmds = [a[3:] for a in self.get_names() if a.startswith(dotext)]
        objects = [
            key.name for key in self.pwd.keys() if key.name.startswith(text)]
        return cmds + objects

    def completedefault(self, text, line, begidx, endidx):
        return self.completion_helper(text, line, begidx, endidx)

    def default(self, line):
        if line.lstrip().startswith('#'):
            return
        try:
            if not re.match(ASSIGN_CMD, line):
                line = line.strip()
                if (not line.startswith('print') and
                        not line.startswith('from ') and
                        not line.startswith('import ') and
                        not line.startswith('with ') and
                        not line.startswith('if ')):
                    line = '__ = ' + line
            exec(line, self.namespace)
            if '__' in self.namespace:
                if self.namespace['__'] is not None:
                    print(repr(self.namespace['__']))
                del self.namespace['__']
            return
        except Exception as e:
            show_exception(e, debug=self.debug, show_type=True)
            return
        return super(ROOSH, self).default(line)


def main():
    parser = ArgumentParser()
    parser.add_argument('--version', action='version',
                        version=__version__,
                        help="show the version number and exit")
    parser.add_argument('script', nargs='?', default=None,
                        help="read input from this file instead of stdin")
    parser.add_argument('-l', action='store_true',
                        dest='nointro', default=False,
                        help="don't print the intro message")
    parser.add_argument('-u', '--update', action='store_true', default=False,
                        help="open the file in UPDATE mode (default: READ)")
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help="print stack traces")
    parser.add_argument('filename', help="a ROOT file")
    parser.add_argument('libs', nargs='*',
                        help="libraries required to read "
                            "contents of the ROOT file")
    args = parser.parse_args()

    if not os.path.isfile(args.filename) and not args.update:
        sys.exit("File {0} does not exist".format(args.filename))

    if args.libs:
        for lib in args.libs:
            log.info("loading {0}".format(lib))
            ROOT.gSystem.Load(lib)

    history_file = os.path.join(DATA_ROOT, 'roosh_history')
    if os.path.exists(history_file):
        readline.read_history_file(history_file)
    history_size = os.getenv('ROOSH_HISTORY_SIZE', 500)
    readline.set_history_length(history_size)

    try:
        if args.script is not None:
            scriptmode = True
            stdin = open(args.script, 'r')
        else:
            scriptmode = False
            stdin = sys.stdin

        terminal = ROOSH(
            args.filename,
            mode='UPDATE' if args.update else 'READ',
            stdin=stdin,
            script=scriptmode,
            debug=args.debug)

        if scriptmode:
            terminal.use_rawinput = False

        if args.nointro or scriptmode:
            terminal.cmdloop()
        else:
            terminal.cmdloop(
                "Welcome to the ROOSH terminal\ntype help for help")
        if not scriptmode:
            readline.write_history_file(history_file)
    except Exception as e:
        if not scriptmode:
            readline.write_history_file(history_file)
        show_exception(e, debug=args.debug)
        sys.exit(e)
