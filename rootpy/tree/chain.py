import sys
import multiprocessing
import time
from ..io import open as ropen, DoesNotExist
from .filtering import EventFilterList
from .. import rootpy_globals


class _BaseTreeChain(object):

    def __init__(self, name,
                 buffer=None,
                 branches=None,
                 events=-1,
                 stream=None,
                 onfilechange=None,
                 cache=False,
                 cache_size=10000000,
                 learn_entries=1,
                 always_read=None,
                 ignore_unsupported=False,
                 filters=None,
                 verbose=False):

        self.name = name
        self.buffer = buffer
        self.branches = branches
        self.weight = 1.
        self.tree = None
        self.file = None
        if filters is None:
            self.filters = EventFilterList([])
        else:
            self.filters = filters
        self.userdata = {}
        self.events = events
        self.total_events = 0
        self.ignore_unsupported = ignore_unsupported
        self.initialized = False

        if stream is None:
            self.stream = sys.stdout
        else:
            self.stream = stream

        if onfilechange is None:
            onfilechange = []
        self.filechange_hooks = onfilechange

        self.usecache = cache
        self.cache_size = cache_size
        self.learn_entries = learn_entries

        self.verbose = verbose

        if not self._rollover():
            raise RuntimeError("unable to initialize TreeChain")

        if always_read is None:
            self._always_read = []
        elif isinstance(always_read, basestring):
            if '*' in always_read:
                always_read = self.tree.glob(always_read)
            else:
                always_read = [always_read]
            self.always_read(always_read)
        else:
            branches = []
            for branch in always_read:
                if '*' in branch:
                    branches += self.tree.glob(branch)
                else:
                    branches.append(branch)
            self.always_read(branches)

    def __nonzero__(self):

        return len(self) > 0

    def _next_file(self):
        """
        Override in subclasses
        """
        return None

    def always_read(self, branches):

        self._always_read = branches
        self.tree.always_read(branches)

    def reset(self):

        if self.tree is not None:
            self.tree = None
        if self.file is not None:
            self.file.Close()
            self.file = None

    def Draw(self, *args, **kwargs):
        '''
        Loop over subfiles, draw each, and sum the output into a single
        histogram.
        '''
        self.reset()
        output = None
        while self._rollover():
            print output
            if not output:
                # Make our own copy of the drawn histogram
                output = self.tree.Draw(*args, **kwargs)
                if output:
                    # Make it memory resident
                    output = output.Clone()
                    output.SetDirectory(0)
            else:
                newoutput = self.tree.Draw(*args, **kwargs)
                if newoutput:
                    output += newoutput
        return output

    def draw(self, *args, **kwargs):

        return self.Draw(*args, **kwargs)

    def __getattr__(self, attr):

        try:
            return getattr(self.tree, attr)
        except AttributeError:
            raise AttributeError("%s instance has no attribute '%s'" % \
                (self.__class__.__name__, attr))

    def __getitem__(self, item):

        return self.tree.__getitem__(item)

    def __contains__(self, branch):

        return self.tree.__contains__(branch)

    def __iter__(self):

        passed_events = 0
        while True:
            entries = 0
            total_entries = float(self.tree.GetEntries())
            if self.verbose:
                t1 = time.time()
                t2 = t1
            for entry in self.tree:
                entries += 1
                self.userdata = {}
                if self.filters(entry):
                    yield entry
                    passed_events += 1
                    if self.events == passed_events:
                        break
                if self.verbose and time.time() - t2 > 60:
                    print >> self.stream, \
                        "%i entries per second. %.0f%% done current tree." % \
                        (int(entries / (time.time() - t1)),
                        100 * entries / total_entries)
                    t2 = time.time()
            if self.events == passed_events:
                break
            if self.verbose:
                print >> self.stream, "%i entries per second" % \
                    int(entries / (time.time() - t1))
                print "Read %i bytes in %i transactions" % \
                    (self.file.GetBytesRead(), self.file.GetReadCalls())
            self.total_events += entries
            if not self._rollover():
                break
        self.filters.finalize()

    def _rollover(self):

        _BaseTreeChain.reset(self)
        filename = self._next_file()
        if filename is None:
            return False
        pwd = rootpy_globals.directory
        try:
            self.file = ropen(filename)
        except IOError:
            self.file = None
            pwd.cd()
            rootpy_globals.directory = pwd
            print >> self.stream, "WARNING: Skipping file. " \
                                  "Could not open file %s" % filename
            return self._rollover()
        pwd.cd()
        rootpy_globals.directory = pwd
        try:
            self.tree = self.file.Get(
                self.name,
                ignore_unsupported=self.ignore_unsupported)
        except DoesNotExist:
            print >> self.stream, "WARNING: Skipping file. " \
                                  "Tree %s does not exist in file %s" % \
                                  (self.name, filename)
            return self._rollover()
        if len(self.tree.GetListOfBranches()) == 0:
            print >> self.stream, "WARNING: skipping tree with " \
                                  "no branches in file %s" % filename
            return self._rollover()
        if self.branches is not None:
            self.tree.activate(self.branches, exclusive=True)
        if self.buffer is None:
            self.buffer = self.tree.buffer
        else:
            self.tree.set_buffer(self.buffer,
                                 ignore_missing=True,
                                 transfer_objects=True)
            self.buffer = self.tree.buffer
        self.tree.use_cache(self.usecache,
                            cache_size=self.cache_size,
                            learn_entries=self.learn_entries)
        self.tree.always_read(self._always_read)
        self.weight = self.tree.GetWeight()
        for target, args in self.filechange_hooks:
            target(*args, name=self.name, file=self.file, tree=self.tree)
        return True


class TreeChain(_BaseTreeChain):
    """
    A ROOT.TChain replacement
    """
    def __init__(self, name, files,
                 buffer=None,
                 branches=None,
                 events=-1,
                 stream=None,
                 onfilechange=None,
                 cache=False,
                 cache_size=10000000,
                 learn_entries=1,
                 always_read=None,
                 ignore_unsupported=False,
                 filters=None,
                 verbose=False):

        if isinstance(files, tuple):
            files = list(files)
        elif not isinstance(files, (list, tuple)):
            files = [files]
        else:
            files = files[:]

        if not files:
            raise RuntimeError(
                "unable to initialize TreeChain: no files")
        self.files = files
        self.curr_file_idx = 0

        super(TreeChain, self).__init__(
                name,
                buffer,
                branches,
                events,
                stream,
                onfilechange,
                cache,
                cache_size,
                learn_entries,
                always_read,
                ignore_unsupported,
                filters,
                verbose)

    def reset(self):
        """
        Reset the chain to the first file
        Note: not valid when in queue mode
        """
        super(TreeChain, self).reset()
        self.curr_file_idx = 0

    def __len__(self):

        return len(self.files)

    def _next_file(self):

        if self.curr_file_idx >= len(self.files):
            return None
        filename = self.files[self.curr_file_idx]
        if self.verbose:
            print >> self.stream, "%i file(s) remaining..." % \
                (len(self.files) - self.curr_file_idx)
        self.curr_file_idx += 1
        return filename


class TreeQueue(_BaseTreeChain):

    SENTINEL = None

    def __init__(self, name, files,
                 buffer=None,
                 branches=None,
                 events=-1,
                 stream=None,
                 onfilechange=None,
                 cache=False,
                 cache_size=10000000,
                 learn_entries=1,
                 always_read=None,
                 ignore_unsupported=False,
                 filters=None,
                 verbose=False):

        # For some reason, multiprocessing.queues d.n.e. until
        # one has been created (Mac OS)
        multiprocessing.Queue()
        if not isinstance(files, multiprocessing.queues.Queue):
            raise TypeError("files must be a multiprocessing.Queue")
        self.files = files

        super(TreeQueue, self).__init__(
                name,
                buffer,
                branches,
                events,
                stream,
                onfilechange,
                cache,
                cache_size,
                learn_entries,
                always_read,
                ignore_unsupported,
                filters,
                verbose)

    def __len__(self):

        # not reliable
        return self.files.qsize()

    def __nonzero__(self):

        # not reliable
        return not self.files.empty()

    def _next_file(self):

        filename = self.files.get()
        if filename == self.SENTINEL:
            return None
        return filename
