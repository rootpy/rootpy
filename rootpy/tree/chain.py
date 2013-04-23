# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
import multiprocessing
import time
from ..io import root_open, DoesNotExist
from .filtering import EventFilterList
from ..util.extras import humanize_bytes
from .. import log; log = log[__name__]
from ..context import preserve_current_directory


class BaseTreeChain(object):

    def __init__(self, name,
                 treebuffer=None,
                 branches=None,
                 ignore_branches=None,
                 events=-1,
                 onfilechange=None,
                 read_branches_on_demand=False,
                 cache=False,
                 cache_size=30000000, # 30MB
                 learn_entries=10,
                 always_read=None,
                 ignore_unsupported=False,
                 filters=None):

        self._name = name
        self._buffer = treebuffer
        self._branches = branches
        self._ignore_branches = ignore_branches
        self._tree = None
        self._file = None
        self._events = events
        self._total_events = 0
        self._ignore_unsupported = ignore_unsupported
        self._initialized = False
        if filters is None:
            self._filters = EventFilterList([])
        else:
            self._filters = filters
        if onfilechange is None:
            onfilechange = []
        self._filechange_hooks = onfilechange

        self._read_branches_on_demand = read_branches_on_demand
        self._use_cache = cache
        self._cache_size = cache_size
        self._learn_entries = learn_entries

        self.weight = 1.
        self.userdata = {}

        if not self._rollover():
            raise RuntimeError("unable to initialize TreeChain")

        if always_read is None:
            self._always_read = []
        elif isinstance(always_read, basestring):
            if '*' in always_read:
                always_read = self._tree.glob(always_read)
            else:
                always_read = [always_read]
            self.always_read(always_read)
        else:
            branches = []
            for branch in always_read:
                if '*' in branch:
                    branches += self._tree.glob(branch)
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
        self._tree.always_read(branches)

    def reset(self):

        if self._tree is not None:
            self._tree = None
        if self._file is not None:
            self._file.Close()
            self._file = None

    def Draw(self, *args, **kwargs):
        '''
        Loop over subfiles, draw each, and sum the output into a single
        histogram.
        '''
        self.reset()
        output = None
        while self._rollover():
            if not output:
                # Make our own copy of the drawn histogram
                output = self._tree.Draw(*args, **kwargs)
                if output:
                    # Make it memory resident
                    output = output.Clone()
                    output.SetDirectory(0)
            else:
                newoutput = self._tree.Draw(*args, **kwargs)
                if newoutput:
                    output += newoutput
        return output

    def draw(self, *args, **kwargs):

        return self.Draw(*args, **kwargs)

    def __getattr__(self, attr):

        try:
            return getattr(self._tree, attr)
        except AttributeError:
            raise AttributeError("%s instance has no attribute '%s'" % \
                (self.__class__.__name__, attr))

    def __getitem__(self, item):

        return self._tree.__getitem__(item)

    def __contains__(self, branch):

        return self._tree.__contains__(branch)

    def __iter__(self):

        passed_events = 0
        while True:
            entries = 0
            total_entries = float(self._tree.GetEntries())
            t1 = time.time()
            t2 = t1
            for entry in self._tree:
                entries += 1
                self.userdata = {}
                if self._filters(entry):
                    yield entry
                    passed_events += 1
                    if self._events == passed_events:
                        break
                if time.time() - t2 > 60:
                    log.info(
                        "%i entries per second. %.0f%% done current tree." %
                        (int(entries / (time.time() - t1)),
                        100 * entries / total_entries))
                    t2 = time.time()
            if self._events == passed_events:
                break
            log.info("%i entries per second" %
                int(entries / (time.time() - t1)))
            log.info("read %i bytes in %i transactions" %
                (self._file.GetBytesRead(), self._file.GetReadCalls()))
            self._total_events += entries
            if not self._rollover():
                break
        self._filters.finalize()

    def _rollover(self):

        BaseTreeChain.reset(self)
        filename = self._next_file()
        if filename is None:
            return False
        try:
            with preserve_current_directory():
                self._file = root_open(filename)
        except IOError:
            self._file = None
            log.warning("could not open file %s (skipping)" % filename)
            return self._rollover()
        try:
            self._tree = self._file.Get(self._name)
        except DoesNotExist:
            log.warning("tree %s does not exist in file %s (skipping)" %
                (self._name, filename))
            return self._rollover()
        if len(self._tree.GetListOfBranches()) == 0:
            log.warning("tree with no branches in file %s (skipping)" %
                filename)
            return self._rollover()
        if self._branches is not None:
            self._tree.activate(self._branches, exclusive=True)
        if self._ignore_branches is not None:
            self._tree.deactivate(self._ignore_branches, exclusive=False)
        if self._buffer is None:
            self._tree.create_buffer(self._ignore_unsupported)
            self._buffer = self._tree._buffer
        else:
            self._tree.set_buffer(
                    self._buffer,
                    ignore_missing=True,
                    transfer_objects=True)
            self._buffer = self._tree._buffer
        if self._use_cache:
            # enable TTreeCache for this tree
            log.info(("enabling a %s TTreeCache for the current tree "
                      "(%d learning entries)") %
                    (humanize_bytes(self._cache_size), self._learn_entries))
            self._tree.SetCacheSize(self._cache_size)
            self._tree.SetCacheLearnEntries(self._learn_entries)
        self._tree.read_branches_on_demand(self._read_branches_on_demand)
        self._tree.always_read(self._always_read)
        self.weight = self._tree.GetWeight()
        for target, args in self._filechange_hooks:
            # run any user-defined functions
            target(*args, name=self._name, file=self._file, tree=self._tree)
        return True


class TreeChain(BaseTreeChain):
    """
    A ROOT.TChain replacement
    """
    def __init__(self, name, files, **kwargs):

        if isinstance(files, tuple):
            files = list(files)
        elif not isinstance(files, list):
            files = [files]
        else:
            files = files[:]

        if not files:
            raise RuntimeError(
                "unable to initialize TreeChain: no files")
        self._files = files
        self.curr_file_idx = 0

        super(TreeChain, self).__init__(name, **kwargs)

    def reset(self):
        """
        Reset the chain to the first file
        Note: not valid when in queue mode
        """
        super(TreeChain, self).reset()
        self.curr_file_idx = 0

    def __len__(self):

        return len(self._files)

    def _next_file(self):

        if self.curr_file_idx >= len(self._files):
            return None
        filename = self._files[self.curr_file_idx]
        log.info("%i file(s) remaining" %
            (len(self._files) - self.curr_file_idx))
        self.curr_file_idx += 1
        return filename


class TreeQueue(BaseTreeChain):

    SENTINEL = None

    def __init__(self, name, files, **kwargs):

        # For some reason, multiprocessing.queues d.n.e. until
        # one has been created (Mac OS)
        multiprocessing.Queue()
        if not isinstance(files, multiprocessing.queues.Queue):
            raise TypeError("files must be a multiprocessing.Queue")
        self._files = files

        super(TreeQueue, self).__init__(name, **kwargs)

    def __len__(self):

        # not reliable
        return self._files.qsize()

    def __nonzero__(self):

        # not reliable
        return not self._files.empty()

    def _next_file(self):

        filename = self._files.get()
        if filename == self.SENTINEL:
            return None
        return filename
