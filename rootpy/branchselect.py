import inspect
import re

class BranchCollection(list):

    def parse(thing, treenames):

        src = inspect.getsource(thing)
        if type(treenames) is not list:
            treenames = [treenames]
        for name in treenames:
            pattern = "%s.%s"% (name, "(?P<branch>\w+)")
            for match in re.finditer(pattern, src):
                self.append(match.group('branch'))
