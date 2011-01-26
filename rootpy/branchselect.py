import inspect
import re

class BranchCollection(set):

    def parse(self, thing, treenames):
        """
        Look for all occurrences of mytree.somebranchname in the source of thing
        and append somebranchname to self.
        This will enable a user to automatically enable only branches that are
        used within his code.
        """
        src = inspect.getsource(thing)
        if type(treenames) is not list:
            treenames = [treenames]
        for name in treenames:
            pattern = "%s.%s"% (name, "(?P<branch>\w+)")
            for match in re.finditer(pattern, src):
                self.add(match.group('branch'))
