class FilterList(list):

    def passes(self,*args):

        for filter in self:
            if not filter.passes(*args):
                return False
        return True
