# Adds Text-based access to trees

from .cut import Cut

class TextBranch(str):
    def __lt__(self, other):
        return Cut('({0}<({1}))'.format(self, other))
    def __gt__(self, other):
        return Cut('({0}>({1}))'.format(self, other))
    def __le__(self, other):
        return Cut('({0}<=({1}))'.format(self, other))
    def __ge__(self, other):
        return Cut('({0}>=({1}))'.format(self, other))
    def __eq__(self, other):
        return Cut('({0}==({1}))'.format(self, other))
    def __neq__(self, other):
        return Cut('({0}!=({1}))'.format(self, other))
    def __sub__(self, other):
        return self.__class__('({0}-{1})'.format(self, other))
    def __rsub__(self, other):
        return self.__class__('({1}-{0})'.format(self, other))
    def __add__(self, other):
        return self.__class__('({0}+{1})'.format(self, other))
    def __radd__(self, other):
        return self.__class__('({1}+{0})'.format(self, other))
    def __mul__(self, other):
        return self.__class__('({0}*{1})'.format(self, other))
    def __rmul__(self, other):
        return self.__class__('({1}*{0})'.format(self, other))
    def __div__(self, other):
        return self.__class__('({0}/{1})'.format(self, other))
    def __rdiv__(self, other):
        return self.__class__('({1}/{0})'.format(self, other))
    def __truediv__(self, other):
        return self.__class__('({0}/{1})'.format(self, other))
    def __rtruediv__(self, other):
        return self.__class__('({1}/{0})'.format(self, other))
    def __abs__(self):
        return self.__class__('abs({0})'.format(self))
    def __rshift__(self, tup):
        if len(tup) == 3:
            return '{0}>>({1[0]},{1[1]},{1[2]})'.format(self,tup)
        elif len(tup) == 4:
            return '{0}>>{1[0]}({1[1]},{1[2]},{1[3]})'.format(self,tup)
        else:
            raise RuntimeError("Must shift a len 3 or 4 tuple")


class TextTree(object):
    def __init__(self, tree):
        self._tree = tree
        self._branch_names = [b.GetName() for b in tree.GetListOfBranches()]
        for name in self._branch_names:
            setattr(self, name, TextBranch(name))
    def __iter__(self):
        return iter(self._branch_names)
