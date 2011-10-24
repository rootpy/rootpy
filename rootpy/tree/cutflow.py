

class Cutflow(object):

    def __init__(self, names=None):
    
        if names is not None:
            self.__names = names
        else:
            self.__names = []
        self.__dict = None
        self.reset()

    def __setitem__(self, name, passes):

        if name not in self.__names:
            self.__names.append(name)
        self.__dict[name] = str(int(bool(passes)))
    
    def passed(self, name):
        
        if name not in self.__names:
            self.__names.append(name)
        self.__dict[name] = '1'
    
    def stages(self):
        
        self.reset()
        yield self
        for name in self.__names:
           self.passes(name)
           yield self
        self.reset()

    def reset(self):

        self.__dict = dict((name, '0') for name in self.__names)

    def bitstring(self):

        return ''.join([self.__dict[item] for item in self.__names])

    def int(self):

        if not self.__dict:
            return 0
        return int(self.bitstring(), 2)


from tabulartext import TextTable


class CutflowTable(object):

    def __init__(self, lumi = 1.):

        self.lumi = lumi
        self.samples = []
        self.cut_titles = []

    def add_sample(sample, name, weight = 1.):

        titles = [cut[0] for cut in sample]
        if not self.samples:
            self.cut_titles = titles
        elif titles != self.cut_titles:
            raise ValueError("mismatching cut-flows: names don't match") 
        self.samples.append((weight, name, [cut[1] for cut in sample]))

    def __str__(self):

        return self.__repr__()

    def __repr__(self):

        if not self.samples:
            return ''
        table = TextTable()
        table.set_deco(TextTable.HEADER)
        table.add_row(['']+[name for weight, name, cuts in self.samples])
        for i, title in enumerate(self.cut_titles):
            table.add_row([self.cut_titles[i]] + 
                          [weight * cuts * self.lumi for \
                           weight, name, cuts in self.samples]) 
        return table.draw()
