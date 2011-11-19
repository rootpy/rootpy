from copy import deepcopy

__MIXINS__ = {}


def mix(cls, mixins):
    
    if not isinstance(mixins, tuple):
        mixins = (mixins,)
    
    cls_names = [cls.__name__] + [m.__name__ for m in mixins]
    mixed_name = '_'.join(cls_names)
    inheritance = ','.join(cls_names)
    cls_def = \
    '''
    class %s(%s): pass
    ''' % (mixed_name, inheritance)

    exec cls_def
    return eval(mixed_name)

'''
def mix_treeobject(mix):

    class TreeObject_mixin(TreeObject, mix):
        
        def __init__(self, *args, **kwargs):

            TreeObject.__init__(self, *args, **kwargs)
            mix.__init__(self)

    return TreeObject_mixin


def mix_treecollectionobject(mix):

    class TreeCollectionObject_mixin(TreeCollectionObject, mix):
        
        def __init__(self, *args, **kwargs):

            TreeCollectionObject.__init__(self, *args, **kwargs)
            mix.__init__(self)

    return TreeCollectionObject_mixin
'''

class TreeObject(object):
    
    __slots__ = 'tree', 'name', 'prefix'

    def __init__(self, tree, name, prefix):

        self.tree = tree
        self.name = name
        self.prefix = prefix
         
    def __eq__(self, other):

        return self.name == other.name and \
               self.prefix == other.prefix
    
    def __getitem__(self, attr):

        return getattr(self, attr)
         
    def __setitem__(self, attr, value):

        setattr(self.tree, self.prefix + attr, value)
    
    def __getattr__(self, attr):
        
        return getattr(self.tree, self.prefix + attr)


class TreeCollectionObject(TreeObject):
    
    __slots__ = 'index'

    def __init__(self, tree, name, prefix, index):

        self.index = index
        super(TreeCollectionObject, self).__init__(tree, name, prefix)
    
    def __eq__(self, other):

        return self.index == other.index and \
               TreeObject.__eq__(self, other)

    def __getattr__(self, attr):
        
        try: 
            return getattr(self.tree, self.prefix + attr)[self.index]
        except IndexError:
            raise IndexError("index %i out of range for "
                             "attribute %s of collection %s of size %i" % \
                             (self.index, attr, self.prefix,
                             len(getattr(self.tree, self.prefix + attr))))


class TreeCollection(object):

    __slots__ = 'tree', 'name', \
                'prefix', 'size', \
                'selection', 'tree_object_cls', \
                '__cache_objects', '__cache'
    
    def __init__(self, tree, name, prefix, size, mixin=None, cache=True):
        
        # TODO support tuple of mixins
         
        super(TreeCollection, self).__init__()
        self.tree = tree
        self.name = name
        self.prefix = prefix
        self.size = size
        self.selection = None

        self.__cache_objects = cache
        self.__cache = {}
        
        self.tree_object_cls = TreeCollectionObject
        if mixin is not None:
            if mixin in __MIXINS__:
                self.tree_object_cls = __MIXINS__[mixin]
            else:
                self.tree_object_cls = mix(TreeCollectionObject, mixin)
                __MIXINS__[mixin] = self.tree_object_cls
        
    def reset(self):

        self.selection = None
        self.__cache = {}
    
    def select(self, func):
        
        if self.selection is None:
            self.selection = range(len(self))
        self.selection = [i for i, thing in zip(self.selection, self) if func(thing)]
    
    def select_indices(self, indices):

        if self.selection is None:
            self.selection = range(len(self))
        self.selection = [self.seletion[i] for i in indices]

    def mask(self, func):

        if self.selection is None:
            self.selection = range(len(self))
        self.selection = [i for i, thing in zip(self.selection, self) if not func(thing)]

    def mask_indices(self, indices):

        if self.selection is None:
            self.selection = range(len(self))
        self.selection = [j for i, j in enumerate(self.selection) if i not in indices]
         
    def _wrap_sort_key(self, key):
        
        def wrapped_key(index):
            return key(self.getitem(index)) 
        return wrapped_key
    
    def sort(self, key, **kwargs):

        if self.selection is None:
            self.selection = range(len(self))
        self.selection.sort(key=self._wrap_sort_key(key), **kwargs)
    
    def slice(self, start=0, stop=None, step=1):

        if self.selection is None:
            self.selection = range(len(self))
        self.selection = self.selection[slice(start, stop, step)]
    
    def make_persistent(self):
        """
        Perform actual selection and sorting on underlying
        attribute vectors
        """
        pass
    
    def getitem(self, index):

        if index >= getattr(self.tree, self.size):
            raise IndexError(index)
        if self.__cache_objects and index in self.__cache:
            return self.__cache[index]
        obj = self.tree_object_cls(self.tree, self.name, self.prefix, index)
        if self.__cache_objects:
            self.__cache[index] = obj
        return obj
    
    def __getitem__(self, index):

        if type(index) is slice:
            return [self[i] for i in xrange(*index.indices(len(self)))]
        if index >= len(self):
            raise IndexError(index)
        if self.selection is not None:
            index = self.selection[index]
        if self.__cache_objects and index in self.__cache:
            return self.__cache[index]
        obj = self.tree_object_cls(self.tree, self.name, self.prefix, index)
        if self.__cache_objects:
            self.__cache[index] = obj
        return obj

    def __len__(self):
        
        if self.selection is not None:
            return len(self.selection)
        return getattr(self.tree, self.size)
    
    def __iter__(self):
        
        for index in xrange(len(self)):
            yield self.__getitem__(index)


def one_to_one_assoc(name, collection, index_branch):
    
    collection = deepcopy(collection)
    collection.reset()
    cls_name = 'OneToOne%s' % name
    cls_def = \
    '''
    class %s(object):
        
        @property
        def %s(self):
            
            return collection[self.index_branch]

    ''' % (cls_name, name)

    return eval(cls_name)


def one_to_many_assoc(name, collection, index_branch):
    
    cls_name = 'OneToMany%s' % name
    cls_def = \
    '''
    class %s(object):
        
        def __init__(self):

            self.%s = deepcopy(collection)
            self.%s.reset()
            self.%s.select_indices(self.index_branch)

    ''' % (cls_name, name)

    return eval(cls_name)
