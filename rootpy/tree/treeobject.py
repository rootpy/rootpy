from copy import deepcopy


__MIXINS__ = {}


def mix_classes(cls, mixins):
    
    if not isinstance(mixins, tuple):
        mixins = (mixins,)
    
    classes = (cls,) + mixins
    cls_names = [cls.__name__] + [m.__name__ for m in mixins]
    mixed_name = '_'.join(cls_names)
    inheritance = ', '.join(cls_names)
    inits = '%s.__init__(self, *args, **kwargs)\n' % cls.__name__
    inits += '\n'.join(['        %s.__init__(self)' % \
                        m.__name__ for m in mixins])
    cls_def = '''class %(mixed_name)s(%(inheritance)s):
    def __init__(self, *args, **kwargs):
        %(inits)s''' % locals()
    namespace = dict([(c.__name__, c) for c in classes])
    exec cls_def in namespace
    return namespace[mixed_name]


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
    
    def __init__(self, tree, name, prefix, size, mix=None, cache=True):
        
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
        if mix is not None:
            if mix in __MIXINS__:
                self.tree_object_cls = __MIXINS__[mix]
            else:
                self.tree_object_cls = mix_classes(TreeCollectionObject, mix)
                __MIXINS__[mix] = self.tree_object_cls
        
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
    cls_def = \
    '''class OneToOne%(name)s(object):
        
        @property
        def %(name)s(self):
            
            return collection[self.%(index_branch)s]
    ''' % locals()
    exec cls_def
    return eval('OneToOne%(name)s' % locals())


def one_to_many_assoc(name, collection, index_branch):
    
    collection = deepcopy(collection)
    collection.reset()
    cls_def = \
    '''class OneToMany%(name)s(object):
        
        def __init__(self):

            self.%(name)s = deepcopy(collection)
            self.%(name)s.reset()
            self.%(name)s.select_indices(self.%(index_branch)s)
    ''' % locals()
    exec cls_def
    return eval('OneToMany%(name)s' % locals())
