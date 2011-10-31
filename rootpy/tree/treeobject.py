
__MIXINS__ = {}


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
                'selection', 'tree_object_cls'
    
    def __init__(self, tree, name, prefix, size, mix=None):
        
        # TODO support tuple of mixins
         
        super(TreeCollection, self).__init__()
        self.tree = tree
        self.name = name
        self.prefix = prefix
        self.size = size
        self.selection = None
        
        self.tree_object_cls = TreeCollectionObject
        if mix is not None:
            if mix in __MIXINS__:
                self.tree_object_cls = __MIXINS__[mix]
            else:
                self.tree_object_cls = mix_treecollectionobject(mix)
                __MIXINS__[mix] = self.tree_object_cls
        
    def reset(self):

        self.selection = None
    
    def select(self, func):
        
        if self.selection is None:
            self.selection = range(len(self))
        self.selection = [i for i, thing in zip(self.selection, self) if func(thing)]
    
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
        return self.tree_object_cls(self.tree, self.name, self.prefix, index)
    
    def __getitem__(self, index):

        if type(index) is slice:
            return [self[i] for i in xrange(*index.indices(len(self)))]
        if index >= len(self):
            raise IndexError(index)
        if self.selection is not None:
            index = self.selection[index]
        return self.tree_object_cls(self.tree, self.name, self.prefix, index)

    def __len__(self):
        
        if self.selection is not None:
            return len(self.selection)
        return getattr(self.tree, self.size)
    
    def __iter__(self):
        
        if self.selection is not None:
            indices = self.selection
        else:
            indices = xrange(len(self))
        for index in indices:
            yield self.tree_object_cls(self.tree,
                                       self.name,
                                       self.prefix,
                                       index)
