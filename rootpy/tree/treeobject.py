# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

from copy import deepcopy

from ..extern.six.moves import range

__all__ = [
    'TreeObject',
    'TreeCollectionObject',
    'TreeCollection',
]

__MIXINS__ = {}


def mix_classes(cls, mixins):
    if not isinstance(mixins, tuple):
        mixins = (mixins,)
    classes = (cls,) + mixins
    cls_names = [cls.__name__] + [m.__name__ for m in mixins]
    mixed_name = '_'.join(cls_names)
    inheritance = ', '.join(cls_names)
    inits = '{cls.__name__}.__init__(self, *args, **kwargs)\n'.format(cls=cls)
    inits += '\n'.join(['        {0}.__init__(self)'.format(m.__name__)
                        for m in mixins])
    cls_def = '''class {mixed_name}({inheritance}):
    def __init__(self, *args, **kwargs):
        {inits}'''.format(
            mixed_name=mixed_name,
            inheritance=inheritance,
            inits=inits)
    namespace = dict([(c.__name__, c) for c in classes])
    exec(cls_def, namespace)
    return namespace[mixed_name]


class TreeObject(object):

    def __init__(self, tree, name, prefix):
        self.tree = tree
        self.name = name
        self.prefix = prefix
        self._inited = True

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.name == other.name and
                self.prefix == other.prefix)

    def __hash__(self):
        return hash((
            self.__class__.__name__,
            self.name,
            self.prefix))

    def __getitem__(self, attr):
        return getattr(self, attr)

    def __setitem__(self, attr, value):
        setattr(self.tree, self.prefix + attr, value)

    def __getattr__(self, attr):
        return getattr(self.tree, self.prefix + attr)

    def __setattr__(self, attr, value):
        if '_inited' not in self.__dict__:
            return object.__setattr__(self, attr, value)
        try:
            setattr(self.tree, self.prefix + attr, value)
        except AttributeError:
            return object.__setattr__(self, attr, value)

    def define_object(self, name, prefix):
        obj = TreeObject(self, name, prefix)
        object.__setattr__(self, name, obj)
        return obj


class TreeCollectionObject(TreeObject):

    def __init__(self, tree, name, prefix, index):
        self.index = index
        super(TreeCollectionObject, self).__init__(tree, name, prefix)
        self._inited = True

    def __eq__(self, other):
        return TreeObject.__eq__(self, other) and self.index == other.index

    def __hash__(self):
        return hash((
            self.__class__.__name__,
            self.name,
            self.prefix,
            self.index))

    def __getattr__(self, attr):
        try:
            return getattr(self.tree, self.prefix + attr)[self.index]
        except IndexError:
            raise IndexError(
                "index {0:d} out of range for "
                "attribute `{1}` of collection `{2}` of size {3:d}".format(
                    self.index, attr, self.prefix,
                    len(getattr(self.tree, self.prefix + attr))))

    def __setattr__(self, attr, value):
        if '_inited' not in self.__dict__:
            return object.__setattr__(self, attr, value)
        try:
            getattr(self.tree, self.prefix + attr)[self.index] = value
        except IndexError:
            raise IndexError(
                "index {0:d} out of range for "
                "attribute `{1}` of collection `{2}` of size {3:d}".format(
                    self.index, attr, self.prefix,
                    len(getattr(self.tree, self.prefix + attr))))
        except AttributeError:
            return object.__setattr__(self, attr, value)


class TreeCollection(object):

    def __init__(self, tree, name, prefix, size, mix=None, cache=True):
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

    def __nonzero__(self):
        return len(self) > 0

    __bool__ = __nonzero__

    def reset(self):
        self.reset_selection()
        self.reset_cache()

    def reset_selection(self):
        self.selection = None

    def reset_cache(self):
        self.__cache = {}

    def remove(self, thing):
        if self.selection is None:
            self.selection = range(len(self))
        for i, other in enumerate(self):
            if thing == other:
                self.selection.pop(i)
                break

    def pop(self, index):
        if self.selection is None:
            self.selection = range(len(self))
        thing = self[index]
        self.selection.pop(index)
        return thing

    def select(self, func):
        if self.selection is None:
            self.selection = range(len(self))
        self.selection = [
            i for i, thing in zip(self.selection, self)
            if func(thing)]

    def select_indices(self, indices):
        if self.selection is None:
            self.selection = range(len(self))
        self.selection = [self.selection[i] for i in indices]

    def mask(self, func):
        if self.selection is None:
            self.selection = range(len(self))
        self.selection = [
            i for i, thing in zip(self.selection, self)
            if not func(thing)]

    def mask_indices(self, indices):
        if self.selection is None:
            self.selection = range(len(self))
        self.selection = [
            j for i, j in enumerate(self.selection)
            if i not in indices]

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
        """
        direct access without going through self.selection
        """
        if index >= getattr(self.tree, self.size):
            raise IndexError(index)
        if self.__cache_objects and index in self.__cache:
            return self.__cache[index]
        obj = self.tree_object_cls(self.tree, self.name, self.prefix, index)
        if self.__cache_objects:
            self.__cache[index] = obj
        return obj

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
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

    def len(self):
        """
        length of original collection
        """
        return getattr(self.tree, self.size)

    def __len__(self):
        if self.selection is not None:
            return len(self.selection)
        return getattr(self.tree, self.size)

    def __iter__(self):
        for index in range(len(self)):
            yield self.__getitem__(index)


def one_to_one_assoc(name, collection, index_branch):
    collection = deepcopy(collection)
    collection.reset()
    cls_def = \
    '''class OneToOne{name}(object):
    @property
    def {name}(self):
        return collection[self.{index_branch}]
    '''.format(name=name, index_branch=index_branch)
    namespace = {}
    eval(cls_def, namespace)
    return namespace['OneToOne{name}'.format(name=name)]


def one_to_many_assoc(name, collection, index_branch):
    collection = deepcopy(collection)
    collection.reset()
    cls_def = \
    '''class OneToMany{name}(object):
    def __init__(self):
        self.{name} = deepcopy(collection)
        self.{name}.reset()
        self.{name}.select_indices(self.{index_branch})
    '''.format(name=name, index_branch=index_branch)
    namespace = {}
    eval(cls_def, namespace)
    return namespace['OneToMany{name}'.format(name=name)]
