.. _trees_and_cuts:

==============
Trees and Cuts
==============

.. currentmodule:: rootpy.tree

.. warning:: The following documentation is still under construction.

rootpy provides pythonized subclasses for ROOT's ``TTrees`` and ``TCuts``

Trees
=====

ROOT's TTree is subclassed in rootpy and additional API is introduced to ease
their creation in Python:

.. sourcecode:: python

   from rootpy.tree import Tree
   from rootpy.io import root_open
   from random import gauss

   f = root_open("test.root", "recreate")

   tree = Tree("test")
   tree.create_branches(
       {'x': 'F',
        'y': 'F',
        'z': 'F',
        'i': 'I'})

   for i in xrange(10000):
       tree.x = gauss(.5, 1.)
       tree.y = gauss(.3, 2.)
       tree.z = gauss(13., 42.)
       tree.i = i
       tree.fill()
   tree.write()

   f.close()


TTree's Draw method is overridden to support returning and styling the created
histogram:

.. sourcecode:: python

   from rootpy.io import root_open

   myfile = root_open('some_file.root')
   mytree = myfile.treename
   hist = mytree.Draw('x_expression:y_expression',
                      selection='10 < a < 20',
                      linecolor='red',
                      fillstyle='/')


Tree Models
===========

A more powerful way to create trees is by defining tree models.
Easily create complex trees by simple class inheritance (inspired by PyTables):

.. testcode::

   from rootpy.tree import Tree, TreeModel, FloatCol, IntCol

   class FourVect(TreeModel):
       eta = FloatCol(default=-1111.)
       phi = FloatCol(default=-1111.)
       pt = FloatCol()
       m = FloatCol()

   class Tau(FourVect):
       numtrack = IntCol()

   class Event(Tau.prefix('tau1_'),
               Tau.prefix('tau2_')):
       event_number = IntCol()
       run_number = IntCol()

   # tree = Tree('data', model=Event)
   print Event

Branches are constructed according to the requested model:

.. testoutput::
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

   event_number -> IntCol()
   run_number -> IntCol()
   tau1_eta -> FloatCol(default=-1111.0)
   tau1_m -> FloatCol()
   tau1_numtrack -> IntCol()
   tau1_phi -> FloatCol(default=-1111.0)
   tau1_pt -> FloatCol()
   tau2_eta -> FloatCol(default=-1111.0)
   tau2_m -> FloatCol()
   tau2_numtrack -> IntCol()
   tau2_phi -> FloatCol(default=-1111.0)
   tau2_pt -> FloatCol()

Support for default values, automatic STL dictionaries, and ROOT objects is
included.

Tree Objects
============

Documentation coming soon.


Tree Chains and Queues
======================

Documentation coming soon.


Cuts
====

The rootpy :class:`rootpy.tree.Cut` class inherits from ``ROOT.TCut`` and
implements logical operators so cuts can be easily combined:

.. testcode::

   from rootpy.tree import Cut

   cut1 = Cut('a < 10')
   cut2 = Cut('b % 2 == 0')

   cut = cut1 & cut2
   print cut

   # expansion of ternary conditions
   cut3 = Cut('10 < a < 20')
   print cut3

   # easily combine cuts arbitrarily
   cut = ((cut1 & cut2) | - cut3)
   print cut

the output of which is:

.. testoutput::
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

   (a<10)&&(b%2==0)
   (10<a)&&(a<20)
   ((a<10)&&(b%2==0))||(!((10<a)&&(a<20)))


Categories
==========

rootpy introduces a new mechanism with :class:`rootpy.tree.Categories`
to ease the creation of cuts that describe non-overlapping categories.

.. sourcecode:: python

   >>> from rootpy.tree.categories import Categories
   >>> categories = Categories.from_string('{a|1,2,3}x{b|4,5,6}')
   >>> for cut in categories:
   ...     print cut
   ...
   (((a<=2)&&(a<=1))&&(b<=5))&&(b<=4)
   (((a<=2)&&(a<=1))&&(b<=5))&&(b>4)
   (((a<=2)&&(a<=1))&&(b>5))&&(b<=6)
   (((a<=2)&&(a<=1))&&(b>5))&&(b>6)
   (((a<=2)&&(a>1))&&(b<=5))&&(b<=4)
   (((a<=2)&&(a>1))&&(b<=5))&&(b>4)
   (((a<=2)&&(a>1))&&(b>5))&&(b<=6)
   (((a<=2)&&(a>1))&&(b>5))&&(b>6)
   (((a>2)&&(a<=3))&&(b<=5))&&(b<=4)
   (((a>2)&&(a<=3))&&(b<=5))&&(b>4)
   (((a>2)&&(a<=3))&&(b>5))&&(b<=6)
   (((a>2)&&(a<=3))&&(b>5))&&(b>6)
   (((a>2)&&(a>3))&&(b<=5))&&(b<=4)
   (((a>2)&&(a>3))&&(b<=5))&&(b>4)
   (((a>2)&&(a>3))&&(b>5))&&(b<=6)
   (((a>2)&&(a>3))&&(b>5))&&(b>6)
