.. _trees_and_cuts:

==============
Trees and Cuts
==============

.. currentmodule:: rootpy.tree

rootpy provides pythonized subclasses for ROOT's ``TTrees`` and ``TCuts``

Trees
=====


Tree Models
===========


Tree Objects
============


Tree Chains and Queues
======================


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
