#!/usr/bin/env python
"""
==================================
A complex hierarchy of tree models
==================================

This example demonstrates how to construct complex tree models by combining
multiple simple models.
"""
print __doc__
from rootpy.tree import TreeModel, BoolCol, IntCol
from rootpy.math.physics.vector import LorentzVector, Vector2


class FourMomentum(TreeModel):
    """
    Base model for all four-momentum objects
    """
    fourmomentum = LorentzVector


class MatchedObject(TreeModel):
    """
    Base model for all objects which may be matched
    to other objects
    """
    matched = BoolCol()


class Jet(FourMomentum, MatchedObject):
    """
    A jet is a matchable four-momentum and
    a boolean flag signifying whether ot not it
    has been flagged as a b-jet
    """
    btagged = BoolCol()


class Tau(FourMomentum, MatchedObject):
    """
    A tau is a matchable four-momentum
    with a number of tracks and a charge
    """
    numtrack = IntCol()
    charge = IntCol()


class Event(Jet.prefix('jet1_'), Jet.prefix('jet2_'),
            Tau.prefix('tau1_'), Tau.prefix('tau2_')):
    """
    An event is composed of two jets and two taus
    an event number and some missing transverse energy
    """
    eventnumber = IntCol()
    missingET = Vector2

print Event

print '=' * 30
# you may also generate classes with simple addition (and subtraction)
print(Jet.prefix('jet1_') + Jet.prefix('jet2_') +
      Tau.prefix('tau1_') + Tau.prefix('tau2_'))

print '=' * 30
# create a TreeBuffer from a TreeModel
buffer = Event()
print type(buffer)
print buffer

print '=' * 30
# convert the Event into a compiled C struct
Event_struct = Event.to_struct()

event = Event_struct()

print event
print dir(event)

event.jet2_matched = True
print event.jet2_matched
