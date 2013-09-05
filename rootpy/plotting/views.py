# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
'''

=====================
Folder "View" Classes
=====================

These classes wrap Directories and perform automatic actions
to Histograms retrieved from them.  The different views can be composited and
layered.

Summary of views:

- ScaleView: scale histogram normalization
- NormalizeView: normalize histograms
- SumView: sum histograms from different folders together
- StyleView: apply a style to histograms
- StackView: build THStacks using histograms from different folders
- TitleView: change the title of histograms
- FunctorView: apply a arbitrary transformation function to the histograms
- MultiFunctorView: apply a arbitrary transformation function to a collection
  of histograms
- SubdirectoryView: A view of a subdirectory, which maintains the same view as
  the base.

Example use case
================

One has a ROOT file with the following content::

    zjets/mutau_mass
    zz/mutau_mass
    wz/mutau_mass
    data_2010/mutau_mass
    data_2011/mutau_mass

and wants to do the following:

1. Merge the two data taking periods together
2. Scale the Z, WZ, and ZZ simulated results to the appropriate int. lumi.
3. Combine WZ and ZZ into a single diboson sample
4. Apply different colors to the MC samples
5. Make a Stack of the expected yields from different simulated processes

This example can be tested by running::

    python -m rootpy.plotting.views

>>> # Mock up the example test case
>>> import rootpy.io as io
>>> # We have to keep these, to make sure PyROOT doesn't garbage collect them
>>> keep = []
>>> zjets_dir = io.Directory('zjets', 'Zjets directory')
>>> zz_dir = io.Directory('zz', 'ZZ directory')
>>> wz_dir = io.Directory('wz', 'WZ directory')
>>> data2010_dir = io.Directory('data2010', 'data2010 directory')
>>> data2011_dir = io.Directory('data2011', 'data2011 directory')
>>> # Make the Zjets case
>>> _ = zjets_dir.cd()
>>> zjets_hist = ROOT.TH1F("mutau_mass", "Mu-Tau mass", 100, 0, 100)
>>> zjets_hist.FillRandom('gaus', 5000)
>>> keep.append(zjets_hist)
>>> # Make the ZZ case
>>> _ = zz_dir.cd()
>>> zz_hist = ROOT.TH1F("mutau_mass", "Mu-Tau mass", 100, 0, 100)
>>> zz_hist.FillRandom('gaus', 5000)
>>> keep.append(zz_hist)
>>> # Make the WZ case
>>> _ = wz_dir.cd()
>>> wz_hist = ROOT.TH1F("mutau_mass", "Mu-Tau mass", 100, 0, 100)
>>> wz_hist.FillRandom('gaus', 5000)
>>> keep.append(wz_hist)
>>> # Make the 2010 data case
>>> _ = data2010_dir.cd()
>>> data2010_hist = ROOT.TH1F("mutau_mass", "Mu-Tau mass", 100, 0, 100)
>>> data2010_hist.FillRandom('gaus', 30)
>>> keep.append(data2010_hist)
>>> # Make the 2011 data case
>>> _ = data2011_dir.cd()
>>> data2011_hist = ROOT.TH1F("mutau_mass", "Mu-Tau mass", 100, 0, 100)
>>> data2011_hist.FillRandom('gaus', 51)
>>> keep.append(data2011_hist)

SumView
-------

We can merge the two data periods into a single case using a SumView.

>>> data = SumView(data2010_dir, data2011_dir)
>>> data_hist = data.Get("mutau_mass")
>>> data_hist.Integral()
81.0
>>> data_hist.Integral() == data2010_hist.Integral() + data2011_hist.Integral()
True

ScaleView
---------

The simulated results (Z & diboson) can be scaled to the expected integrated
luminosity using ScaleViews.

>>> zjets = ScaleView(zjets_dir, 0.01)
>>> zjets_hist = zjets.Get("mutau_mass")
>>> abs(zjets_hist.Integral() - 50.0) < 1e-5
True
>>> # Scale the diboson contribution
>>> zz = ScaleView(zz_dir, 0.001)
>>> wz = ScaleView(wz_dir, 0.003)

Combining views
---------------

The dibosons individually are tiny, let's put them together using a SumView.
Note that this operation nests two ScaleViews into a SumView.

>>> dibosons = SumView(zz, wz)
>>> # We expect 5000*0.001 + 5000*0.003 = 20 events
>>> dibosons_hist = dibosons.Get("mutau_mass")
>>> abs(dibosons_hist.Integral() - 20) < 1e-4
True

StyleView
---------

A style view automatically applies a style to retrieved Plottable objects.
The style is specified using the same arguments as the Plottable.decorate.
Let's make the Z background red and the diboson background blue.

>>> zjets = StyleView(zjets, fillcolor=ROOT.EColor.kRed)
>>> dibosons = StyleView(dibosons, fillcolor=ROOT.EColor.kBlue)
>>> zjets_hist = zjets.Get("mutau_mass")
>>> zjets_hist.GetFillColor() == ROOT.EColor.kRed
True
>>> dibosons_hist = dibosons.Get("mutau_mass")
>>> dibosons_hist.GetFillColor() == ROOT.EColor.kBlue
True

StackView
---------

The StackView combines multiple items into a HistStack.  In our example
we stack the SM backgrounds to compare to the data.

>>> sm_bkg = StackView(zjets, dibosons)
>>> sm_bkg_stack = sm_bkg.Get("mutau_mass")
>>> '%0.0f' % sm_bkg_stack.Integral()
'70'

Looks like we have an excess of 11 events - must be the Higgs.


Other Examples
==============

NormalizeView
-------------

The normalization view renormalizes histograms to a given value (default 1.0).
Here is an example of using the NormalizeView to compare the Z and diboson
shapes.

>>> z_shape = NormalizeView(zjets)
>>> z_shape_hist = z_shape.Get("mutau_mass")
>>> abs(1 - z_shape_hist.Integral()) < 1e-5
True
>>> # Let's compare the shapes using a HistStack, using the "nostack" option.
>>> diboson_shape = NormalizeView(dibosons)
>>> shape_comparison = StackView(z_shape, diboson_shape)
>>> # To draw the comparison:
>>> # shape_comparison.Get("mutau_mass").Draw('nostack')

FunctorView
-----------

FunctorView allows you to apply an arbitrary transformation to the object.
Here we show how you can change the axis range for all histograms in a
directory.

>>> rebin = lambda x: x.Rebin(2)
>>> zjets_rebinned = FunctorView(zjets, rebin)
>>> zjets.Get("mutau_mass").GetNbinsX()
100
>>> zjets_rebinned.Get("mutau_mass").GetNbinsX()
50

The functor doesn't have to return a histogram.

>>> mean_getter = lambda x: x.GetMean()
>>> mean = zjets.Get("mutau_mass").GetMean()
>>> zjets_mean = FunctorView(zjets, mean_getter)
>>> zjets_mean.Get("mutau_mass") == mean
True


MultiFunctorView
----------------

MultiFunctorView is similar except that it operates on a group of histograms.
The functor should take one argument, a *generator* of the sub-objects.

Here's an example to get the integral of the biggest histogram in a set:

>>> biggest_histo = lambda objects: max(y.Integral() for y in objects)
>>> biggest = MultiFunctorView(biggest_histo, zjets, dibosons)
>>> biggest.Get("mutau_mass") == zjets.Get("mutau_mass").Integral()
True

SubdirectoryView
----------------

If you'd like to "cd" into a lower subdirectory, while still maintaining
the same view, use a SubdirectoryView.

>>> basedir = io.Directory('base', 'base directory')
>>> _ = basedir.cd()
>>> subdir1 = io.Directory('subdir1', 'subdir directory in 1')
>>> _ = subdir1.cd()
>>> hist = ROOT.TH1F("mutau_mass", "Mu-Tau mass", 100, 0, 100)
>>> hist.FillRandom('gaus', 2000)
>>> keep.append(hist)
>>> _ = basedir.cd()
>>> subdir2 = io.Directory('subdir2', 'subdir directory 2')
>>> _ = subdir2.cd()
>>> hist = ROOT.TH1F("mutau_mass", "Mu-Tau mass", 100, 0, 100)
>>> hist.FillRandom('gaus', 5000)
>>> keep.append(hist)

The directory structure is now::
    base/subdir1/hist
    base/subdir2/hist

Subdirectory views work on top of other views.

>>> baseview = ScaleView(basedir, 0.1)
>>> subdir1view = SubdirectoryView(baseview, 'subdir1')
>>> subdir2view = SubdirectoryView(baseview, 'subdir2')
>>> histo1 = subdir1view.Get('mutau_mass')
>>> histo2 = subdir2view.Get('mutau_mass')
>>> exp_histo1 = baseview.Get("subdir1/mutau_mass")
>>> exp_histo2 = baseview.Get("subdir2/mutau_mass")
>>> def equivalent(h1, h2):
...     return (abs(h1.GetMean() - h2.GetMean()) < 1e-4 and
...             abs(h1.GetRMS() - h2.GetRMS()) < 1e-4 and
...             abs(h1.Integral() - h2.Integral()) < 1e-4)
>>> equivalent(exp_histo1, histo1)
True
>>> equivalent(exp_histo2, histo2)
True
>>> equivalent(histo1, histo2)
False

'''
from __future__ import absolute_import

import os
import ROOT

from .base import Plottable
from .hist import HistStack
from ..io import Directory, DoesNotExist

__all__ = [
    'ScaleView',
    'NormalizeView',
    'StyleView',
    'TitleView',
    'SumView',
    'StackView',
    'FunctorView',
    'MultiFunctorView',
    'PathModifierView',
    'SubdirectoryView',
]


class _FolderView(object):
    '''
    Abstract view of an individual folder

    Provides one interface: Get(path) which returns a modified version
    of whatever exists at path.  Subclasses should define::

        apply_view(self, obj)

    which should return the modified [object] as necessary.

    The subclass can get access to the queried path via the self.getting
    variable.
    '''
    def __init__(self, directory):
        ''' Initialize with the directory to be wrapped '''
        self.dir = directory

    def path(self):
        ''' Get the path of the wrapped folder '''
        if isinstance(self.dir, Directory):
            return self.dir._path
        elif isinstance(self.dir, ROOT.TDirectory):
            return self.dir.GetPath()
        elif isinstance(self.dir, _FolderView):
            return self.dir.path()
        else:
            return str(self.dir)

    def __str__(self):
        return "{0}('{1}')".format(self.__class__.__name__, self.path())

    def Get(self, path):
        ''' Get the (modified) object from path '''
        self.getting = path
        try:
            obj = self.dir.Get(path)
            return self.apply_view(obj)
        except DoesNotExist as dne:
            #print dir(dne)
            raise DoesNotExist(
                str(dne) + "[{0}]".format(self.__class__.__name__))


class _MultiFolderView(object):
    '''
    Abstract view of a collection of folders

    Applies some type of "merge" operation to the result of the get from each
    folder.  Subclasses should define::

        merge_views(self, objects)

    which takes a *generator* of objects returns a merged object.

    The subclass can get access to the queried path via the self.getting
    variable.
    '''
    def __init__(self, *directories):
        self.dirs = directories

    def __str__(self):
        return "{0}({1})".format(
            self.__class__.__name__,
            ','.join(str(x) for x in self.dirs))

    def Get(self, path):
        ''' Merge the objects at path in all subdirectories '''
        return self.merge_views(x.Get(path) for x in self.dirs)


class ScaleView(_FolderView):
    ''' View of a folder which applies a scaling factor to histograms. '''
    def __init__(self, directory, scale_factor):
        super(ScaleView, self).__init__(directory)
        self.factor = scale_factor

    def apply_view(self, obj):
        if not hasattr(obj, 'Scale'):
            raise ValueError(
                "`ScaleView` can't determine how to handle"
                "an object of type `{0}`; "
                "it has no `Scale` method".format(type(obj)))
        clone = obj.Clone()
        clone.Scale(self.factor)
        return clone


class NormalizeView(ScaleView):
    ''' Normalize histograms to a constant value '''
    def __init__(self, directory, normalization=1.0):
        # Initialize the scale view with a dummy scale factor.
        # The scale factor is changed dynamically for each histogram.
        super(NormalizeView, self).__init__(directory, None)
        self.norm = normalization

    def apply_view(self, obj):
        current_norm = obj.Integral()
        # Update the scale factor (in the base)
        if current_norm > 0:
            self.factor = self.norm / current_norm
        else:
            self.factor = 0
        return super(NormalizeView, self).apply_view(obj)


class StyleView(_FolderView):
    '''
    View of a folder which applies a style to Plottable objects.

    The kwargs are passed to Plottable.decorate
    '''
    def __init__(self, directory, **kwargs):
        super(StyleView, self).__init__(directory)
        self.kwargs = kwargs

    def apply_view(self, obj):
        if not isinstance(obj, Plottable):
            raise TypeError(
                "`ScaleView` can't determine how to handle "
                "an object of type `{0}`; it is not a subclass of "
                "`Plottable`".format(type(obj)))
        clone = obj.Clone()
        clone.decorate(**self.kwargs)
        return clone


class TitleView(_FolderView):
    ''' Override the title of gotten histograms '''
    def __init__(self, directory, title):
        self.title = title
        super(TitleView, self).__init__(directory)

    def apply_view(self, obj):
        clone = obj.Clone()
        clone.SetTitle(self.title)
        return clone


class SumView(_MultiFolderView):
    ''' Add a collection of histograms together '''
    def __init__(self, *directories):
        super(SumView, self).__init__(*directories)

    def merge_views(self, objects):
        output = None
        for obj in objects:
            if output is None:
                output = obj.Clone()
            else:
                output += obj
        return output


class StackView(_MultiFolderView):
    '''
    Build a HistStack from the input histograms

    The default draw option that histograms will use is "hist".

    One can override this for all histograms by passing a string.
    Individual behavior can be controlled by passing a list of draw options,
    corresponding to the input directories. In this case the option for
    all histograms must be specified.

    The name and title of the HistStack is taken from the first histogram in
    the list.

    Normally the histograms will be added to the stack in the order
    of the constructor.  Optionally, one can add them in order of ascending
    integral by passing the kwarg sorted=True.
    '''
    def __init__(self, *directories, **kwargs):
        super(StackView, self).__init__(*directories)
        self.sort = kwargs.get(sorted, False)

    def merge_views(self, objects):
        output = None
        if self.sort:
            objects = sorted(objects, key=lambda x: x.Integral())
        for obj in objects:
            if output is None:
                output = HistStack(name=obj.GetName(),
                                   title=obj.GetTitle())
            output.Add(obj)
        return output


class FunctorView(_FolderView):
    '''
    Apply an arbitrary function to the output histogram.

    The histogram is always cloned before it is passed to the function.
    '''
    def __init__(self, directory, function):
        self.f = function
        super(FunctorView, self).__init__(directory)

    def apply_view(self, obj):
        clone = obj.Clone()
        return self.f(clone)


class MultiFunctorView(_MultiFolderView):
    '''
    Apply an arbitrary function to the output histograms.

    The function must take one argument, a generator of objects.
    '''
    def __init__(self, f, *directories):
        self.f = f
        super(MultiFunctorView, self).__init__(*directories)

    def merge_views(self, objects):
        return self.f(objects)


class PathModifierView(_FolderView):
    '''
    Does some magic to the path

    User should supply a functor which transforms the path argument
    passed to Get(...)
    '''
    def __init__(self, dir, path_modifier):
        self.path_modifier = path_modifier
        super(PathModifierView, self).__init__(dir)

    def Get(self, path):
        newpath = self.path_modifier(path)
        return super(PathModifierView, self).Get(newpath)

    def apply_view(self, obj):
        ''' Do nothing '''
        return obj


class SubdirectoryView(PathModifierView):
    '''
    Add some base directories to the path of Get()

    <subdir> is the directory you want to 'cd' too.
    '''
    def __init__(self, dir, subdirpath):
        functor = lambda path: os.path.join(subdirpath, path)
        super(SubdirectoryView, self).__init__(dir, functor)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
