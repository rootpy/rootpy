# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License

import gc
import weakref

import ROOT as R

import rootpy.plotting
from rootpy.context import invisible_canvas
from rootpy.memory.deletion import monitor_deletion


def test_keepalive():
    class went_away:
        value = False

    def callback(*args):
        went_away.value = True

    with invisible_canvas() as c:
        c.cd()

        # No primitives to start with
        assert c.GetListOfPrimitives().GetSize() == 0

        h = R.TH1F()
        h.Draw()

        hproxy = weakref.proxy(h, callback)

        # Now we've got one primitive on the canvas
        assert c.GetListOfPrimitives().GetSize() == 1

        del h
        gc.collect()
        # We should still have it due to the keepalive
        assert c.GetListOfPrimitives().GetSize() == 1

    # Canvas should now have gone away
    assert not c

    # And so should the histogram object
    assert went_away.value

def test_nokeepalive():
    with invisible_canvas() as c:

        assert c.GetListOfPrimitives().GetSize() == 0

        h = R.TH1F()
        h.Draw()

        assert c.GetListOfPrimitives().GetSize() == 1
        del h
        from rootpy.memory import KEEPALIVE
        KEEPALIVE.clear()

        # ROOT automatically cleans things up like this on deletion, and since
        # we cleared the keepalive dictionary, they should have gone away.
        assert c.GetListOfPrimitives().GetSize() == 0

def test_canvas_divide():
    monitor, is_alive = monitor_deletion()

    with invisible_canvas() as c:
        monitor(c, "c")

        c.Divide(2)

        p = c.cd(1)

        monitor(p, "p")
        assert is_alive("p")

        h = R.TH1F()
        h.Draw()
        monitor(h, "h")

        assert is_alive("h")
        del h
        assert is_alive("h")

        del p
        # p should be kept alive because of the canvas
        assert is_alive("p")
        # h should still be alive because of the pad
        assert is_alive("h")

        c.Clear()

        # clearing the canvas means that the pad (and therefore the hist) should
        # be deleted.
        assert not is_alive("p")
        assert not is_alive("h")

        # -------------
        # Next test, check that when the canvas is deleted, everything goes away

        p = c.cd(2)
        h = R.TH1F()
        h.Draw()

        monitor(p, "p")
        monitor(p, "h")

        del p
        del h

        assert is_alive("p")
        assert is_alive("h")

    # The canvas is deleted by exiting the with statement.
    # Everything should go away.
    assert not is_alive("c")
    assert not is_alive("p")
    assert not is_alive("h")

