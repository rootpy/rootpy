import gc
import weakref

import ROOT as R

import rootpy.plotting
from rootpy.context import invisible_canvas


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
        import rootpy.memory.keepalive as K
        K.KEEPALIVE.clear()

        # ROOT automatically cleans things up like this on deletion, and since
        # we cleared the keepalive dictionary, they should have gone away.
        assert c.GetListOfPrimitives().GetSize() == 0
