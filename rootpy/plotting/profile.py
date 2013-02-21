from .. import QROOT, log; log = log[__name__]
from .hist import _Hist, _Hist2D, _Hist3D


class Profile(_Hist, QROOT.TProfile):
    pass


class Profile2D(_Hist2D, QROOT.TProfile):
    pass


class Profile3D(_Hist3D, QROOT.TProfile2D):
    pass
