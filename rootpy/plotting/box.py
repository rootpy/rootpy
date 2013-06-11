
from .. import QROOT
from ..core import Object

class Pave(Object, QROOT.TPave):
    
    @property
    def position(self):
        return self.GetX1NDC(), self.GetY1NDC(), self.GetX2NDC(), self.GetY2NDC()

    @position.setter
    def position(self, value):
        x1, y1, x2, y2 = value
        self.SetX1NDC(x1)
        self.SetY1NDC(y1)
        self.SetX2NDC(x2)
        self.SetY2NDC(y2)

