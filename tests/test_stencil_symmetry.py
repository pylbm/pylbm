import pyLBM
import numpy as np

def test_1D():
    dsten = {
        'dim':1,
        'schemes':[{'velocities':range(5)}, {'velocities':[2,1,0,5,0,6]}, {'velocities':range(6)}],
    }
    s = pyLBM.Stencil(dsten)

def test_2D():
    dsten = {
        'dim':2,
        'schemes':[{'velocities':range(9)}, {'velocities':[3,1,0,5,0,7]}, {'velocities':range(6)}],
    }
    s = pyLBM.Stencil(dsten)

def test_3D():
    pass

if __name__ == "__main__":
    #test_1D()
    test_2D()
    #test_3D()
