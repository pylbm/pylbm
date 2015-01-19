import pyLBM
import pyLBM.stencil as LBMStencil
from pyLBM.viewer import MatplotlibViewer, VtkViewer

if __name__ == "__main__":
    dsten = {
        'dim':2,
        'schemes':[{'velocities':range(9)},
                   {'velocities':range(25)},
                   {'velocities':range(49)},
                   ],
    }
    s = LBMStencil.Stencil(dsten)
    v = MatplotlibViewer()
    s.visualize(v, k=0)
    s.visualize(v, k=1)
    s.visualize(v, k=2)

    dsten = {
        'dim':3,
        'number_of_schemes':1,
        'schemes':[{'velocities':range(19)},],
    }
    s = LBMStencil.Stencil(dsten)
    v = VtkViewer()
    s.visualize(v, k=0)
