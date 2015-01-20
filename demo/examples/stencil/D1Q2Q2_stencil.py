import pyLBM
from pyLBM.viewer import MatplotlibViewer

if __name__ == "__main__":
    dsten = {
        'dim':1,
        'schemes':[{'velocities':range(1,3)},
                   {'velocities':range(1,3)},
                   ],
    }
    s = pyLBM.Stencil(dsten)
    print s
    v = MatplotlibViewer()
    s.visualize(v)
