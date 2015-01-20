import pyLBM
from pyLBM.viewer import MatplotlibViewer

if __name__ == "__main__":
    dsten = {
        'dim':2,
        'schemes':[{'velocities':range(9)}
                   ],
    }
    s = pyLBM.Stencil(dsten)
    print s
    v = MatplotlibViewer()
    s.visualize(v)
