import pyLBM
from pyLBM.viewer import VtkViewer

if __name__ == "__main__":
    dsten = {
        'dim':3,
        'number_of_schemes':1,
        'schemes':[{'velocities':range(19)},],
    }
    s = pyLBM.Stencil(dsten)
    print s
    v = VtkViewer()
    s.visualize(v)
