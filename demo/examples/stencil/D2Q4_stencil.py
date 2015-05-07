# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of two different two velocities schemes in 2D
"""
import pyLBM
from pyLBM.viewer import MatplotlibViewer
dsten = {
    'dim':2,
    'schemes':[{'velocities':range(1,5)}],
}
s = pyLBM.Stencil(dsten)
print s
v = MatplotlibViewer()
s.visualize(v)

dsten = {
    'dim':2,
    'schemes':[{'velocities':range(5,9)}],
}
s = pyLBM.Stencil(dsten)
print s
v = MatplotlibViewer()
s.visualize(v)
