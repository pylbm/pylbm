# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 19 velocities scheme in 3D
"""
import pyLBM
dsten = {
    'dim':3,
    'schemes':[{'velocities':range(19)},],
}
s = pyLBM.Stencil(dsten)
print s
s.visualize(pyLBM.viewer.vispyViewer)
