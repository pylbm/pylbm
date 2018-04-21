from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 19 velocities scheme in 3D
"""
from six.moves import range
import pylbm
dsten = {
    'dim':3,
    'schemes':[{'velocities':list(range(19))},],
}
s = pylbm.Stencil(dsten)
print(s)
s.visualize(pylbm.viewer.matplotlibViewer)
