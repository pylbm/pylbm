from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 2 velocities scheme in 1D
"""
from six.moves import range
import pylbm
dsten = {
    'dim':1,
    'schemes':[{'velocities':range(1,3)},],
}
s = pylbm.Stencil(dsten)
print(s)
s.visualize()
