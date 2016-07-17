from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of two different two velocities schemes in 2D
"""
from six.moves import range
import pyLBM
dsten = {
    'dim':2,
    'schemes':[{'velocities':list(range(1,5))}],
}
s = pyLBM.Stencil(dsten)
print(s)
s.visualize()

dsten = {
    'dim':2,
    'schemes':[{'velocities':list(range(5,9))}],
}
s = pyLBM.Stencil(dsten)
print(s)
s.visualize()
