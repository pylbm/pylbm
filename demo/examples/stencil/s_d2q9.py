

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 9 velocities scheme in 2D
"""
from six.moves import range
import pylbm

# pylint: disable=invalid-name

dsten = {
    'dim': 2,
    'schemes': [{'velocities': list(range(9))}],
}
s = pylbm.Stencil(dsten)
print(s)
s.visualize()
