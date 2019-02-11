

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a vectorial 2, 3 velocities scheme in 1D
"""
from six.moves import range
import pylbm

# pylint: disable=invalid-name

dsten = {
    'dim': 1,
    'schemes': [
        {'velocities': list(range(1, 3))},
        {'velocities': list(range(3))},
    ],
}
s = pylbm.Stencil(dsten)
print(s)
s.visualize()
