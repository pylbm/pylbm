

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of two different two velocities schemes in 2D
"""
from six.moves import range
import pylbm

# pylint: disable=invalid-name

dsten = {
    'dim': 2,
    'schemes': [{'velocities': list(range(1, 5))}],
}
s = pylbm.Stencil(dsten)
print(s)
s.visualize()

dsten = {
    'dim': 2,
    'schemes': [{'velocities': list(range(5, 9))}],
}
s = pylbm.Stencil(dsten)
print(s)
s.visualize()
