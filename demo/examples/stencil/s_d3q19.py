

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 19 velocities scheme in 3D
"""
from six.moves import range
import pylbm

# pylint: disable=invalid-name

dsten = {
    'dim': 3,
    'schemes': [{'velocities': list(range(19))}],
}
s = pylbm.Stencil(dsten)
print(s)
s.visualize(pylbm.viewer.matplotlib_viewer)
