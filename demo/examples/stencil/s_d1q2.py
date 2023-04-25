# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 2 velocities scheme in 1D
"""

import pylbm

# pylint: disable=invalid-name

dsten = {
    "dim": 1,
    "schemes": [{"velocities": list(range(1, 3))}],
}
s = pylbm.Stencil(dsten)
print(s)
s.visualize()
