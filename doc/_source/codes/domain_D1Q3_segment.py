# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a segment in 1D with a D1Q3
"""

import pylbm
dico = {
    'box': {'x': [0, 1], 'label': 0},
    'space_step': 0.1,
    'schemes': [{'velocities': list(range(3))}],
}
dom = pylbm.Domain(dico)
dom.visualize()
