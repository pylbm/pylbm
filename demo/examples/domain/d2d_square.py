

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a square in 2D
"""
from six.moves import range
import pylbm

# pylint: disable=invalid-name

ddom = {
    'box': {'x': [0, 1], 'y': [0, 1], 'label': [0, 1, 2, 3]},
    'space_step': 0.1,
    'schemes': [
        {
            'velocities': list(range(9))
        }
    ],
}
dom = pylbm.Domain(ddom)
print(dom)
dom.visualize()
dom.visualize(view_distance=True)
