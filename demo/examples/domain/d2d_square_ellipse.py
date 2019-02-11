

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a square in 2D with a circular hole
"""
from six.moves import range
import pylbm

# pylint: disable=invalid-name

ddom = {
    'box': {
        'x': [0, 2],
        'y': [0, 1],
        'label': 0
    },
    'elements': [
        pylbm.Ellipse((0.5, 0.5), (0.25, 0.25), (0.1, -0.1), label=4)
    ],
    'space_step': 0.05,
    'schemes': [
        {
            'velocities': list(range(13))
        }
    ],
}
dom = pylbm.Domain(ddom)
print(dom)
dom.visualize()
dom.visualize(view_distance=True, view_bound=True, label=4)
