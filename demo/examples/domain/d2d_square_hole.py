

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
    'box': {'x': [0, 2], 'y': [0, 1], 'label': 0},
    'elements': [
        pylbm.Circle((0.5, 0.5), 0.2, label=1)
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
dom.visualize(scale=1.5)
dom.visualize(
    view_distance=True,
    label=None,
    view_in=True,
    view_out=True,
    view_bound=True
)
