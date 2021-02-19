

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a square in 2D with a square hole
"""
from six.moves import range
import pylbm

# pylint: disable=invalid-name

ddom = {
    'box': {'x': [0, 2], 'y': [0, 1], 'label': 0},
    'elements': [
        pylbm.Parallelogram((0.5, 0.5), (0, 0.2), (0.2, 0), label=1),
        pylbm.Parallelogram((0.5, 0.5), (0, 0.1), (0.1, 0), label=2, isfluid=True),
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
dom.visualize(
    view_distance=True,
    label=None,
    view_in=True,
    view_out=True,
    view_bound=True,
    view_normal=True,
)
