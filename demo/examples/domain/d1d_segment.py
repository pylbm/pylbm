

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a segment in 1D
"""
from six.moves import range
import pylbm

# pylint: disable=invalid-name

ddom = {
    'box': {
        'x': [0, 1],
        'label': 0
    },
    'space_step': 0.1,
    'schemes': [
        {
            'velocities': list(range(3))
        }
    ],
}
dom = pylbm.Domain(ddom)
print(dom)
dom.visualize()
dom.visualize(
    view_distance=True,
    label=0,
    view_in=True,
    view_out=True,
    view_bound=True,
)
