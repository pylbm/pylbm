

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of the cube in 3D
"""
from six.moves import range
import pylbm

# pylint: disable=invalid-name

ddom = {
    'box': {
        'x': [0, 2],
        'y': [0, 2],
        'z': [0, 2],
        'label': list(range(6))
    },
    'space_step': 0.5,
    'schemes': [
        {
            'velocities': list(range(19))
        }
    ]
}
dom = pylbm.Domain(ddom)
print(dom)
dom.visualize(view_distance=True)
