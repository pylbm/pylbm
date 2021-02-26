

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of the cube in 3D
"""
import pylbm

D_DOM = {
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
DOMAIN = pylbm.Domain(D_DOM)
print(DOMAIN)
DOMAIN.visualize(view_distance=True)
DOMAIN.visualize(view_normal=True)
