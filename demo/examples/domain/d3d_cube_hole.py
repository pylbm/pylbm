

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of the cube in 3D with a spherical hole
"""
import pylbm

D_DOM = {
    'box': {
        'x': [0, 2],
        'y': [0, 2],
        'z': [0, 2],
        'label': list(range(1, 7))
    },
    'elements': [
        pylbm.Sphere((1, 1, 1), 0.5, label=0)
    ],
    'space_step': 0.25,
    'schemes': [
        {
            'velocities': list(range(19))
        }
    ]
}
DOMAIN = pylbm.Domain(D_DOM)
print(DOMAIN)
DOMAIN.visualize(
    view_distance=[1, 3],
    view_in=False,
    view_out=False,
    view_bound=True
)
DOMAIN.visualize(
    view_distance=False,
    view_in=False,
    view_out=False,
    view_bound=True,
    view_normal=True,
)
