

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of the cube in 3D with a icosaedric hole
"""
import pylbm

icosaedre = pylbm.STLElement('icosaedre.stl', label=0)

a, b = -3, 3
D_DOM = {
    'box': {
        'x': [a, b],
        'y': [a, b],
        'z': [a, b],
        'label': list(range(1, 7))
    },
    'elements': [
        icosaedre
    ],
    'space_step': 0.1,
    'schemes': [
        {
            'velocities': list(range(7))
        }
    ]
}
DOMAIN = pylbm.Domain(D_DOM)
print(DOMAIN)
DOMAIN.visualize(
    view_distance=False,
    label=0,
    view_in=False,
    view_out=True,
    view_bound=False,
    view_normal=False,
    view_geom=True,
)
DOMAIN.visualize(
    view_distance=True,
    view_in=False,
    view_out=False,
    view_bound=True,
    view_normal=True
)
# DOMAIN.visualize(
#     view_distance=False,
#     view_in=False,
#     view_out=False,
#     view_bound=True,
#     view_normal=True,
# )
