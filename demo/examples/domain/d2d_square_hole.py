

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a square in 2D with a circular hole
"""
import pylbm

D_DOM = {
    'box': {'x': [0, 2], 'y': [0, 1], 'label': 0},
    'elements': [
        pylbm.Circle((0.5, 0.5), 0.2, label=1),
    ],
    'space_step': 0.05,
    'schemes': [
        {
            'velocities': list(range(13))
        }
    ],
}
DOMAIN = pylbm.Domain(D_DOM)
print(DOMAIN)
DOMAIN.visualize(scale=1.5)
DOMAIN.visualize(
    view_distance=True,
    label=None,
    view_in=True,
    view_out=True,
    view_bound=True
)
DOMAIN.visualize(
    view_distance=True,
    label=None,
    view_in=True,
    view_out=True,
    view_bound=True,
    view_normal=True,
)
