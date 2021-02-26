

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a square in 2D with a cavity
"""
import pylbm

D_DOM = {
    'box': {'x': [0, 1], 'y': [0, 1], 'label': 0},
    'elements': [
        pylbm.Parallelogram((0.4, 0.3), (0, 0.4), (0.2, 0), label=1),
        pylbm.Circle((0.4, 0.5), 0.2, label=3),
        pylbm.Circle((0.6, 0.5), 0.2, label=3),
        pylbm.Parallelogram(
            (0.45, 0.3), (0, 0.4), (0.1, 0),
            label=2, isfluid=True
        ),
    ],
    'space_step': 0.025,
    'schemes': [
        {
            'velocities': list(range(9))
        }
    ],
}
DOMAIN = pylbm.Domain(D_DOM)
print(DOMAIN)
DOMAIN.visualize(
    view_distance=True,
    label=None,
    view_in=True,
    view_out=True,
    view_bound=True
)
DOMAIN.visualize(
    view_distance=False,
    label=None,
    view_in=True,
    view_out=True,
    view_bound=True,
    view_normal=True,
)
