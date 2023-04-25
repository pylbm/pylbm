# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of the cube in 3D with an ellipsoidal hole
"""
import pylbm

D_DOM = {
    "box": {"x": [0, 3], "y": [0, 3], "z": [0, 3], "label": list(range(1, 7))},
    "elements": [
        pylbm.Ellipsoid(
            [1.5, 1.5, 1.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0], label=0
        )
    ],
    "space_step": 0.25,
    "schemes": [{"velocities": list(range(19))}],
}
DOMAIN = pylbm.Domain(D_DOM)
print(DOMAIN)
DOMAIN.visualize(
    view_distance=False,
    label=0,
    view_in=False,
    view_out=False,
    view_bound=True,
    view_normal=True,
)
