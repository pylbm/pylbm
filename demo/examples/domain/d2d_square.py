# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a square in 2D
"""
import pylbm

D_DOM = {
    "box": {"x": [0, 1], "y": [0, 1], "label": [0, 1, 2, 3]},
    "space_step": 0.1,
    "schemes": [{"velocities": list(range(9))}, {"velocities": list(range(25))}],
}
DOMAIN = pylbm.Domain(D_DOM)
print(DOMAIN)
DOMAIN.visualize()
DOMAIN.visualize(view_bound=True, scale=3)
DOMAIN.visualize(view_distance=True)
DOMAIN.visualize(view_distance=True, view_bound=True, view_normal=True, scale=3)
