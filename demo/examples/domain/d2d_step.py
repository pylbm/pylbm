# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of the backward facing step in 2D
"""
import pylbm

D_DOM = {
    "box": {"x": [0, 3], "y": [0, 1], "label": [0, 1, 0, 2]},
    "elements": [pylbm.Parallelogram((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), label=0)],
    "space_step": 0.125,
    "schemes": [{"velocities": list(range(9))}],
}
DOMAIN = pylbm.Domain(D_DOM)
print(DOMAIN)
DOMAIN.visualize()
DOMAIN.visualize(view_distance=True, view_normal=True)
