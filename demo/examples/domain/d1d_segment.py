# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a segment in 1D
"""
import pylbm

D_DOM = {
    "box": {"x": [0, 1], "label": 0},
    "space_step": 0.1,
    "schemes": [{"velocities": list(range(3))}],
}
DOMAIN = pylbm.Domain(D_DOM)
print(DOMAIN)
DOMAIN.visualize()
DOMAIN.visualize(
    view_distance=True,
    label=0,
    view_in=True,
    view_out=True,
    view_bound=True,
    view_normal=True,
)
