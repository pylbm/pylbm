# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 3D domain:
the cube [-3, 3] x [-3, 3] x [-3, 3] with a cylindrical hole
"""
import pylbm

V1 = [0, 1.0, 1.0]
V2 = [0, -1.5, 1.5]
V3 = [1, 0, 0]
D_DOM = {
    "box": {"x": [-3, 3], "y": [-3, 3], "z": [-3, 3], "label": 0},
    "elements": [pylbm.CylinderEllipse((0.5, 0, 0), V1, V2, V3, label=[1, 2, 3])],
    "space_step": 1,
    "schemes": [{"velocities": list(range(19))}],
}
DOMAIN = pylbm.Domain(D_DOM)
print(DOMAIN)
DOMAIN.visualize(
    view_distance=False,
    view_in=False,
    view_out=False,
    view_bound=True,
    view_geom=True,
    label=[1, 2, 3],
)
DOMAIN.visualize(
    view_distance=False,
    view_in=False,
    view_out=False,
    view_bound=True,
    view_normal=True,
    label=[1, 2, 3],
)
