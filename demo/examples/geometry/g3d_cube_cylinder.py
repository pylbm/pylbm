

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 3D geometry: the cube [0,1] x [0,1] x [0,1]
with a cylindrical hole
"""
import pylbm

# pylint: disable=invalid-name

v1 = [0, 1., 1.]
v2 = [0, -1.5, 1.5]
v3 = [1, -1, 0]
w1 = [.5, 0, 0]
w2 = [0, .5, 0]
w3 = [0, 0, 1.5]
dgeom = {
    'box': {
        'x': [-3, 3],
        'y': [-3, 3],
        'z': [-3, 3],
        'label': 9
    },
    'elements': [
        # pylbm.CylinderEllipse((0.5,0,0), v1, v2, v3, label=[1,0,0]),
        pylbm.CylinderTriangle((0.5, 0, 0), v1, v2, v3, label=0),
        pylbm.CylinderCircle((-1.5, -1.5, 0), w1, w2, w3, label=[1, 0, 0]),
    ],
}
geom = pylbm.Geometry(dgeom)
print(geom)
geom.visualize(viewlabel=True)
