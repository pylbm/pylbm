

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 3D geometry: the cube [0,1] x [0,1] x [0,1]
with a ellipsoidal hole
"""
import pylbm

# pylint: disable=invalid-name

v1 = [.3, .3, 0]
v2 = [.25, -.25, 0]
v3 = [0, 0, .125]
dgeom = {
    'box': {
        'x': [0, 1],
        'y': [0, 1],
        'z': [0, 1],
        'label': 0
    },
    'elements': [
        pylbm.Ellipsoid((.5, .5, .5), v1, v2, v3, label=1)
    ],
}
geom = pylbm.Geometry(dgeom)
print(geom)
geom.visualize(viewlabel=True)
