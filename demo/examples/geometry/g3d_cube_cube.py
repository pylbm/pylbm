

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 3D geometry: the cube [0,1] x [0,1] x [0,1]
with a cubic hole
"""
import numpy as np
import pylbm

# pylint: disable=invalid-name

a, b = 1./np.sqrt(3), 1./np.sqrt(2)
c = a*b
v0 = [a, a, a]
v1 = [b, -b, 0]
v2 = [c, c, -2*c]
v0, v1, v2 = [1, 0, 0], [0, 1, 0], [0, 0, 1]
dgeom = {
    'box': {
        'x': [-3, 3],
        'y': [-3, 3],
        'z': [-3, 3],
        'label': list(range(6))
    },
    'elements': [
        pylbm.Parallelepiped(
            (0, 0, 0), v0, v1, v2, label=0
        )
    ],
}
geom = pylbm.Geometry(dgeom)
print(geom)
geom.visualize(viewlabel=True)
