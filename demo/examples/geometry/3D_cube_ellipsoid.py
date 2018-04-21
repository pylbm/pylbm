from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 3D geometry: the cube [0,1] x [0,1] x [0,1] with a ellipsoidal hole
"""
import pylbm

v1 = [.3, .3, 0]
v2 = [.25, -.25, 0]
v3 = [0, 0, .125]
dico = {
    'box':{'x': [0, 1], 'y': [0, 1], 'z':[0, 1], 'label':0},
    'elements':[pylbm.Ellipsoid((.5,.5,.5), v1, v2, v3, label=1)],
}
geom = pylbm.Geometry(dico)
print(geom)
geom.visualize(viewlabel=True)
