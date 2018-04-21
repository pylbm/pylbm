from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 3D geometry: the cube [0,1] x [0,1] x [0,1] with a cylindrical hole
"""
import pylbm

v1 = [0, 1., 1.]
v2 = [0,-1.5, 1.5]
v3 = [1, -1, 0]
w1 = [.5,0,0]
w2 = [0,.5,0]
w3 = [0,0,1.5]
dico = {
    'box':{'x': [-3, 3], 'y': [-3, 3], 'z':[-3, 3], 'label':9},
    'elements':[#pylbm.Cylinder_Ellipse((0.5,0,0), v1, v2, v3, label=[1,0,0]),
                pylbm.Cylinder_Triangle((0.5,0,0), v1, v2, v3, label=0),
                pylbm.Cylinder_Circle((-1.5,-1.5,0), w1, w2, w3, label=[1,0,0]),],
}
geom = pylbm.Geometry(dico)
print(geom)
geom.visualize(viewlabel=True)
