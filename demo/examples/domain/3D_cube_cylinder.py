from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 3D domain: the cube [0,1] x [0,1] x [0,1] with a cylindrical hole
"""
from six.moves import range
import pylbm

v1 = [0, 1., 1.]
v2 = [0,-1.5, 1.5]
v3 = [1, -1, 0]
dico = {
    'box':{'x': [-3, 3], 'y': [-3, 3], 'z':[-3, 3], 'label':0},
    'elements':[pylbm.Cylinder_Ellipse((0.5,0,0), v1, v2, v3, label=[1,2,3])],
    'space_step':.5,
    'schemes':[{'velocities':list(range(19))}]
}
dom = pylbm.Domain(dico)
print(dom)
dom.visualize(view_distance=False, view_in=False, view_out=False, view_bound=True, label=[1,2,3])
