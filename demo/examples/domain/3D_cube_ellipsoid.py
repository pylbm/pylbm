from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of the cube in 3D with an ellipsoidal hole
"""
from six.moves import range
import pylbm
dico = {
    'box':{'x': [0, 3], 'y': [0, 3], 'z':[0, 3], 'label':list(range(1,7))},
    'elements':[pylbm.Ellipsoid((1.5,1.5,1.5), [.5,0,0], [0,.5,0], [0,0,1], label = 0)],
    'space_step':0.25,
    'schemes':[{'velocities':list(range(19))}]
}
dom = pylbm.Domain(dico)
print(dom)
dom.visualize(view_distance=False, label=0, view_in=False, view_out=False, view_bound=True)
