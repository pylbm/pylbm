# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of the cube in 3D with a D3Q19
"""
from six.moves import range
import pylbm
dico = {
    'box':{'x': [0, 2], 'y': [0, 2], 'z':[0, 2], 'label':0},
    'space_step':.5,
    'schemes':[{'velocities':list(range(19))}]
}
dom = pylbm.Domain(dico)
dom.visualize()
dom.visualize(view_distance=True)
