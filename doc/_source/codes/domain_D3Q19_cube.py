# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of the cube in 3D with a D3Q19
"""
import pyLBM
dico = {
    'box':{'x': [0, 2], 'y': [0, 2], 'z':[0, 2]},
    'space_step':1,
    'schemes':[{'velocities':range(19)}]
}
dom = pyLBM.Domain(dico)
dom.visualize(view_distance=True)
