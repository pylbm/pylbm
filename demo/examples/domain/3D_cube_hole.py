# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of the cube in 3D with a spherical hole
"""
import pyLBM
dico = {
    'box':{'x': [0, 2], 'y': [0, 2], 'z':[0, 2], 'label':range(1,7)},
    'elements':[pyLBM.Sphere((1,1,1), 0.5, label = 0)],
    'space_step':0.25,
    'schemes':[{'velocities':range(19)}]
}
dom = pyLBM.Domain(dico)
print dom
dom.visualize(view_distance=[1,3], view_in=False, view_out=False, view_bound=True)
