# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 3D geometry: the cube [0,1]x[0,1]x[0,1]
"""
import pylbm
d = {
    'box':{'x': [0, 1], 'y': [0, 1], 'z':[0, 1], 'label':0},
    'elements':[pylbm.Sphere((.5,.5,.5), .25, label=1)],
}
g = pylbm.Geometry(d)
g.visualize(viewlabel=True)
