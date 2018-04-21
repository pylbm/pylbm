# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 2D geometry: the square [0,1]x[0,1] with a circular hole
"""
import pylbm
d = {'box':{'x': [0, 1], 'y': [0, 1], 'label':0},
     'elements':[pylbm.Circle((.5, .5), .125, label = 1)],
}
g = pylbm.Geometry(d)
g.visualize(viewlabel=True)
