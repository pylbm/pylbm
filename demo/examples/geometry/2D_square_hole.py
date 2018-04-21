from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 2D geometry: the square [0,1]x[0,1] with a circular hole
"""
import pylbm
dgeom = {
    'box':{'x': [0, 1], 'y': [0, 1], 'label':0},
    'elements':[pylbm.Circle((0.5,0.5), 0.125, label = 1)],
}
geom = pylbm.Geometry(dgeom)
geom.visualize(viewlabel=True)
