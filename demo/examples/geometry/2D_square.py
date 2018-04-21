from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 2D geometry: the square [0,1] x [0,1]
"""
import pylbm
dgeom = {
    'box':{'x': [0, 1], 'y': [0, 1], 'label':0},
}
geom = pylbm.Geometry(dgeom)
#geom.visualize(pylbm.viewer.vispyViewer, viewlabel=True)
geom.visualize()
