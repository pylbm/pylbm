# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 3D geometry: the cube [0,1]x[0,1]x[0,1]
"""

import pylbm
d = {'box': {'x': [0, 1], 'y': [0, 1], 'z': [0, 1], 'label': list(range(6))}}
g = pylbm.Geometry(d)
g.visualize(viewlabel=True)
