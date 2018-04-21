from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a square in 2D with a circular hole
"""
from six.moves import range
import pylbm
dico = {
    'box':{'x': [0, 2], 'y': [0, 1], 'label':0},
    'elements':[pylbm.Circle((0.5,0.5), 0.2, label = 1)],
    'space_step':0.05,
    'schemes':[{'velocities':list(range(13))}],
}
dom = pylbm.Domain(dico)
dom.visualize()
dom.visualize(view_distance=True, label=1)
