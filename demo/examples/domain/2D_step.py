from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of the backward facing step in 2D
"""
from six.moves import range
import pylbm
dico = {
    'box':{'x': [0, 3], 'y': [0, 1], 'label':[0, 1, 0, 2]},
    'elements':[pylbm.Parallelogram((0.,0.), (.5,0.), (0., .5), label=0)],
    'space_step':0.125,
    'schemes':[{'velocities':list(range(9))}],
}
dom = pylbm.Domain(dico)
dom.visualize()
dom.visualize(view_distance=True)
