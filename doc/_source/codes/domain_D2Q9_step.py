# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of the backward facing step in 2D with a D2Q9
"""
from six.moves import range
import pyLBM
dico = {
    'box':{'x': [0, 3], 'y': [0, 1], 'label':0},
    'elements':[pyLBM.Parallelogram((0.,0.), (.5,0.), (0., .5), label=1)],
    'space_step':0.125,
    'schemes':[{'velocities':list(range(9))}],
}
dom = pyLBM.Domain(dico)
dom.visualize()
dom.visualize(view_distance=True, label=1)
