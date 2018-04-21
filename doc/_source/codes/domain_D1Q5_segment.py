# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a segment in 1D with a D1Q5
"""
from six.moves import range
import pylbm
dico = {
    'box':{'x': [0, 1], 'label':0},
    'space_step':0.1,
    'schemes':[{'velocities':list(range(5))}],
}
dom = pylbm.Domain(dico)
dom.visualize()
