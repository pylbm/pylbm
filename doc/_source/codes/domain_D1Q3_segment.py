# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a segment in 1D with a D1Q3
"""
import pyLBM
dico = {
    'box':{'x': [0, 1],},
    'space_step':0.1,
    'schemes':[{'velocities':range(3)}],
}
dom = pyLBM.Domain(dico)
dom.visualize()
