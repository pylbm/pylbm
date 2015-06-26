# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a square in 2D
"""
import pyLBM
dico = {
    'box':{'x': [0, 1], 'y': [0, 1]},
    'space_step':0.1,
    'schemes':[{'velocities':range(9)}],
}
dom = pyLBM.Domain(dico)
dom.visualize()
dom.visualize(view_distance=True)
