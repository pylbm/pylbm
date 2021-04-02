# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Visualization options for the domain
"""
from six.moves import range
import pylbm
dico = {
    'box': {'x': [0, 1], 'label': [0, 1]},
    'space_step': 0.1,
    'schemes': [{'velocities': list(range(3))}],
}
dom = pylbm.Domain(dico)
dom.visualize(view_in=False, view_out=True, view_bound=True)
dom.visualize(view_bound=True, view_distance=True)
dom.visualize(view_bound=True, view_distance=True, view_normal=True)
