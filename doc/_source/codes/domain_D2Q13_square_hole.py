# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a square in 2D with a circular hole with a D2Q13
"""
from six.moves import range
import pylbm
dico = {
    'box': {'x': [0, 1], 'y': [0, 1], 'label': 0},
    'elements': [pylbm.Circle((0.5, 0.5), 0.2, label=1)],
    'space_step': 0.05,
    'schemes': [{'velocities': list(range(13))}],
}
dom = pylbm.Domain(dico)
dom.visualize(view_bound=True, scale=0.25)
dom.visualize(view_distance=True, scale=0.25)
dom.visualize(view_normal=True, scale=0.25, label=[1])
