# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of the cube in 3D with a D3Q19
"""
from six.moves import range
import pylbm
dico = {
    'box': {'x': [-3, 3], 'y': [-3, 3], 'z': [-3, 3], 'label': 0},
    'elements': [pylbm.STLElement("icosaedre.stl", label=1)],
    'space_step': .5,
    'schemes': [{'velocities': list(range(19))}]
}
dom = pylbm.Domain(dico)
dom.visualize()
dom.visualize(
    view_distance=False, view_bound=True,
    label=1, view_in=False, view_out=False
)
dom.visualize(
    view_distance=True, view_bound=True,
    label=1, view_in=False, view_out=False
)
dom.visualize(
    view_normal=True, view_bound=True,
    label=1, view_in=False, view_out=False
)
