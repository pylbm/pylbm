

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 1D geometry: the segment [0,1]
"""
import pylbm
dgeom = {'box':{'x': [0, 1], 'label': [0,1]},}
geom = pylbm.Geometry(dgeom)
geom.visualize(viewlabel = True)
