# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 1D geometry: the segment [0,1]
"""
import pylbm
d = {'box':{'x': [0, 1], 'label': [0, 1]}}
g = pylbm.Geometry(d)
g.visualize(viewlabel = True)
