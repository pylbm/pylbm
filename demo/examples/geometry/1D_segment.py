# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 1D geometry: the segment [0,1]
"""
import pyLBM
dgeom = {'box':{'x': [0, 1], 'label': [0,1]},}
geom = pyLBM.Geometry(dgeom)
geom.visualize(viewer_app=pyLBM.viewer.VispyViewer, viewlabel = True)

geom.visualize(viewlabel = True)
