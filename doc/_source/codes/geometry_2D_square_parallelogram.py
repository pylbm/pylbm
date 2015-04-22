# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 2D geometry: the square [0,1]x[0,1] with a step
"""
import pyLBM
d = {'box':{'x': [0, 3], 'y': [0, 1], 'label':[1, 2, 0, 0]},
     'elements':[pyLBM.Parallelogram((0.,0.), (.5,0.), (0., .5), label = 0)],
}
g = pyLBM.Geometry(d)
g.visualize()
