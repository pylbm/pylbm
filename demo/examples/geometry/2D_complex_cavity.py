from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import pyLBM

square = pyLBM.Parallelogram((.1, .1), (.8, 0), (0, .8), isfluid=False)
strip = pyLBM.Parallelogram((0, .4), (1, 0), (0, .2), isfluid=True)
circle = pyLBM.Circle((.5, .5), .25, isfluid=True)
inner_square = pyLBM.Parallelogram((.4, .5), (.1, .1), (.1, -.1), isfluid=False)
dgeom = {
    'box':{'x': [0, 1], 'y': [0, 1], 'label':0},
    'elements':[square, strip, circle, inner_square],
}
geom = pyLBM.Geometry(dgeom)
#geom.visualize()
# rounded inner angles
geom.add_elem(pyLBM.Parallelogram((0.1, 0.9), (0.05, 0), (0, -0.05), isfluid=True))
geom.add_elem(pyLBM.Circle((0.15, 0.85), 0.05, isfluid=False))
geom.add_elem(pyLBM.Parallelogram((0.1, 0.1), (0.05, 0), (0, 0.05), isfluid=True))
geom.add_elem(pyLBM.Circle((0.15, 0.15), 0.05, isfluid=False))
geom.add_elem(pyLBM.Parallelogram((0.9, 0.9), (-0.05, 0), (0, -0.05), isfluid=True))
geom.add_elem(pyLBM.Circle((0.85, 0.85), 0.05, isfluid=False))
geom.add_elem(pyLBM.Parallelogram((0.9, 0.1), (-0.05, 0), (0, 0.05), isfluid=True))
geom.add_elem(pyLBM.Circle((0.85, 0.15), 0.05, isfluid=False))
geom.visualize()
