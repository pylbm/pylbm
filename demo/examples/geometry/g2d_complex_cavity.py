# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a complex 2D geometry
"""
import pylbm

# pylint: disable=invalid-name

square = pylbm.Parallelogram((0.1, 0.1), (0.8, 0), (0, 0.8), isfluid=False)
strip = pylbm.Parallelogram((0, 0.4), (1, 0), (0, 0.2), isfluid=True)
circle = pylbm.Circle((0.5, 0.5), 0.25, isfluid=True)
inner_square = pylbm.Parallelogram((0.4, 0.5), (0.1, 0.1), (0.1, -0.1), isfluid=False)
dgeom = {
    "box": {"x": [0, 1], "y": [0, 1], "label": 0},
    "elements": [square, strip, circle, inner_square],
}
geom = pylbm.Geometry(dgeom)
# rounded inner angles
geom.add_elem(pylbm.Parallelogram((0.1, 0.9), (0.05, 0), (0, -0.05), isfluid=True))
geom.add_elem(pylbm.Circle((0.15, 0.85), 0.05, isfluid=False))
geom.add_elem(pylbm.Parallelogram((0.1, 0.1), (0.05, 0), (0, 0.05), isfluid=True))
geom.add_elem(pylbm.Circle((0.15, 0.15), 0.05, isfluid=False))
geom.add_elem(pylbm.Parallelogram((0.9, 0.9), (-0.05, 0), (0, -0.05), isfluid=True))
geom.add_elem(pylbm.Circle((0.85, 0.85), 0.05, isfluid=False))
geom.add_elem(pylbm.Parallelogram((0.9, 0.1), (-0.05, 0), (0, 0.05), isfluid=True))
geom.add_elem(pylbm.Circle((0.85, 0.15), 0.05, isfluid=False))
print(geom)
geom.visualize()
