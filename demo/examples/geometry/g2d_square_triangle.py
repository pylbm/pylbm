

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 2D geometry: the square [0,1]x[0,1] cut with a triangle
"""
import pylbm

# pylint: disable=invalid-name

dgeom = {
    'box': {'x': [0, 1], 'y': [0, 1], 'label': 0},
    'elements': [
        pylbm.Triangle((0., 0.), (0., .5), (.5, 0.), label=1)
    ],
}
geom = pylbm.Geometry(dgeom)
print(geom)
geom.visualize(viewlabel=True)
