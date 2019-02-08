

# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 2D geometry: the backward facing step
"""
import pylbm

# pylint: disable=invalid-name

dgeom = {
    'box': {
        'x': [0, 3],
        'y': [0, 1],
        'label': [0, 1, 2, 3],
    },
    'elements': [
        pylbm.Parallelogram((0., 0.), (.5, 0.), (0., .5), label=[4, 5, 6, 7])
    ],
}
geom = pylbm.Geometry(dgeom)
print(geom)
geom.visualize(viewlabel=True)
