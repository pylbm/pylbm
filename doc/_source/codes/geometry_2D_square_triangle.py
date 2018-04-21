# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 2D geometry: the square [0,1]x[0,1] with a triangular hole
"""
import pylbm
d = {'box':{'x': [0, 1], 'y': [0, 1], 'label':0},
      'elements':[pylbm.Triangle((0.,0.), (0.,.5), (.5, 0.), label = 1)],
}
g = pylbm.Geometry(d)
g.visualize(viewlabel=True)
