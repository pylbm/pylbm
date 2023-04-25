# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 3D geometry with a STL file
"""
import pylbm

# pylint: disable=invalid-name

a, b = -1.5, 1.5

sponge = pylbm.STLElement("Menger_sponge_sample.stl", label=1)
dico = {
    "box": {"x": [a, b], "y": [a, b], "z": [a, b], "label": 0},
    "elements": [sponge],
}
geom = pylbm.Geometry(dico)
print(geom)
geom.visualize(viewlabel=True, alpha=0.25)
