# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 9 velocities scheme in 2D
"""
import pyLBM
dsten = {
    'dim':2,
    'schemes':[{'velocities':range(9)}],
}
s = pyLBM.Stencil(dsten)
print s
s.visualize()
