# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Example of a 3 velocities scheme in 1D
"""
import pyLBM
dsten = {
    'dim':1,
    'schemes':[{'velocities':range(3)},],
}
s = pyLBM.Stencil(dsten)
print s
s.visualize()
