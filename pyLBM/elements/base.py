# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np

from ..logs import setLogger

class Element:
    """
    Class Element

    generic class for the elements
    """
    number_of_bounds = 0
    def __init__(self):
        self.log = setLogger(__name__)
        self.isfluid = False
        self.label = []
        self.log.info(self.__str__())

    def get_bounds(self):
        return float('Inf'), -float('Inf')

    def point_inside(self, x, y, z=None):
        if z is None:
            return x**2 + y**2 < -1.
        else:
            return x**2 + y**2 + z**2 < -1.

    def __str__(self):
        return 'Generic element'

    def __repr__(self):
        return self.__str__()

    def _visualize(self, viewer, color, viewlabel=False, scale=np.ones(2)):
        pass

    def test_label(self):
        return len(self.label) == self.number_of_bounds
