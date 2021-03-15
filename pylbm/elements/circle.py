# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Circle element
"""

# pylint: disable=invalid-name

import logging
# from textwrap import dedent
import numpy as np

from .base import Element
from .utils import distance_ellipse

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Circle(Element):
    """
    Class Circle

    Parameters
    ----------
    center : list
        the two coordinates of the center
    radius : float
        the radius
    label : list
        default [0]
    isfluid : boolean
        - True if the circle is added
        - False if the circle is deleted

    Attributes
    ----------
    number_of_bounds : int
        1
    dimension: int
        2
    center : ndarray
        the coordinates of the center of the circle
    radius : double
        positive float for the radius of the circle
    label : list
        the list of the label of the edge
    isfluid : boolean
        True if the circle is added
        and False if the circle is deleted

    Examples
    --------

    the circle centered in (0, 0) with radius 1

    >>> center = [0., 0.]
    >>> radius = 1.
    >>> Circle(center, radius)
    +--------+
    | Circle |
    +--------+
        - dimension: 2
        - center: [0. 0.]
        - radius: 1.0
        - label: [0]
        - type: solid

    """
    def __init__(self, center, radius, label=0, isfluid=False):
        self.number_of_bounds = 1  # number of edges
        self.dim = 2
        self.center = np.asarray(center)
        if radius >= 0:
            self.radius = radius
        else:
            log.error('The radius of the circle should be positive')
        super(Circle, self).__init__(label, isfluid)
        log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the circle.
        """
        return self.center - self.radius, self.center + self.radius

    def point_inside(self, grid):
        """
        return a boolean array which defines
        if a point is inside or outside of the circle.

        Parameters
        ----------

        grid : ndarray
            coordinates of the points

        Returns
        -------

        ndarray
            Array of boolean (True inside the circle, False otherwise)

        Notes
        -----

        the edge of the circle is considered as inside.

        """
        x, y = grid
        v2 = np.asarray([x - self.center[0], y - self.center[1]], dtype=object)
        return (v2[0]**2 + v2[1]**2) <= self.radius**2

    def distance(self, grid, v, dmax=None, normal=False):
        """
        Compute the distance in the v direction between
        the circle and the points defined by (x, y).
        if normal==True, compute also the normal vector

        .. image:: ../figures/Circle.png
            :width: 100%

        Parameters
        ----------

        grid : ndarray
            coordinates of the points
        v : ndarray
            direction of interest
        dmax : float
            distance max (default None)
        normal : bool
            return the normal vector if True (default False)

        Returns
        -------

        ndarray
            array of distances if normal is False and
            the coordinates of the normal vectors
            if normal is True
        """
        x, y = grid
        v1 = self.radius*np.array([1, 0])
        v2 = self.radius*np.array([0, 1])
        return distance_ellipse(
            x, y, v, self.center, v1, v2,
            dmax, self.label, normal
        )

    def __str__(self):
        from ..utils import header_string
        from ..jinja_env import env
        template = env.get_template('circle.tpl')
        elem_type = 'fluid' if self.isfluid else 'solid'
        return template.render(
            header=header_string('Circle'),
            elem=self, type=elem_type
        )

    def visualize(self,
                  viewer, color, viewlabel=False,
                  scale=np.ones(2), alpha=1.
                  ):
        viewer.ellipse(self.center*scale,
                       tuple(self.radius*scale),
                       color,
                       alpha=alpha
                       )
        if viewlabel:
            theta = self.center[0] + 2*self.center[1]+10*self.radius
            x = self.center[0] + self.radius*np.cos(theta)
            y = self.center[1] + self.radius*np.sin(theta)
            viewer.text(str(self.label[0]), [x*scale[0], y*scale[1]])
