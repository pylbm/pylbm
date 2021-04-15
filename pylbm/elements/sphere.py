# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Sphere element
"""

# pylint: disable=invalid-name

import logging
# from textwrap import dedent
import numpy as np

from .base import Element
from .utils import distance_ellipsoid

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Sphere(Element):
    """
    Class Sphere

    Parameters
    ----------

    center : list
        the three coordinates of the center
    radius : real
        a positive real number for the radius
    label : list
        one integer (default [0])
    isfluid : boolean
        - True if the sphere is added
        - False if the sphere is deleted

    Attributes
    ----------

    number_of_bounds : int
        1
    dim: int
        3
    center : ndarray
        the coordinates of the center of the sphere
    radius : real
        a positive real number for the radius of the sphere
    label : list
        the list of the label of the edge
    isfluid : boolean
        True if the sphere is added
        and False if the sphere is deleted

    Examples
    --------

    the sphere centered in (0, 0, 0) with radius 1

    >>> center = [0., 0., 0.]
    >>> radius = 1.
    >>> Sphere(center, radius)
    +--------+
    | Sphere |
    +--------+
        - dimension: 3
        - center: [0. 0. 0.]
        - radius: 1.0
        - label: [0]
        - type: solid

    """
    def __init__(self, center, radius, label=0, isfluid=False):
        self.number_of_bounds = 1  # number of edges
        self.dim = 3
        self.center = np.asarray(center)
        if radius >= 0:
            self.radius = radius
        else:
            log.error('The radius of the sphere should be positive')
        super(Sphere, self).__init__(label, isfluid)
        log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the sphere.
        """
        return self.center - self.radius, self.center + self.radius

    def point_inside(self, grid):
        """
        return a boolean array which defines
        if a point is inside or outside of the sphere.

        Notes
        -----

        the edge of the sphere is considered as inside.

        Parameters
        ----------

        grid : ndarray
            coordinates of the points

        Returns
        -------

        ndarray
            Array of boolean (True inside the sphere, False otherwise)

        """
        x, y, z = grid
        v2 = np.asarray([
            x - self.center[0],
            y - self.center[1],
            z - self.center[2]
        ], dtype=object)
        return (v2[0]**2 + v2[1]**2 + v2[2]**2) <= self.radius**2

    def distance(self, grid, v, dmax=None, normal=False):
        """
        Compute the distance in the v direction between
        the sphere and the points defined by (x, y, z).

        .. image:: ../figures/Sphere.png
            :width: 100%

        Parameters
        ----------

        grid : ndarray
            coordinates of the points
        v : ndarray
            direction of interest
        dmax : float
            distance max
        normal : bool
            return the normal vector if True (default False)

        Returns
        -------

        ndarray
            array of distances if normal is False and
            the coordinates of the normal vectors
            if normal is True
        """
        x, y, z = grid
        v1 = self.radius*np.array([1, 0, 0])
        v2 = self.radius*np.array([0, 1, 0])
        v3 = self.radius*np.array([0, 0, 1])
        return distance_ellipsoid(
            x, y, z, v, self.center,
            v1, v2, v3, dmax, self.label, normal
        )

    def __str__(self):
        from ..utils import header_string
        from ..jinja_env import env
        template = env.get_template('circle.tpl')
        elem_type = 'fluid' if self.isfluid else 'solid'
        return template.render(
            header=header_string(self.__class__.__name__),
            elem=self, type=elem_type
        )

    def visualize(self,
                  viewer, color, viewlabel=False,
                  scale=np.ones(3), alpha=1.
                  ):
        if not isinstance(color, list):
            color = [color]
        v1 = self.radius*np.array([1, 0, 0])*scale
        v2 = self.radius*np.array([0, 1, 0])*scale
        v3 = self.radius*np.array([0, 0, 1])*scale
        viewer.ellipse_3d(self.center*scale, v1, v2, v3, color[0], alpha=alpha)
        if viewlabel:
            x, y, z = self.center[0], self.center[1], self.center[2]
            viewer.text(str(self.label[0]), [x, y, z])
