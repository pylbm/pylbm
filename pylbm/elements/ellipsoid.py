# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Ellipsoid element
"""

# pylint: disable=invalid-name

import logging
# from textwrap import dedent
import numpy as np

from .base import Element
from .utils import distance_ellipsoid

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Ellipsoid(Element):
    """
    Class Ellipsoid

    Parameters
    ----------
    center : list
        the three coordinates of the center
    v1 : list
        a vector
    v2 : list
        a vector
    v3 : list
        a vector (v1, v2, and v3 have to be orthogonal)
    label : list
        one integer (default [0])
    isfluid : boolean
        - True if the ellipsoid is added
        - False if the ellipsoid is deleted

    Attributes
    ----------
    number_of_bounds : int
        1
    dim: int
        3
    center : ndarray
        the coordinates of the center of the sphere
    v1 : ndarray
        the coordinates of the first vector
    v2 : ndarray
        the coordinates of the second vector
    v3 : ndarray
        the coordinates of the third vector
    label : list
        the list of the label of the edge
    isfluid : boolean
        True if the ellipsoid is added
        and False if the ellipsoid is deleted

    Examples
    --------

    the ellipsoid centered in (0, 0, 0)
    with v1=[3,0,0], v2=[0,2,0], and v3=[0,0,1]

    >>> center = [0., 0., 0.]
    >>> v1, v2, v3 = [3,0,0], [0,2,0], [0,0,1]
    >>> Ellipsoid(center, v1, v2, v3)
    +-----------+
    | Ellipsoid |
    +-----------+
        - dimension: 3
        - center: [0. 0. 0.]
        - v1: [3 0 0]
        - v2: [0 2 0]
        - v3: [0 0 1]
        - label: [0]
        - type: solid

    """
    def __init__(self, center, v1, v2, v3, label=0, isfluid=False):
        self.number_of_bounds = 1  # number of edges
        self.dim = 3
        self.center = np.asarray(center)
        p12 = abs(v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2])
        p23 = abs(v2[0]*v3[0] + v2[1]*v3[1] + v2[2]*v3[2])
        p31 = abs(v3[0]*v1[0] + v3[1]*v1[1] + v3[2]*v1[2])
        if max(p12, p23, p31) > 1.e-14:
            log.error('The vectors of the ellipsoid are not orthogonal')
        else:
            self.v1 = np.asarray(v1)
            self.v2 = np.asarray(v2)
            self.v3 = np.asarray(v3)
        super(Ellipsoid, self).__init__(label, isfluid)
        log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the ellipsoid.
        """
        r = max(np.linalg.norm(self.v1),
                np.linalg.norm(self.v2),
                np.linalg.norm(self.v3))
        return self.center - r, self.center + r

    # pylint: disable=too-many-locals
    def point_inside(self, grid):
        """
        return a boolean array which defines
        if a point is inside or outside of the ellipsoid.

        Notes
        -----

        the edge of the ellipsoid is considered as inside.

        Parameters
        ----------

        grid : ndarray
            coordinates of the points

        Returns
        -------

        ndarray
            Array of boolean (True inside the ellipsoid, False otherwise)

        """
        x, y, z = grid

        X = x - self.center[0]
        Y = y - self.center[1]
        Z = z - self.center[2]
        v12 = np.cross(self.v1, self.v2)
        v23 = np.cross(self.v2, self.v3)
        v31 = np.cross(self.v3, self.v1)
        d = np.inner(self.v1, v23)**2
        # equation of the ellipsoid:
        # cxx XX + cyy YY + czz ZZ + cxy XY + cyz YZ + czx ZX = d
        cxx = v12[0]**2 + v23[0]**2 + v31[0]**2
        cyy = v12[1]**2 + v23[1]**2 + v31[1]**2
        czz = v12[2]**2 + v23[2]**2 + v31[2]**2
        cxy = 2 * (v12[0]*v12[1] + v23[0]*v23[1] + v31[0]*v31[1])
        cyz = 2 * (v12[1]*v12[2] + v23[1]*v23[2] + v31[1]*v31[2])
        czx = 2 * (v12[2]*v12[0] + v23[2]*v23[0] + v31[2]*v31[0])
        return cxx*X**2 + cyy*Y**2 + czz*Z**2 + \
            cxy*X*Y + cyz*Y*Z + czx*Z*X <= d

    def distance(self, grid, v, dmax=None, normal=False):
        """
        Compute the distance in the v direction between
        the ellipsoid and the points defined by (x, y, z).

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
        return distance_ellipsoid(
            x, y, z, v,
            self.center, self.v1, self.v2, self.v3,
            dmax, self.label, normal
        )

    def __str__(self):
        from ..utils import header_string
        from ..jinja_env import env
        template = env.get_template('ellipsoid.tpl')
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
        v1 = scale*self.v1
        v2 = scale*self.v2
        v3 = scale*self.v3
        viewer.ellipse_3d(self.center*scale, v1, v2, v3, color[0], alpha=alpha)
        if viewlabel:
            x, y, z = self.center[0], self.center[1], self.center[2]
            viewer.text(str(self.label[0]), [x, y, z])
