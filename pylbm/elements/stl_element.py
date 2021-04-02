# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
STL element
https://en.wikipedia.org/wiki/STL_(file_format)
"""

# pylint: disable=invalid-name

from os.path import isfile
import logging
# from textwrap import dedent
import numpy as np
from stl import mesh

from .base import Element
# from .utils import distance_lines

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


def intersection(grid, v, tri):
    """
    We look for the intersection between
    - the lines with the direction v that passes by x, y, z
    - the triangle tri

    t>0, a>=0, b>=0, a+b<=1
    x + t vx = a tri[6] + b tri[3] + (1-a-b) tri[0]
    y + t vy = a tri[7] + b tri[4] + (1-a-b) tri[1]
    z + t vz = a tri[8] + b tri[5] + (1-a-b) tri[2]
    <=>
    a (tri[6]-tri[0]) + b (tri[3]-tri[0]) - t vx = x-tri[0]
    a (tri[7]-tri[1]) + b (tri[4]-tri[1]) - t vy = y-tri[1]
    a (tri[8]-tri[2]) + b (tri[5]-tri[2]) - t vz = z-tri[2]
    """
    # the matrix of the intersection problem
    # A is 3x3 and does not depend on the grid
    A = np.array(
        [
            [tri[6]-tri[0], tri[3]-tri[0], -v[0]],
            [tri[7]-tri[1], tri[4]-tri[1], -v[1]],
            [tri[8]-tri[2], tri[5]-tri[2], -v[2]]
        ]
    )
    # if the direction v is in the planed of the triangle tri
    # then the matrix A is not invertible
    # the intersection is not computed with that v but with another one
    try:
        invA = np.linalg.inv(A)
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' not in str(e):
            raise
        return None

    # right member of the intersection problem
    # it depends on the grid and is a (xsize, ysize, zsize) vector
    x, y, z = grid
    P = np.asarray(
        [x - tri[0], y - tri[1], z - tri[2]],
        dtype=object
    )
    # solve the problem
    sol = invA.dot(P)
    # test if the intersection point is
    # - inside the triangle: s is computed
    # - outside the triangle: s = np.inf
    indin = np.logical_and(
        np.logical_and(
            sol[0] >= 0, sol[1] >= 0
        ), np.logical_and(
            sol[0] + sol[1] <= 1, sol[2] >= 0
        )
    )
    s = np.full((x.size, y.size, z.size), np.inf)
    s[indin] = sol[2][indin]
    # # compute the coordinates of the intersection point
    # p = np.full((3, x.size, y.size, z.size), np.inf)
    # for k in range(3):
    #     p[k][indin] = tri[3*k] * sol[0][indin] + \
    #         tri[3*k+1] * sol[1][indin] + \
    #         tri[3*k+2] * (1-sol[0][indin]-sol[1][indin])

    return s


class STLElement(Element):
    """
    Class STLElement

    Notes
    --------
    Add a STLElement requires the module numpy-stl!

    Parameters
    ----------

    filename: string
        the file containing the STL element
    label: int
        the label on the element
    isfluid: boolean
        - True if the element is added
        - False if the element is deleted

    Examples
    --------

    Attributes
    ----------

    mesh: stl.mesh.Mesh
        the surface mesh of the element
    dim: int
        3
    """
    def __init__(self, filename, label=0, isfluid=False):
        self.filename = filename
        self.mesh = mesh.Mesh.from_file(filename)
        self.number_of_bounds = 1  # just one bound for the labels
        self.nb_tri = self.mesh.points.shape[0]  # nb of triangles
        self.dim = 3
        super(STLElement, self).__init__(label, isfluid)
        log.info(self.__str__())

    def get_bounds(self):
        return self.mesh.min_, self.mesh.max_

    def _center(self):
        return self.mesh.get_mass_properties()[1]

    def point_inside(self, grid):
        shape = tuple([gk.size for gk in grid])
        in_or_out = np.full(shape, False, dtype=bool)
        smin = np.full(shape, np.inf)

        # for each spatial direction, we plot a line
        # and we compute the distance of the first intersection
        # with the element in that direction.
        # if the normal is in the opposite side, the point is outside
        for direction in range(self.dim):
            smin[:] = np.inf
            v = np.zeros((self.dim,))
            v[direction] = 1
            for ind_tri in range(self.nb_tri):
                tri = self.mesh.points[ind_tri, :]
                s = intersection(grid, v, tri)
                if s is not None:
                    indices = np.asarray(s < smin).nonzero()
                    smin[indices] = s[indices]
                    if np.inner(v, self.mesh.normals[ind_tri]) >= 0:
                        in_or_out[indices] = True
                    else:
                        in_or_out[indices] = False
        return in_or_out

    def distance(self, grid, v, dmax=None, normal=False):
        shape = tuple([gk.size for gk in grid])
        alpha = np.full(shape, np.inf)
        border = np.full(shape, -1)
        if normal:
            normal_x = np.zeros(shape)
            normal_y = np.zeros(shape)
            normal_z = np.zeros(shape)
        # loop over the triangles
        for ind_tri in range(self.nb_tri):
            tri = self.mesh.points[ind_tri, :]
            s = intersection(grid, v, tri)
            if s is not None:
                if dmax is None:
                    indices = np.asarray(s < alpha).nonzero()
                else:
                    indices = np.logical_and(
                        s < alpha, s <= dmax
                    )
                alpha[indices] = s[indices]
                border[indices] = self.label[0]
                if normal:
                    normal_x[indices] = self.mesh.normals[ind_tri][0]
                    normal_y[indices] = self.mesh.normals[ind_tri][1]
                    normal_z[indices] = self.mesh.normals[ind_tri][2]
        if normal:
            normal = np.zeros(tuple(list(shape) + [3]))
            normal[..., 0] = normal_x
            normal[..., 1] = normal_y
            normal[..., 2] = normal_z
        return alpha, border, normal

    def __str__(self):
        from ..utils import header_string
        from ..jinja_env import env
        template = env.get_template('stl.tpl')
        elem_type = 'fluid' if self.isfluid else 'solid'
        elem_dim = np.asarray(self.get_bounds()).reshape((6,))
        return template.render(
            header=header_string(self.__class__.__name__),
            elem=self, type=elem_type, dim=elem_dim
        )

    def visualize(self,
                  viewer, color, viewlabel=False,
                  scale=np.ones(3), alpha=0.25,
                  ):
        # coordinates of the basis triangle
        p = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        ).T
        vv = np.array([0, 1])
        # meshgrid used for plot_surface
        Xb = np.outer(p[0], vv)
        Yb = np.outer(p[1], vv)
        Zb = np.outer(p[2], vv)

        points = self.mesh.points
        normals = self.mesh.normals
        for k in range(points.shape[0]):
            # plot the kth triangle
            tri = points[k]
            n = normals[k]
            # the coordinates of the three points
            x, y, z = tri[::3], tri[1::3], tri[2::3]
            # matrix of the change of basis
            A = np.array(
                [
                    [x[1]-x[0], x[2]-x[0], n[0]],
                    [y[1]-y[0], y[2]-y[0], n[1]],
                    [z[1]-z[0], z[2]-z[0], n[2]]
                ]
            )
            # write the coordinates of the triangle
            X = x[0] + A[0, 0]*Xb + A[0, 1]*Yb + A[0, 2]*Zb
            Y = y[0] + A[1, 0]*Xb + A[1, 1]*Yb + A[1, 2]*Zb
            Z = z[0] + A[2, 0]*Xb + A[2, 1]*Yb + A[2, 2]*Zb
            # plot
            viewer.surface(X, Y, Z, 'black', alpha=alpha)

        if viewlabel:
            x, y, z = self._center()
            viewer.text(str(self.label[0]), [x, y, z])
