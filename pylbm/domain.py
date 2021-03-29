# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause
"""
Domain definitions for LBM
"""
import logging
import sys
import copy
import numpy as np
import mpi4py.MPI as mpi

from .geometry import Geometry
from .stencil import Stencil
from .mpi_topology import MpiTopology
from .validator import validate
from . import viewer
from .utils import hsl_to_rgb

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _create_view_something(something, unvtot):
    """
    The user can pass in parameters of the visualize function
    a boolean or a list or a tuple of integers

    This function returns
    a list (eventually empty) of the considered velocities

    Examples
    --------

    > _create_view_something(True, 3)
        [0, 1, 2]

    > _create_view_something(False, 3)
        []

    > _create_view_something(1, 3)
        [1]

    > _create_view_something([0, 2, 3], 3)
        [0, 2]

    """
    if isinstance(something, bool):
        if something:
            return list(range(unvtot))
        else:
            return []
    elif isinstance(something, int):
        # pylint: disable=chained-comparison
        if something >= 0 and something < unvtot:
            return [something, ]
        else:
            return []
    elif isinstance(something, (list, tuple)):
        lst_something = []
        for s_k in something:
            # pylint: disable=chained-comparison
            if s_k >= 0 and s_k < unvtot:
                lst_something.append(s_k)
        return lst_something
    else:
        s = "Error in visualize (domain): \n"
        s += "optional parameter view_distance and view_bound should be\n"
        s += "  boolean, integer, list or tuple\n"
        log.error(s)
        return []


def _fix_color(vk):
    """
    fix the color of the plot
    for a given velocity
    """
    return hsl_to_rgb(
        np.sin(vk[0]+17)*np.cos(vk[1]+18)*np.cos(vk[2]+12),
        1, 0.5
    )


class Domain:
    """
    Create a domain that defines the fluid part and the solid part
    and computes the distances between these two states.

    Parameters
    ----------

    dico : dictionary
        that contains the following `key:value`

            - box : a dictionary that defines the computational box
            - elements : the list of the elements
                (available elements are given in the module
                :ref:`elements <mod_elements>`)
            - space_step : the spatial step
            - schemes : a list of dictionaries,

        each of them defining a elementary
        :py:class:`Scheme <pylbm.Scheme>`
        we only need the velocities to define a domain

    need_validation : bool
        boolean to specify if the dictionary has to be validated (optional)

    Notes
    -----

    The dictionary that defines the box should contains
    the following `key:value`

    - x : a list of the bounds in the first direction
    - y : a list of the bounds in the second direction (optional)
    - z : a list of the bounds in the third direction (optional)
    - label : an integer or a list of integers
      (length twice the number of dimensions)
      used to label each edge (optional)

    See :py:class:`Geometry <pylbm.geometry.Geometry>` for more details.

    In 1D, distance[q, i] is the distance between the point x[i]
    and the border in the direction of the qth velocity.

    In 2D, distance[q, j, i] is the distance between the point
    (x[i], y[j]) and the border in the direction of qth
    velocity

    In 3D, distance[q, k, j, i] is the distance between the point
    (x[i], y[j], z[k]) and the border in the direction of qth
    velocity

    In 1D, flag[q, i] is the flag of the border reached by the point
    x[i] in the direction of the qth velocity

    In 2D, flag[q, j, i] is the flag of the border reached by the point
    (x[i], y[j]) in the direction of qth velocity

    In 2D, flag[q, k, j, i] is the flag of the border reached by the point
    (x[i], y[j], z[k]) in the direction of qth velocity

    Warnings
    --------

    the sizes of the box must be a multiple of the space step dx

    Attributes
    ----------

    dim : int
      number of spatial dimensions (example: 1, 2, or 3)
    globalbounds : ndarray
      the bounds of the box in each spatial direction
    bounds : ndarray
      the local bounds of the process in each spatial direction
    dx : double
      space step (example: 0.1, 1.e-3)
    type : string
      type of data (example: 'float64')
    stencil : Stencil
      the stencil of the velocities (object of the class
      :py:class:`Stencil <pylbm.stencil.Stencil>`)
    global_size : list
      number of points in each direction
    extent : list
      number of points to add on each side (max velocities)
    coords : ndarray
      coordinates of the domain
    in_or_out : ndarray
      defines the fluid and the solid part
      (fluid: value=valin, solid: value=valout)
    distance : ndarray
      defines the distances to the borders.
      The distance is scaled by dx and is not equal to valin only for
      the points that reach the border with the specified velocity.
      shape = (number_of_velocities, nx, ny, nz)
    flag : ndarray
      NumPy array that defines the flag of the border reached with the
      specified velocity
    normal : ndarray
      numpy array containing the normal vector at the boundary points
      reached with the specified velocity.
      shape = (number_of_velocities, nx, ny, nz, dim)
    valin : int
        value in the fluid domain
    valout : int
        value in the solid domain

    Examples
    --------

    >>> dico = {'box': {'x': [0, 1], 'label': 0},
    ...         'space_step': 0.1,
    ...         'schemes': [{'velocities': list(range(3))}],
    ...        }
    >>> dom = Domain(dico)
    >>> dom
    +--------------------+
    | Domain information |
    +--------------------+
        - spatial dimension: 1
        - space step: 0.1
        - with halo:
            bounds of the box: [-0.05] x [1.05]
            number of points: [12]
        - without halo:
            bounds of the box: [0.05] x [0.95]
            number of points: [10]
    <BLANKLINE>
        +----------------------+
        | Geometry information |
        +----------------------+
            - spatial dimension: 1
            - bounds of the box: [0. 1.]

    >>> dico = {'box': {'x': [0, 1], 'y': [0, 1], 'label': [0, 0, 1, 1]},
    ...         'space_step': 0.1,
    ...         'schemes': [{'velocities': list(range(9))},
    ...                     {'velocities': list(range(5))}
    ...                    ],
    ...        }
    >>> dom = Domain(dico)
    >>> dom
    +--------------------+
    | Domain information |
    +--------------------+
        - spatial dimension: 2
        - space step: 0.1
        - with halo:
            bounds of the box: [-0.05 -0.05] x [1.05 1.05]
            number of points: [12, 12]
        - without halo:
            bounds of the box: [0.05 0.05] x [0.95 0.95]
            number of points: [10, 10]
    <BLANKLINE>
        +----------------------+
        | Geometry information |
        +----------------------+
            - spatial dimension: 2
            - bounds of the box: [0. 1.] x [0. 1.]

    see demo/examples/domain/


    """

    def __init__(self, dico, need_validation=True):
        self.valin = 999  # value in the fluid domain
        self.valout = -1   # value in the solid domain

        if dico is not None and need_validation:
            # pylint: disable=undefined-variable
            validate(dico, __class__.__name__)

        self.geom = Geometry(dico, need_validation=False)
        self.stencil = Stencil(dico, need_validation=False)
        self.dx = dico['space_step']
        self.dim = self.geom.dim
        self.compute_normal = True

        self.box_label = copy.copy(self.geom.box_label)

        self.mpi_topo = None
        self.construct_mpi_topology(dico)

        self.global_size = []
        self.create_coords()

        # pylint: disable=no-value-for-parameter
        region = self.mpi_topo.get_region(*self.global_size)

        # Modify box_label if the border becomes an interface
        for i in range(self.dim):
            if region[i][0] != 0:
                self.box_label[2*i] = -2
            if region[i][1] != self.global_size[i]:
                self.box_label[2*i + 1] = -2

        # distance to the borders
        total_size = [self.stencil.unvtot] + self.shape_halo
        vect_total_size = [self.stencil.unvtot] + self.shape_halo + [self.dim]
        self.in_or_out = self.valin*np.ones(self.shape_halo)
        self.distance = self.valin*np.ones(total_size)
        self.flag = self.valin*np.ones(total_size, dtype='int')
        if self.compute_normal:
            self.normal = np.zeros(vect_total_size)
        else:
            self.normal = None

        # compute the distance and the flag for the primary box
        self.__add_init(self.box_label)
        for elem in self.geom.list_elem:
            # treat each element of the geometry
            self.__add_elem(elem)

        self.clean()
        log.info(self.__str__())

    @property
    def shape_halo(self):
        """
        shape of the whole domain with the halo points.
        """
        return [c.size for c in self.coords_halo]

    @property
    def shape_in(self):
        """
        shape of the interior domain.
        """
        return [c.size for c in self.coords]

    @property
    def x(self):
        """
        x component of the coordinates in the interior domain.
        """
        return self.coords[0]

    @property
    def y(self):
        """
        y component of the coordinates in the interior domain.
        """
        return self.coords[1]

    @property
    def z(self):
        """
        z component of the coordinates in the interior domain.
        """
        return self.coords[2]

    @property
    def x_halo(self):
        """
        x component of the coordinates of the whole domain
        (halo points included).
        """
        return self.coords_halo[0]

    @property
    def y_halo(self):
        """
        y component of the coordinates of the whole domain
        (halo points included).
        """
        return self.coords_halo[1]

    @property
    def z_halo(self):
        """
        z component of the coordinates of the whole domain
        (halo points included).
        """
        return self.coords_halo[2]

    def __str__(self):
        from .utils import header_string
        from .jinja_env import env
        template = env.get_template('domain.tpl')
        return template.render(
            header=header_string('Domain information'),
            dom=self
        )

    def __repr__(self):
        return self.__str__()

    def construct_mpi_topology(self, dico):
        """
        Create the mpi topology
        """
        period = [True]*self.dim

        if dico is None:
            comm = mpi.COMM_WORLD
        else:
            comm = dico.get('comm', mpi.COMM_WORLD)
        self.mpi_topo = MpiTopology(self.dim, period, comm)

    def create_coords(self):
        """
        Create the coordinates of the interior domain and the whole domain
        with halo points.
        """
        phys_box = self.geom.bounds  # the physical box where the domain lies

        # validation of the space step with the physical box size
        for k in range(self.dim):
            self.global_size.append((phys_box[k][1] - phys_box[k][0])/self.dx)
            if not self.global_size[-1].is_integer():
                dummy = round(self.global_size[-1])
                diff_n = dummy - self.global_size[-1]
                if abs(diff_n) < self.dx:
                    message = "The length of the box "
                    message += "in the direction {0:d} ".format(k)
                    message += "is not exactly a multiple of the space step\n"
                    message += "The error in the length of the domain "
                    message += "in the direction {0:d} ".format(k)
                    message += "is {0:10.3e}".format(diff_n*self.dx)
                    log.info(message)
                    self.global_size[-1] = dummy
                else:
                    message = "The length of the box "
                    message += "in the direction {0:d} ".format(k)
                    message += "must be a multiple of the space step\n"
                    message += "The number of points is "
                    message += "{0:.15f}".format(self.global_size[-1])
                    log.error(message)
                    sys.exit()

        # we now are sure that global_size item are integers
        self.global_size = np.asarray(self.global_size, dtype='int')
        # pylint: disable=no-value-for-parameter
        region = self.mpi_topo.get_region(*self.global_size)
        region_size = [r[1] - r[0] for r in region]

        # spatial mesh
        halo_size = np.asarray(self.stencil.vmax)
        halo_beg = self.dx*(halo_size - 0.5)

        self.coords_halo = [
            np.linspace(
                phys_box[k][0] + self.dx*region[k][0] - halo_beg[k],
                phys_box[k][0] + self.dx*region[k][1] + halo_beg[k],
                region_size[k] + 2*halo_size[k]
            )
            for k in range(self.dim)
        ]
        # modification to avoid the case
        # with no velocity in one direction
        ind_beg = [k for k in halo_size]
        ind_end = [-k if k > 0 else 1 for k in halo_size]
        self.coords = [
            # self.coords_halo[k][halo_size[k]:-halo_size[k]]
            self.coords_halo[k][ind_beg[k]:ind_end[k]]
            for k in range(self.dim)
        ]

    def get_bounds_halo(self):
        """
        Return the coordinates of the bottom right and upper left corner of the
        whole domain with halo points.
        """
        bottom_right = np.asarray(
            [self.coords_halo[k][0] for k in range(self.dim)]
        )
        upper_left = np.asarray(
            [self.coords_halo[k][-1] for k in range(self.dim)]
        )
        return bottom_right, upper_left

    def get_bounds(self):
        """
        Return the coordinates of the bottom right and upper left corner of the
        interior domain.
        """
        bottom_right = np.asarray([self.coords[k][0] for k in range(self.dim)])
        upper_left = np.asarray([self.coords[k][-1] for k in range(self.dim)])
        return bottom_right, upper_left

    # pylint: disable=too-many-locals
    def __add_init(self, label):
        halo_size = np.asarray(self.stencil.vmax)
        phys_domain = [
            slice(h, -h) if h > 0 else slice(None)
            for h in halo_size
        ]
        phys_domain_vect = [
            slice(h, -h) if h > 0 else slice(None)
            for h in halo_size
        ]
        self.in_or_out[:] = self.valout
        in_view = self.in_or_out[tuple(phys_domain)]
        in_view[:] = self.valin

        # first index: the velocity index
        # following indices: the physical points
        # last index (for vectors): the dimension
        phys_domain.insert(0, slice(None))
        phys_domain_vect.insert(0, slice(None))
        phys_domain_vect.insert(len(phys_domain_vect), slice(None))
        dist_view = self.distance[tuple(phys_domain)]
        flag_view = self.flag[tuple(phys_domain)]
        if self.compute_normal:
            norm_view = self.normal[tuple(phys_domain_vect)]

        def new_indices(dvik, iuv, indices, dist_view):
            new_ind = copy.deepcopy(indices)
            ind = np.where(dist_view[tuple(indices)] > dvik)
            i = 1
            for j in range(self.dim):
                if j != iuv:
                    new_ind[j + 1] = ind[i]
                    i += 1
            return new_ind

        s = self.stencil
        uvels = [s.uvx, s.uvy, s.uvz]
        # loop over the dimension
        for iuvel, uvel in enumerate(uvels[:self.dim]):
            # uvel is the list of all the velocities
            # in the x-direction, then y-direction and z-direction
            # loop over the velocities
            for k, vk in np.ndenumerate(uvel):
                indices = [k] + [slice(None)]*self.dim
                if vk < 0 and label[2*iuvel] != -2:
                    for i in range(-vk):
                        indices[iuvel + 1] = i
                        dvik = -(i + .5)/vk
                        nind = new_indices(
                            dvik, iuvel,
                            indices,
                            dist_view
                        )
                        dist_view[tuple(nind)] = dvik
                        flag_view[tuple(nind)] = label[2*iuvel]
                        if self.compute_normal:
                            norm_view[tuple(nind + [slice(0, iuvel)])] = 0
                            norm_view[tuple(nind + [iuvel])] = -1
                elif vk > 0 and label[2*iuvel + 1] != -2:
                    for i in range(vk):
                        indices[iuvel + 1] = -i - 1
                        dvik = (i + .5)/vk
                        nind = new_indices(
                            dvik, iuvel,
                            indices,
                            dist_view
                        )
                        dist_view[tuple(nind)] = dvik
                        flag_view[tuple(nind)] = label[2*iuvel+1]
                        if self.compute_normal:
                            norm_view[tuple(nind + [slice(0, iuvel)])] = 0
                            norm_view[tuple(nind + [iuvel])] = 1

    # pylint: disable=too-many-locals
    def __add_elem(self, elem):
        """
        Add an element

            - if elem.isfluid = False as a solid part. (bw=0)
            - if elem.isfluid = True as a fluid part.  (bw=1)

        FIX: this function works only for a 2D problem.
             Need to be improved and implement for the 3D.
        """
        # compute the box around the element adding vmax safety points
        vmax = self.stencil.vmax
        elem_bl, elem_ur = elem.get_bounds()
        phys_bl, _ = self.get_bounds_halo()
        tmp = np.array((elem_bl - phys_bl)/self.dx, int) - vmax
        nmin = np.maximum(vmax, tmp)
        tmp = np.array((elem_ur - phys_bl)/self.dx, int) + vmax + 1
        nmax = np.minimum(vmax + self.shape_in, tmp)

        # set the grid
        space_slice = [slice(imin, imax) for imin, imax in zip(nmin, nmax)]
        total_slice = [slice(None)] + space_slice
        total_slice_vect = [slice(None)] + space_slice + [slice(None)]
        # local view of the arrays
        ioo_view = self.in_or_out[tuple(space_slice)]
        dist_view = self.distance[tuple(total_slice)]
        flag_view = self.flag[tuple(total_slice)]
        if self.compute_normal:
            norm_view = self.normal[tuple(total_slice_vect)]

        tcoords = (self.coords_halo[d][s] for d, s in enumerate(space_slice))
        grid = np.meshgrid(*tcoords, sparse=True, indexing='ij')

        if not elem.isfluid:  # add a solid part
            ind_solid = elem.point_inside(grid)
            ind_fluid = np.logical_not(ind_solid)
            ioo_view[ind_solid] = self.valout
        else:  # add a fluid part
            ind_fluid = elem.point_inside(grid)
            ind_solid = np.logical_not(ind_fluid)
            ioo_view[ind_fluid] = self.valin

        for k in range(self.stencil.unvtot):
            vk = np.asarray(self.stencil.unique_velocities[k].v)
            if np.any(vk != 0):
                space_slice = [
                    slice(imin + vk[d], imax + vk[d])
                    for imin, imax, d in zip(nmin, nmax, range(self.dim))
                ]
                # check the cells that are out
                # when we move with the vk velocity
                out_cells = self.in_or_out[tuple(space_slice)] == self.valout
                # compute the distance and set the boundary label
                # of each cell and the element with the vk velocity
                alpha, border, normvect = elem.distance(
                    grid, self.dx*vk, 1.,
                    self.compute_normal
                )
                # take the indices where the distance is lower than 1
                # between a fluid cell and the border of the element
                # with the vk velocity
                indx = np.logical_and(alpha > 0, ind_fluid)
                if out_cells.size != 0:
                    indx = np.logical_and(indx, out_cells)

                if elem.isfluid:
                    # take all points in the fluid in the ioo_view
                    indfluidinbox = ioo_view == self.valin
                    # take all the fluid points in the box
                    # (not only in the created element)
                    # which always are in fluid after a displacement
                    # of the velocity vk
                    border_to_interior = np.logical_and(
                        np.logical_not(out_cells), indfluidinbox
                    )
                    dist_view[k][border_to_interior] = self.valin
                    flag_view[k][border_to_interior] = self.valin
                else:
                    dist_view[k][ind_solid] = self.valin
                    flag_view[k][ind_solid] = self.valin

                # set distance
                ind4 = np.where(indx)
                if not elem.isfluid:
                    ind3 = np.where(alpha[ind4] < dist_view[k][ind4])[0]
                else:
                    ind3 = np.where(
                        np.logical_or(
                            alpha[ind4] > dist_view[k][ind4],
                            dist_view[k][ind4] == self.valin
                        )
                    )[0]

                ind = [i[ind3] for i in ind4]
                dist_view[k][tuple(ind)] = alpha[tuple(ind)]
                flag_view[k][tuple(ind)] = border[tuple(ind)]
                if self.compute_normal:
                    for i in range(self.dim):
                        norm_view[k][tuple(ind + [i])] = - normvect[
                            tuple(ind + [i])
                        ]

    def clean(self):
        """
        clean the domain when multiple elements are added
        some unused distances or normal vectors have been computed
        """
        # loop of the unique velocity
        # look for the outer points where the distance is computed
        # fix these points to full outer points
        for k in range(self.stencil.unvtot):
            indk = np.logical_and(
                self.distance[k] > 0,
                self.in_or_out == self.valout
            )
            self.distance[k][indk] = self.valin
            self.flag[k][indk] = self.valin
            if self.compute_normal:
                self.normal[k][indk] = 0

    def list_of_labels(self):
        """
        Get the list of all the labels used in the geometry.
        """
        labels = np.unique(self.box_label)
        return np.union1d(labels, self.geom.list_of_elements_labels())

    # pylint: disable=too-complex
    def visualize(self,
                  viewer_app=viewer.matplotlib_viewer,
                  view_geom=True,
                  view_distance=False,
                  view_in=True,
                  view_out=True,
                  view_bound=False,
                  view_normal=False,
                  label=None,
                  scale=1):
        """
        Visualize the domain by creating a plot.

        Parameters
        ----------
        viewer_app : Viewer, optional
            define the viewer to plot the domain
            default is viewer.matplotlib_viewer
        view_geom : boolean
            view the underlying geometry
            default is True
        view_distance : boolean or int or list, optional
            view the distance between the interior points and the border
            default is False
            if True, then all velocities are considered
            can specify some specific velocities in a list
        view_in : boolean, optional
            view the inner points
            default is True
        view_out : boolean, optional
            view the outer points
            default is True
        view_bound : boolean or int or list, optional
            view the points on the bounds
            default is False
        view_normal : boolean or int or list, optional
            view the normal vectors
            default is False
        label : int or list, optional
            view the distance only for the specified labels
        scale : int or float, optional
            scale used for the symbol (default 1)

        Returns
        -------

        object
            views

        """

        fig = viewer_app.Fig(dim=self.dim)
        view = fig[0]
        view.title = "Domain"

        # fix the axis for all dimensions
        delta_l = .25 * max(
            np.diff(self.geom.bounds, axis=1).flatten()
        )  # small length to fix the empty area around the figure
        bornes = [*[-delta_l, delta_l]]*max(2, self.dim)
        bornes[:2*self.dim] += self.geom.bounds.flatten()
        view.axis(*bornes, dim=self.dim, aspect='equal')

        if view_geom and self.dim == 2:
            color_fluid = (0.9, 0.9, 0.9)
            xmin, xmax = self.geom.bounds[0][:]
            ymin, ymax = self.geom.bounds[1][:]
            view.polygon(
                np.array(
                    [
                        [xmin, ymin],
                        [xmin, ymax],
                        [xmax, ymax],
                        [xmax, ymin]
                    ]
                ),
                color_fluid, alpha=1
            )
            for elem in self.geom.list_elem:
                if elem.isfluid:
                    color = color_fluid
                else:
                    color = (1., 1., 1.)
                elem.visualize(
                    view, color,
                    alpha=1
                )
        if view_geom and self.dim == 3:
            color_fluid = (0.9, 0.9, 0.9)
            for elem in self.geom.list_elem:
                if elem.isfluid:
                    color = color_fluid
                else:
                    color = (.5, .5, .5)
                elem.visualize(
                    view, color,
                    alpha=0.25
                )

        # compute the size of the symbols for the plot
        # in 1d and 2d: 100*dx
        # in 3d: 10*dx*dx
        coeff = 1 + 9 * (self.dim <= 2)
        size = scale * 10 * coeff * self.dx**(1+(self.dim == 3))

        # view_distance : list of the concerned velocities
        view_distance = _create_view_something(
            view_distance, self.stencil.unvtot
        )
        # view_bound : list of the concerned velocities
        view_bound = _create_view_something(
            view_bound, self.stencil.unvtot
        )
        # view_normal : list of the concerned velocities
        view_normal = _create_view_something(
            view_normal, self.stencil.unvtot
        )

        # temporary boolean array for considering specific labels
        if isinstance(label, int):
            label = (label,)

        def get_inorout_points(val):
            # returns the coordinates where the value is equal to val
            ind = np.where(self.in_or_out == val)
            data = np.zeros((ind[0].size, 3))
            for i in range(self.dim):
                data[:, i] = self.coords_halo[i][ind[i]]
            return data

        # visualize the inner points
        if view_in:
            view.markers(
                get_inorout_points(self.valin),
                size, symbol='o', alpha=.25, color='navy',
                dim=self.dim
            )

        # visualize the outer points
        if view_out:
            view.markers(
                get_inorout_points(self.valout),
                size, symbol='s', alpha=.5, color='orange',
                dim=self.dim
            )

        def get_bounds(label, k, compute_normal=False):
            # returns
            # 1. the indices of the bounds
            #    for the kth velocity and for the given labels
            # 2. the corresponding distances
            # 3. the corresponding normal vectors
            #    if compute_normal is True
            if label is not None:
                dummy = np.zeros(self.distance.shape[1:])
                for labelk in label:
                    dummy += self.flag[k, :] == labelk
                dummy *= self.distance[k, :] <= 1
                indbord = np.where(dummy)
            else:
                indbord = np.where(self.distance[k, :] <= 1)
            if indbord:
                data = np.zeros((indbord[0].size, max(2, self.dim)))
                for i in range(self.dim):
                    data[:, i] = self.coords_halo[i][indbord[i]]
                dist = self.distance[k][indbord]
                if compute_normal:
                    normal = self.normal[k][indbord]
                    return data, dist, normal
                else:
                    return data, dist
            else:
                if compute_normal:
                    return None, 0, None
                else:
                    return None, 0

        # visualize the distance as small lines
        for k in view_distance:
            bound, dist = get_bounds(label, k)
            if bound.size != 0:
                vk = self.stencil.unique_velocities[k].v_full
                color = _fix_color(vk)
                # pylint: disable=unsubscriptable-object
                lines = np.empty((2*bound.shape[0], max(2, self.dim)))
                lines[::2, :] = bound
                lines[1::2, :] = bound \
                    + self.dx*np.outer(dist, vk[:max(2, self.dim)])
                view.segments(lines, alpha=0.5, width=2, color=color)

        # visualize the bounds as diamond
        for k in view_bound:
            bound, dist = get_bounds(label, k)
            if bound.size != 0:
                vk = self.stencil.unique_velocities[k].v_full
                color = np.array([[*_fix_color(vk)]])
                lines = bound + self.dx*np.outer(dist, vk[:max(2, self.dim)])
                view.markers(lines, size, symbol='d', color=color)

        # visualize the normal vectors as small lines
        for k in view_normal:
            bound, dist, normal = get_bounds(label, k, compute_normal=True)
            if bound.size != 0:
                vk = self.stencil.unique_velocities[k].v_full
                color = _fix_color(vk)
                # pylint: disable=unsubscriptable-object
                lines = np.empty((2*bound.shape[0], max(2, self.dim)))
                lines[::2, :] = bound \
                    + self.dx*np.outer(dist, vk[:max(2, self.dim)])
                lines[1::2, :self.dim] = lines[::2, :self.dim] \
                    + self.dx*normal
                view.segments(lines, alpha=0.5, width=1, color=color)

        fig.show()
        return fig
