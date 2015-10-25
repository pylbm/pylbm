from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np
import sympy as sp
import sys
import copy
from six.moves import range
from six import string_types

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

from .elements import *
from .geometry import Geometry
from .stencil import Stencil
from .validate_dictionary import *
from .logs import setLogger
from . import viewer

proto_domain = {
    'box':(is_dico_box,),
    'elements':(type(None), is_list_elem),
    'dim':(type(None), int),
    'space_step':(int, float,),
    'scheme_velocity':(type(None), int, float, sp.Symbol),
    'parameters':(type(None), is_dico_sp_float),
    'schemes':(is_list_sch_dom,),
    'boundary_conditions':(type(None), is_dico_bc),
    'generator':(type(None), is_generator),
    'stability':(type(None), is_dico_stab),
    'consistency':(type(None), is_dico_cons),
    'inittype':(type(None),) + string_types,
}

class Domain(object):
    """
    Create a domain that defines the fluid part and the solid part
    and computes the distances between these two states.

    Parameters
    ----------

    dico : a dictionary that contains the following `key:value`

        - box : a dictionary that defines the computational box
        - elements : the list of the elements
          (available elements are given in the module :py:mod:`elements <pyLBM.elements>`)
        - space_step : the spatial step
        - schemes : a list of dictionaries,
          each of them defining a elementary :py:class:`Scheme <pyLBM.scheme.Scheme>`

    Notes
    -----

    The dictionary that defines the box should contains the following `key:value`

    - x : a list of the bounds in the first direction
    - y : a list of the bounds in the second direction (optional)
    - z : a list of the bounds in the third direction (optional)
    - label : an integer or a list of integers
      (length twice the number of dimensions)
      used to label each edge (optional)

    See :py:class:`Geometry <pyLBM.geometry.Geometry>` for more details.

    If the geometry and/or the stencil were previously generated,
    it can be used directly as following

    >>> Domain(dico, geometry = geom, stencil = sten)

    where geom is an object of the class
    :py:class:`Geometry <pyLBM.geometry.Geometry>`
    and sten an object of the class
    :py:class:`Stencil <pyLBM.stencil.Stencil>`
    In that case, dico does not need to contain the informations for generate
    the geometry and/or the stencil

    In 1D, distance[q, i] is the distance between the point x[0][i]
    and the border in the direction of the qth velocity.

    In 2D, distance[q, j, i] is the distance between the point
    (x[0][i], x[1][j]) and the border in the direction of qth
    velocity

    In 3D, distance[q, k, j, i] is the distance between the point
    (x[0][i], x[1][j], x[2][k]) and the border in the direction of qth
    velocity

    In 1D, flag[q, i] is the flag of the border reached by the point
    x[0][i] in the direction of the qth velocity

    In 2D, flag[q, j, i] is the flag of the border reached by the point
    (x[0][i], x[1][j]) in the direction of qth velocity

    In 2D, flag[q, k, j, i] is the flag of the border reached by the point
    (x[0][i], x[1][j], x[2][k]) in the direction of qth velocity

    Warnings
    --------

    the sizes of the box must be a multiple of the space step dx

    Attributes
    ----------

    dim : int
      number of spatial dimensions (example: 1, 2, or 3)
    globalbounds : numpy array
      the bounds of the box in each spatial direction
    bounds : numpy array
      the local bounds of the process in each spatial direction
    dx : double
      space step (example: 0.1, 1.e-3)
    type : string
      type of data (example: 'float64')
    stencil :
      the stencil of the velocities (object of the class
      :py:class:`Stencil <pyLBM.stencil.Stencil>`)
    N : list of int
      number of points in each direction
    extent : list of int
      number of points to add on each side (max velocities)
    x : numpy array
      coordinates of the domain
    in_or_out : numpy array
      defines the fluid and the solid part
      (fluid: value=valin, solid: value=valout)
    distance : numpy array
      defines the distances to the borders.
      The distance is scaled by dx and is not equal to valin only for
      the points that reach the border with the specified velocity.
    flag : numpy array
      NumPy array that defines the flag of the border reached with the
      specified velocity


    Methods
    -------

    visualize :
      Visualize the domain by creating a plot

    Examples
    --------

    see demo/examples/domain/


    """
    def __init__(self, dico=None, geometry=None, stencil=None, space_step=None, verif=True):
        self.log = setLogger(__name__)

        if dico is not None:
            self.log.info('Check the dictionary')
            test, aff = validate(dico, proto_domain, test_comp = False)
            if test:
                self.log.info(aff)
            else:
                self.log.error(aff)
                sys.exit()

        self.geom = Geometry(dico) if geometry is None else geometry
        self.stencil = Stencil(dico) if stencil is None else stencil
        self.dx = dico['space_step'] if space_step is None else space_step

        self.dim = self.geom.dim

        self.globalbounds = self.geom.globalbounds # the box where the domain lies
        self.bounds = self.geom.bounds # the local box of the process

        get_shape = lambda x: int((x[1] - x[0] + .5*self.dx)/self.dx)
        self.Ng = list(map(get_shape, self.globalbounds[:self.dim]))
        self.N = list(map(get_shape, self.bounds[:self.dim]))

        # spatial mesh
        self.extent = np.asarray(self.stencil.vmax[:self.dim])
        debord = self.dx*(self.extent - 0.5)
        Na = np.asarray(self.N) + 2*self.extent
        self.x = np.asarray([np.linspace(self.bounds[k][0] - debord[k],
                                         self.bounds[k][1] + debord[k],
                                         Na[k]) for k in range(self.dim)])

        for k in range(self.dim):
            extra_points = (self.globalbounds[k][1] - self.globalbounds[k][0])/self.dx
            if not extra_points.is_integer():
                self.log.error('The length of the box in the direction {0} must be a multiple of the space step'.format(k))

        # distance to the borders
        self.valin = 999  # value in the fluid domain
        self.valout = -1   # value in the solid domain

        s2 = np.concatenate(([self.stencil.unvtot], Na))
        self.in_or_out = self.valin*np.ones(Na)
        self.distance = self.valin*np.ones(s2)
        self.flag = self.valin*np.ones(s2, dtype = 'int')

        self.__add_init(self.geom.box_label) # compute the distance and the flag for the primary box
        for elem in self.geom.list_elem: # treat each element of the geometry
            self.__add_elem(elem)

        self.log.info(self.__str__())

    @property
    def shape(self):
        return [x.size for x in self.x]

    def __str__(self):
        s = "Domain informations\n"
        s += "\t spatial dimension: {0:d}\n".format(self.dim)
        s += "\t bounds of the box: bounds = " + self.bounds.__str__() + "\n"
        s += "\t space step: dx={0:10.3e}\n".format(self.dx)
        #s += "\t Number of points in each direction: N=" + self.N.__str__() + ", Na=" + self.Na.__str__() + "\n"
        return s

    def __add_init(self, label):
        phys_domain = [slice(e, e + n) for e, n in zip(self.extent, self.N)]

        self.in_or_out[:] = self.valout

        in_view = self.in_or_out[phys_domain]
        in_view[:] = self.valin

        phys_domain.insert(0, slice(None))
        dist_view = self.distance[phys_domain]
        flag_view = self.flag[phys_domain]

        def new_indices(dvik, iuv, indices, dist_view):
            new_ind = copy.deepcopy(indices)
            ind = np.where(dist_view[indices] > dvik)
            ii = 1
            for j in range(self.dim):
                if j != iuv:
                    new_ind[j + 1] = ind[ii]
                    ii += 1
            return new_ind

        s = self.stencil
        uvel = [s.uvx, s.uvy, s.uvz]

        for iuv, uv in enumerate(uvel[:self.dim]):
            for k, vk in np.ndenumerate(uv):
                indices = [k] + [slice(None)]*self.dim
                if vk < 0 and label[2*iuv] != -2:
                    for i in range(-vk):
                        indices[iuv + 1] = i
                        dvik = -(i + .5)/vk
                        nind = new_indices(dvik, iuv, indices, dist_view)
                        dist_view[nind] = dvik
                        flag_view[nind] = label[2*iuv]
                elif vk > 0 and label[2*iuv + 1] != -2:
                    for i in range(vk):
                        indices[iuv + 1] = -i -1
                        dvik = (i + .5)/vk
                        nind = new_indices(dvik, iuv, indices, dist_view)
                        dist_view[nind] = dvik
                        flag_view[nind] = label[2*iuv+1]


    def __add_elem(self, elem):
        """
        Add an element

            - if elem.isfluid = False as a solid part. (bw=0)
            - if elem.isfluid = True as a fluid part.  (bw=1)

        FIX: this function works only for a 2D problem.
             Need to be improved and implement for the 3D.
        """
        # compute the box around the element adding vmax safety points
        indbe = np.asarray([(self.stencil.vmax[k],
                             self.stencil.vmax[k] + self.N[k]) for k in range(self.dim)])
        bmin, bmax = elem.get_bounds()

        #if self.dim < 3:
        #    xbeg = np.asarray([self.x[0][0], self.x[1][0]])
        #else:
        #    xbeg = np.asarray([self.x[0][0], self.x[1][0], self.x[2][0]])
        xbeg = np.asarray([self.x[k][0] for k in range(self.dim)])

        tmp = np.array((bmin - xbeg)/self.dx - self.stencil.vmax[:self.dim], np.int)
        nmin = np.maximum(indbe[:, 0], tmp)
        tmp = np.array((bmax - xbeg)/self.dx + self.stencil.vmax[:self.dim] + 1, np.int)
        nmax = np.minimum(indbe[:, 1], tmp)

        # set the grid
        x = self.x[0][nmin[0]:nmax[0]]
        y = self.x[1][nmin[1]:nmax[1]]
        if self.dim == 3:
            z = self.x[2][nmin[2]:nmax[2]]

        if self.dim == 2:
            gridx = x[:, np.newaxis]
            gridy = y[np.newaxis, :]

            # local view of the arrays
            ioo_view = self.in_or_out[nmin[0]:nmax[0], nmin[1]:nmax[1]]
            dist_view = self.distance[:, nmin[0]:nmax[0], nmin[1]:nmax[1]]
            flag_view = self.flag[:, nmin[0]:nmax[0], nmin[1]:nmax[1]]
        else:
            gridx = x[:, np.newaxis, np.newaxis]
            gridy = y[np.newaxis, :, np.newaxis]
            gridz = z[np.newaxis, np.newaxis, :]

            # local view of the arrays
            ioo_view = self.in_or_out[nmin[0]:nmax[0], nmin[1]:nmax[1], nmin[2]:nmax[2]]
            dist_view = self.distance[:, nmin[0]:nmax[0], nmin[1]:nmax[1], nmin[2]:nmax[2]]
            flag_view = self.flag[:, nmin[0]:nmax[0], nmin[1]:nmax[1], nmin[2]:nmax[2]]

        if not elem.isfluid: # add a solid part
            if self.dim == 2:
                ind_solid = elem.point_inside(gridx, gridy)
            else:
                ind_solid = elem.point_inside(gridx, gridy, gridz)
            ind_fluid = np.logical_not(ind_solid)
            ioo_view[ind_solid] = self.valout
        else: # add a fluid part
            if self.dim == 2:
                ind_fluid = elem.point_inside(gridx, gridy)
            else:
                ind_fluid = elem.point_inside(gridx, gridy, gridz)
            ind_solid = np.logical_not(ind_fluid)
            ioo_view[ind_fluid] = self.valin

        if self.dim == 2:
            for k in range(self.stencil.unvtot):
                vxk = self.stencil.unique_velocities[k].vx
                vyk = self.stencil.unique_velocities[k].vy
                if (vxk != 0 or vyk != 0):
                    condx = self.in_or_out[nmin[0] + vxk:nmax[0] + vxk, nmin[1] + vyk:nmax[1] + vyk] == self.valout
                    alpha, border = elem.distance(gridx, gridy, (self.dx*vxk, self.dx*vyk), 1.)
                    with np.errstate(invalid='ignore'):
                        indx = np.logical_and(np.logical_and(alpha > 0, ind_fluid), condx)

                    if elem.isfluid:
                        # take all points in the fluid in the ioo_view
                        indfluidinbox = ioo_view == self.valin
                        # take only points which
                        borderToInt = np.logical_and(np.logical_not(condx), indfluidinbox)
                        dist_view[k][borderToInt] = self.valin
                        flag_view[k][borderToInt] = self.valin
                    else:
                        dist_view[k][ind_solid] = self.valin
                        flag_view[k][ind_solid] = self.valin

                    #set distance
                    ind4 = np.where(indx)
                    if not elem.isfluid:
                        ind3 = np.where(alpha[ind4] < dist_view[k][ind4])
                    else:
                        ind3 = np.where(np.logical_or(alpha[ind4] > dist_view[k][ind4], dist_view[k][ind4] == self.valin))

                    dist_view[k][ind4[0][ind3[0]], ind4[1][ind3[0]]] = alpha[ind4[0][ind3[0]], ind4[1][ind3[0]]]
                    flag_view[k][ind4[0][ind3[0]], ind4[1][ind3[0]]] = border[ind4[0][ind3[0]], ind4[1][ind3[0]]]
        else:
            for k in range(self.stencil.unvtot):
                vxk = self.stencil.unique_velocities[k].vx
                vyk = self.stencil.unique_velocities[k].vy
                vzk = self.stencil.unique_velocities[k].vz

                if (vxk != 0 or vyk != 0 or vzk != 0):
                    condx = self.in_or_out[nmin[0] + vxk:nmax[0] + vxk, nmin[1] + vyk:nmax[1] + vyk, nmin[2] + vzk:nmax[2] + vzk] == self.valout
                    alpha, border = elem.distance(gridx, gridy, gridz, (self.dx*vxk, self.dx*vyk, self.dx*vzk), 1.)
                    with np.errstate(invalid='ignore'):
                        indx = np.logical_and(np.logical_and(alpha > 0, ind_fluid), condx)

                    if elem.isfluid:
                        # take all points in the fluid in the ioo_view
                        indfluidinbox = ioo_view == self.valin
                        # take only points which
                        borderToInt = np.logical_and(np.logical_not(condx), indfluidinbox)
                        dist_view[k][borderToInt] = self.valin
                        flag_view[k][borderToInt] = self.valin
                    else:
                        dist_view[k][ind_solid] = self.valin
                        flag_view[k][ind_solid] = self.valin

                    #set distance
                    ind4 = np.where(indx)
                    if not elem.isfluid:
                        ind3 = np.where(alpha[ind4] < dist_view[k][ind4])
                    else:
                        ind3 = np.where(np.logical_or(alpha[ind4] > dist_view[k][ind4], dist_view[k][ind4] == self.valin))

                    dist_view[k][ind4[0][ind3[0]], ind4[1][ind3[0]], ind4[2][ind3[0]]] = alpha[ind4[0][ind3[0]], ind4[1][ind3[0]], ind4[2][ind3[0]]]
                    flag_view[k][ind4[0][ind3[0]], ind4[1][ind3[0]], ind4[2][ind3[0]]] = border[ind4[0][ind3[0]], ind4[1][ind3[0]], ind4[2][ind3[0]]]

    def visualize(self, viewer_app=viewer.matplotlibViewer, view_distance=False, view_in=True, view_out=True, view_bound=False, label=None):
        """
        Visualize the domain by creating a plot.

        Parameters
        ----------
        viewer_app : Viewer, optional
            define the viewer to plot the domain
            default is viewer.matplotlibViewer
        view_distance : boolean, optional
            view the distance between the interior points and the border
        label : int or list of int, optional
            view the distance only for the specified labels

        Returns
        -------
        a figure representing the domain
        """
        fig = viewer_app.Fig(dim = self.dim)
        view = fig[0]

        if isinstance(view_distance, bool):
            view_seg = view_distance
            view_distance = list(range(self.stencil.unvtot))
        elif isinstance(view_distance, int):
            view_seg = True
            view_distance = (view_distance,)
        elif isinstance(view_distance, (list, tuple)):
            view_seg = True
        else:
            s = "Error in visualize (domain): \n"
            s += "optional parameter view_distance should be\n"
            s += "  boolean, integer, list or tuple\n"
            self.log.error(s)

        if self.dim == 1:
            x = self.x[0]
            y = np.zeros(x.shape)
            vkmax = self.stencil.vmax[0]
            for k in range(self.stencil.unvtot):
                vk = self.stencil.unique_velocities[k].vx
                color = (1.-(vkmax+vk)*0.5/vkmax, 0., (vkmax+vk)*0.5/vkmax)
                indbord = np.where(self.distance[k,:]<=1)[0]
                if indbord.size != 0:
                    xx = x[indbord]
                    yy = y[indbord]
                    dist = self.distance[k, indbord]
                    dx = self.dx
                    l = np.empty((2*indbord.size, 2))
                    l[::2, :] = np.asarray([xx, yy]).T
                    l[1::2, :] = np.asarray([xx + dx*dist*vk, yy]).T
                    view.segments(l, color=color)
            indin = np.where(self.in_or_out==self.valin)
            view.markers(np.asarray([x[indin],y[indin]]).T, 200*self.dx, symbol='*')
            indout = np.where(self.in_or_out==self.valout)
            view.markers(np.asarray([x[indout],y[indout]]).T, 200*self.dx, symbol='s')

            xmin, xmax = self.bounds[0][:]
            L = xmax-xmin
            h = L/20
            l = L/50
            view.axis(xmin - L/2, xmax + L/2, -10*h, 10*h)

        elif self.dim == 2:

            if not view_seg:
                inT = self.in_or_out
                xmax, ymax = inT.shape
                xmax -= 1
                ymax -= 1
                xpercent = 0.05*xmax
                ypercent = 0.05*ymax
                view.axis(-xpercent, xmax+xpercent, -ypercent, ymax+ypercent)
                view.image(inT.transpose()>=0)
            else:
                xmin, xmax = self.bounds[0][:]
                ymin, ymax = self.bounds[1][:]

                xpercent = 0.05*(xmax-xmin)
                ypercent = 0.05*(ymax-ymin)
                view.axis(xmin-xpercent, xmax+xpercent, ymin-ypercent, ymax+ypercent)

                x, y = self.x[:]
                dx = self.dx
                vxkmax, vykmax = self.stencil.vmax[:self.dim]

                for k in view_distance:
                    vxk = self.stencil.unique_velocities[k].vx
                    vyk = self.stencil.unique_velocities[k].vy
                    color = (1.-(vxkmax+vxk)*0.5/vxkmax, (vykmax+vyk)*0.5/vykmax, (vxkmax+vxk)*0.5/vxkmax)
                    indbordx, indbordy = np.where(self.distance[k, :]<=1)
                    if indbordx.size != 0:
                        dist = self.distance[k, indbordx, indbordy]
                        xx = x[indbordx]
                        yy = y[indbordy]
                        l = np.empty((2*xx.size, 2))
                        l[::2, :] = np.asarray([xx, yy]).T
                        l[1::2, :] = np.asarray([xx + dx*dist*vxk, yy + dx*dist*vyk]).T
                        view.segments(l, color=color)

                indinx, indiny = np.where(self.in_or_out==self.valin)
                view.markers(np.asarray([x[indinx], y[indiny]]).T, 500*self.dx, symbol='o')
                indoutx, indouty = np.where(self.in_or_out==self.valout)
                view.markers(np.asarray([x[indoutx], y[indouty]]).T, 500*self.dx, symbol='s')

        elif self.dim == 3:
            x, y, z = self.x[:]
            dx = self.dx
            xmin, xmax = self.bounds[0][:]
            ymin, ymax = self.bounds[1][:]
            zmin, zmax = self.bounds[2][:]

            xpercent = 0.05*(xmax-xmin)
            ypercent = 0.05*(ymax-ymin)
            zpercent = 0.05*(zmax-zmin)
            view.axis(xmin-xpercent, xmax+xpercent, ymin-ypercent, ymax+ypercent, zmin-zpercent, zmax+zpercent)

            if view_in:
                indinx, indiny, indinz = np.where(self.in_or_out==self.valin)
                view.markers(np.asarray([x[indinx], y[indiny], z[indinz]]).T, 50*self.dx**2, symbol='o', color='1.')
            if view_out:
                indoutx, indouty, indoutz = np.where(self.in_or_out==self.valout)
                view.markers(np.asarray([x[indoutx], y[indouty], z[indoutz]]).T, 200*self.dx**2, symbol='o', color='0.')
            view.set_label("X", "Y", "Z")
            if view_seg or view_bound:
                vxkmax, vykmax, vzkmax = self.stencil.vmax[:self.dim]
                for k in view_distance:
                    vxk = self.stencil.unique_velocities[k].vx
                    vyk = self.stencil.unique_velocities[k].vy
                    vzk = self.stencil.unique_velocities[k].vz
                    color = (1.-(vxkmax+vxk)*0.5/vxkmax, (vykmax+vyk)*0.5/vykmax, (vzkmax+vzk)*0.5/vzkmax)
                    if label is not None:
                        dummy = np.zeros(self.distance.shape[1:])
                        if isinstance(label, int):
                            dummy = np.logical_or(dummy, self.flag[k,:]==label)
                        elif isinstance(label, (tuple, list)):
                            for labelk in label:
                                dummy = np.logical_or(dummy, self.flag[k,:]==labelk)
                        else:
                            self.log.error("Error in visualize (domain): wrong type for optional argument label")
                    else:
                        dummy = np.ones(self.distance.shape[1:])
                    dummy = np.logical_and(dummy, self.distance[k,:]<=1)
                    indbordx, indbordy, indbordz = np.where(dummy)
                    if indbordx.size != 0:
                        dist = self.distance[k, indbordx, indbordy, indbordz]
                        xx = x[indbordx]
                        yy = y[indbordy]
                        zz = z[indbordz]
                        if view_seg:
                            l = np.empty((2*xx.size, 3))
                            l[::2, :] = np.asarray([xx, yy, zz]).T
                            l[1::2, :] = np.asarray([xx + dx*dist*vxk, yy + dx*dist*vyk, zz + dx*dist*vzk]).T
                            view.segments(l, color=color, width=3)
                        if view_bound:
                            l = np.asarray([xx + dx*dist*vxk, yy + dx*dist*vyk, zz + dx*dist*vzk]).T
                            view.markers(l, 200*self.dx**2, symbol='o', color=color)
        else:
            self.log.error('Error in domain.visualize(): the dimension {0} is not allowed'.format(self.dim))

        view.title = "Domain"
        fig.show()

def verification(dom, with_color=False):
    """
    Function that writes all the informations of a domain

    Parameters
    ----------
    dom : a valid object of the class :py:class:`LBMpy.domain.Domain`
    with_color : a boolean (False by default) to use color in the shell

    Returns
    -------
    print the number of points
    print the array ``in_or_out`` (999 for inner points and -1 for outer points)
    for each velocity v, print the distance to the border and the flag of the reached border (for each boundary point)

    """
    # some terminal colors
    if with_color:
        blue = '\033[01;05;44m'
        black = '\033[0m'
        green = '\033[92m'
        white = '\033[01;37;44m'
    else:
        blue = ''
        black = ''
        green = ''
        white = ''

    print('Nombre de points : ' + str(dom.Na) + '\n')
    if (dom.dim==1):
        for k in range(dom.Na[0]):
            print('{0:3d}'.format((int)(dom.in_or_out[k])), end=' ')
        print(' ')
        for k in range(1, dom.stencil.unvtot):
            vx = dom.stencil.unique_velocities[k].vx
            print('*'*50)
            print('Check the velocity {0:2d} = {1:2d}'.format(k, vx))
            print('-'*50)
            print('Distances')
            for i in range(dom.Na[0]):
                if (dom.in_or_out[i]==dom.valout):
                    print(blue + ' *  ' + black, end=' ')
                elif (dom.distance[k,i]==dom.valin):
                    print(' .  ', end=' ')
                else:
                    print(green + '{0:.2f}'.format(dom.distance[k,i]) + black, end=' ')
            print()
            print('-'*50)
            print('Border Flags')
            for i in range(dom.Na[0]):
                if (dom.in_or_out[i]==dom.valout):
                    print(blue + ' *  ' + black, end=' ')
                elif (dom.distance[k,i]==dom.valin):
                    print(' .  ', end=' ')
                else:
                    print(green + '{0:.2f}'.format(dom.flag[k,i]) + black, end=' ')
            print()
            print('*'*50)
    if (dom.dim==2):
        for k in range(dom.Na[1]-1, -1, -1):
            for l in range(dom.Na[0]):
                print('{0:3d}'.format((int)(dom.in_or_out[k,l])), end=' ')
            print(' ')
        for k in range(dom.stencil.unvtot):
            vx = dom.stencil.unique_velocities[k].vx
            vy = dom.stencil.unique_velocities[k].vy
            print('*'*50)
            print('Check the velocity {0:2d} = ({1:2d},{2:2d})'.format(k, vx, vy))
            print('-'*50)
            print('Distances')
            for j in range(dom.Na[1]-1,-1,-1):
                for i in range(dom.Na[0]):
                    if (dom.distance[k,j,i] > 1 and dom.distance[k,j,i]<dom.valin): # nothing
                        print(white + '{0:3d} '.format(int(dom.distance[k,j,i])) + black, end=' ')
                    elif (dom.in_or_out[j,i] == dom.valout): # outer
                        print(blue + '  * ' + black, end=' ')
                    elif (dom.distance[k,j,i]==dom.valin):  # inner
                        print('  . ', end=' ')
                    else: # border
                        print(green + '{0:.2f}'.format(dom.distance[k,j,i]) + black, end=' ')
                print()
            print('-'*50)
            print('Border flags')
            for j in range(dom.Na[1]-1,-1,-1):
                for i in range(dom.Na[0]):
                    if (dom.distance[k,j,i] > 1 and dom.distance[k,j,i]<dom.valin):
                        print(white + '{0:3d} '.format(int(dom.distance[k,j,i])) + black, end=' ')
                    elif (dom.in_or_out[j,i] == dom.valout):
                        print(blue + '  * ' + black, end=' ')
                    elif (dom.distance[k,j,i]==dom.valin):
                        print('  . ', end=' ')
                    else:
                        print(green + '{0:.2f}'.format(dom.flag[k,j,i]) + black, end=' ')
                print()
            print('*'*50)


if __name__ == "__main__":
    import pyLBM
    dico = {
        'box':{'x': [0, 1], 'y': [0, 1], 'label':0},
        'elements':[pyLBM.Circle((0.5,0.5), 0.2, label = 1)],
        'space_step':0.05,
        'schemes':[{'velocities':list(range(9))}]
    }
    dom = pyLBM.Domain(dico)
    dom.visualize(opt=0)
    dom.visualize(opt=1)
    verification(dom, with_color = True)
