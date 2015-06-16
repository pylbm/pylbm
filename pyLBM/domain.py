# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np
import sys
import copy

from .elements import *
from .geometry import Geometry
from .stencil import Stencil
from .logs import setLogger
from . import viewer

class Domain:
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

    In 1D, distance[k, i] is the distance between the point x[0][i]
    and the border in the direction of the kth velocity.

    In 2D, distance[k, j, i] is the distance between the point
    (x[0][i], x[1][j]) and the border in the direction of kth
    velocity

    In 3D, TODO

    In 1D, flag[k, i] is the flag of the border reached by the point
    x[0][i] in the direction of the kth velocity

    In 2D, flag[k, j, i] is the flag of the border reached by the point
    (x[0][i], x[1][j]) in the direction of kth velocity

    In 3D, TODO

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
    def __init__(self, dico=None, geometry=None, stencil=None, space_step=None):
        self.log = setLogger(__name__)

        self.geom = Geometry(dico) if geometry is None else geometry
        self.stencil = Stencil(dico) if stencil is None else stencil
        self.dx = dico['space_step'] if space_step is None else space_step
        self.dim = self.geom.dim

        if self.geom.dim != self.stencil.dim:
            s = 'Error in the dimension: stencil and geometry dimensions are different'
            s += 'geometry: {0:d}, stencil: {1:d}'.format(self.geom.dim, self.stencil.dim)
            self.log.error(s)

        self.globalbounds = self.geom.globalbounds # the box where the domain lies
        self.bounds = self.geom.bounds # the local box of the process

        get_shape = lambda x: int((x[1] - x[0] + .5*self.dx)/self.dx)
        self.Ng = map(get_shape, self.globalbounds[:self.dim])
        self.N = map(get_shape, self.bounds[:self.dim])

        # spatial mesh
        self.extent = np.asarray(self.stencil.vmax[:self.dim])
        debord = self.dx*(self.extent - 0.5)
        Na = np.asarray(self.N) + 2*self.extent
        self.x = np.asarray([np.linspace(self.bounds[k][0] - debord[k],
                                         self.bounds[k][1] + debord[k],
                                         Na[k]) for k in xrange(self.dim)])

        for k in xrange(self.dim):
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
            for j in xrange(self.dim):
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
                    for i in xrange(-vk):
                        indices[iuv + 1] = i
                        dvik = -(i + .5)/vk
                        nind = new_indices(dvik, iuv, indices, dist_view)
                        dist_view[nind] = dvik
                        flag_view[nind] = label[2*iuv]
                elif vk > 0 and label[2*iuv + 1] != -2:
                    for i in xrange(vk):
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
                             self.stencil.vmax[k] + self.N[k]) for k in xrange(self.dim)])
        bmin, bmax = elem.get_bounds()
        xbeg = np.asarray([self.x[0][0], self.x[1][0]])

        tmp = np.array((bmin - xbeg)/self.dx - self.stencil.vmax[:self.dim], np.int)
        nmin = np.maximum(indbe[:, 0], tmp)
        tmp = np.array((bmax - xbeg)/self.dx + self.stencil.vmax[:self.dim] + 1, np.int)
        nmax = np.minimum(indbe[:, 1], tmp)

        # set the grid
        x = self.x[0][nmin[0]:nmax[0]]
        y = self.x[1][nmin[1]:nmax[1]]
        gridx = x[:, np.newaxis]
        gridy = y[np.newaxis, :]

        # local view of the arrays
        ioo_view = self.in_or_out[nmin[0]:nmax[0], nmin[1]:nmax[1]]
        dist_view = self.distance[:, nmin[0]:nmax[0], nmin[1]:nmax[1]]
        flag_view = self.flag[:, nmin[0]:nmax[0], nmin[1]:nmax[1]]

        if not elem.isfluid: # add a solid part
            ind_solid = elem.point_inside(gridx, gridy)
            ind_fluid = np.logical_not(ind_solid)
            ioo_view[ind_solid] = self.valout
        else: # add a fluid part
            ind_fluid = elem.point_inside(gridx, gridy)
            ind_solid = np.logical_not(ind_fluid)
            ioo_view[ind_fluid] = self.valin

        for k in xrange(self.stencil.unvtot):
            vxk = self.stencil.unique_velocities[k].vx
            vyk = self.stencil.unique_velocities[k].vy
            if (vxk != 0 or vyk != 0):
                condx = self.in_or_out[nmin[0] + vxk:nmax[0] + vxk, nmin[1] + vyk:nmax[1] + vyk] == self.valout
                alpha, border = elem.distance(gridx, gridy, (self.dx*vxk, self.dx*vyk), 1.)
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

    def visualize(self, viewer_app=viewer.matplotlibViewer, opt=0):
        """
        Visualize the domain by creating a plot.

        Parameters
        ----------
        opt : int, optional
          optional argument for 2D geometries

        Returns
        -------
        If dim = 1 or (dim = 2 or 3 and opt = 1)
             - plot a star on inner points and a square on outer points
             - plot the flag on the boundary (each boundary point + s[k]*unique_velocities[k] for each velocity k)
        If dim = 2 or 3 and opt = 0
             - plot a imshow figure, white for inner domain and black for outer domain
        """
        fig = viewer_app.Fig()
        view = fig[0]

        if (self.dim == 1):
            x = self.x[0]
            y = np.zeros(x.shape)
            vkmax = self.stencil.vmax[0]
            for k in xrange(self.stencil.unvtot):
                vk = self.stencil.unique_velocities[k].vx
                color = (1.-(vkmax+vk)*0.5/vkmax, 0., (vkmax+vk)*0.5/vkmax)
                indbord = np.where(self.distance[k,:]<=1)[0]
                #plt.scatter(x[indbord]+self.dx*self.distance[k,[indbord]]*vk, y[indbord], 1000*self.dx, c=coul, marker='^')
                if indbord.size != 0:
                    xx = x[indbord]
                    yy = y[indbord]
                    dist = self.distance[k, indbord]
                    dx = self.dx
                    l = np.empty((2*indbord.size, 2))
                    l[::2, :] = np.asarray([xx, yy]).T
                    l[1::2, :] = np.asarray([xx + dx*dist*vk, yy]).T
                    view.segments(l, color=color)
                # for i in indbord[0]:
                #     view.text(str(self.flag[k,i]), [x[i]+self.dx*self.distance[k,i]*vk, y[i]])
                #     view.line(np.asarray([[x[i], y[i]],
                #                          [x[i]+self.dx*self.distance[k,i]*vk, y[i]]]), color=coul)
            indin = np.where(self.in_or_out==self.valin)
            view.markers(np.asarray([x[indin],y[indin]]).T, 200*self.dx, symbol='*')
            indout = np.where(self.in_or_out==self.valout)
            view.markers(np.asarray([x[indout],y[indout]]).T, 200*self.dx, symbol='s')

            xmin, xmax = self.bounds[0][:]
            L = xmax-xmin
            h = L/20
            l = L/50
            view.axis(xmin - L/2, xmax + L/2, -10*h, 10*h)

        elif (self.dim == 2):

            if (opt==0):
                inT = self.in_or_out
                xmax, ymax = inT.shape
                xpercent = 0.05*xmax
                ypercent = 0.05*ymax
                view.axis(-xpercent, xmax+xpercent, -ypercent, ymax+ypercent)
                view.image(inT>=0)
            else:
                xmin, xmax = self.bounds[0][:]
                ymin, ymax = self.bounds[1][:]

                xpercent = 0.05*(xmax-xmin)
                ypercent = 0.05*(ymax-ymin)
                view.axis(xmin-xpercent, xmax+xpercent, ymin-ypercent, ymax+ypercent)

                x, y = self.x[:]
                dx = self.dx
                vxkmax, vykmax = self.stencil.vmax[:self.dim]

                for k in xrange(self.stencil.unvtot):
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
        # elif (self.dim == 3):
        #     ax = fig.add_subplot(111, projection='3d')
        #     x = self.x[0][:, np.newaxis, np.newaxis]
        #     y = self.x[1][np.newaxis, :, np.newaxis]
        #     z = self.x[1][np.newaxis, np.newaxis, :]
        #     indinx, indiny, indinz = np.where(self.in_or_out==self.valin)
        #     ax.scatter(x[indinx, 0, 0], y[0, indiny, 0], z[0, 0, indinz],
        #                s = 100*self.dx**2, color='1.', marker='o'
        #                )
        #     indoutx, indouty, indoutz = np.where(self.in_or_out==self.valout)
        #     ax.scatter(x[indoutx, 0, 0], y[0, indouty, 0], z[0, 0, indoutz],
        #                s = 100*self.dx**2, c='0.', marker='o'
        #                )
        #     ax.set_xlabel("X")
        #     ax.set_ylabel("Y")
        #     ax.set_zlabel("Z")
        #     if (opt!=0):
        #         vxkmax = self.stencil.vmax[0]
        #         vykmax = self.stencil.vmax[1]
        #         vzkmax = self.stencil.vmax[2]
        #         for k in xrange(self.stencil.unvtot):
        #             vxk = self.stencil.unique_velocities[k].vx
        #             vyk = self.stencil.unique_velocities[k].vy
        #             vzk = self.stencil.unique_velocities[k].vz
        #             coul = (1.-(vxkmax+vxk)*0.5/vxkmax, (vykmax+vyk)*0.5/vykmax, (vzkmax+vzk)*0.5/vzkmax)
        #             indbordx, indbordy, indbordz = np.where(self.distance[k,:]<=1)
        #             for i in xrange(indbordx.shape[0]):
        #                 ax.text(x[indbordx[i],0,0]+self.dx*self.distance[k,indbordx[i],indbordy[i],indbordz[i]]*vxk,
        #                          y[0,indbordy[i],0]+self.dx*self.distance[k,indbordx[i],indbordy[i],indbordz[i]]*vyk,
        #                          z[0,0,indbordz[i]]+self.dx*self.distance[k,indbordx[i],indbordy[i],indbordz[i]]*vzk,
        #                          str(self.flag[k,indbordx[i],indbordy[i],indbordz[i]]),
        #                          fontsize=18)#, horizontalalignment='center',verticalalignment='center')
        #                 ax.plot([x[indbordx[i],0,0],x[indbordx[i],0,0]+self.dx*self.distance[k,indbordx[i],indbordy[i],indbordz[i]]*vxk],
        #                          [y[0,indbordy[i],0],y[0,indbordy[i],0]+self.dx*self.distance[k,indbordx[i],indbordy[i],indbordz[i]]*vyk],
        #                          [z[0,0,indbordz[i]],z[0,0,indbordz[i]]+self.dx*self.distance[k,indbordx[i],indbordy[i],indbordz[i]]*vzk],
        #                          c=coul)
        else:
            self.log.error('Error in domain.visualize(): the dimension {0} is not allowed'.format(self.dim))

        view.title = "Domain"
        view.draw()

        # plt.title("Domain",fontsize=14)
        # plt.draw()
        # plt.hold(False)
        # plt.ioff()
        # plt.show()


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

    print 'Nombre de points : ' + str(dom.Na) + '\n'
    if (dom.dim==1):
        for k in xrange(dom.Na[0]):
            print '{0:3d}'.format((int)(dom.in_or_out[k])),
        print ' '
        for k in xrange(1, dom.stencil.unvtot):
            vx = dom.stencil.unique_velocities[k].vx
            print '*'*50
            print 'Check the velocity {0:2d} = {1:2d}'.format(k, vx)
            print '-'*50
            print 'Distances'
            for i in xrange(dom.Na[0]):
                if (dom.in_or_out[i]==dom.valout):
                    print blue + ' *  ' + black,
                elif (dom.distance[k,i]==dom.valin):
                    print ' .  ',
                else:
                    print green + '{0:.2f}'.format(dom.distance[k,i]) + black,
            print
            print '-'*50
            print 'Border Flags'
            for i in xrange(dom.Na[0]):
                if (dom.in_or_out[i]==dom.valout):
                    print blue + ' *  ' + black,
                elif (dom.distance[k,i]==dom.valin):
                    print ' .  ',
                else:
                    print green + '{0:.2f}'.format(dom.flag[k,i]) + black,
            print
            print '*'*50
    if (dom.dim==2):
        for k in xrange(dom.Na[1]-1, -1, -1):
            for l in xrange(dom.Na[0]):
                print '{0:3d}'.format((int)(dom.in_or_out[k,l])),
            print ' '
        for k in xrange(dom.stencil.unvtot):
            vx = dom.stencil.unique_velocities[k].vx
            vy = dom.stencil.unique_velocities[k].vy
            print '*'*50
            print 'Check the velocity {0:2d} = ({1:2d},{2:2d})'.format(k, vx, vy)
            print '-'*50
            print 'Distances'
            for j in xrange(dom.Na[1]-1,-1,-1):
                for i in xrange(dom.Na[0]):
                    if (dom.distance[k,j,i] > 1 and dom.distance[k,j,i]<dom.valin): # nothing
                        print white + '{0:3d} '.format(int(dom.distance[k,j,i])) + black,
                    elif (dom.in_or_out[j,i] == dom.valout): # outer
                        print blue + '  * ' + black,
                    elif (dom.distance[k,j,i]==dom.valin):  # inner
                        print '  . ',
                    else: # border
                        print green + '{0:.2f}'.format(dom.distance[k,j,i]) + black,
                print
            print '-'*50
            print 'Border flags'
            for j in xrange(dom.Na[1]-1,-1,-1):
                for i in xrange(dom.Na[0]):
                    if (dom.distance[k,j,i] > 1 and dom.distance[k,j,i]<dom.valin):
                        print white + '{0:3d} '.format(int(dom.distance[k,j,i])) + black,
                    elif (dom.in_or_out[j,i] == dom.valout):
                        print blue + '  * ' + black,
                    elif (dom.distance[k,j,i]==dom.valin):
                        print '  . ',
                    else:
                        print green + '{0:.2f}'.format(dom.flag[k,j,i]) + black,
                print
            print '*'*50


if __name__ == "__main__":
    import pyLBM
    dico = {
        'box':{'x': [0, 1], 'y': [0, 1], 'label':0},
        'elements':[pyLBM.Circle((0.5,0.5), 0.2, label = 1)],
        'space_step':0.05,
        'schemes':[{'velocities':range(9)}]
    }
    dom = pyLBM.Domain(dico)
    dom.visualize(opt=0)
    dom.visualize(opt=1)
    verification(dom, with_color = True)
