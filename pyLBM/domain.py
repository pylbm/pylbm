# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np
import logging
import sys

from .elements import *
import geometry as pyLBMGeom
import stencil as pyLBMSten

from .logs import setLogger
log = setLogger(__name__)

import pylab as plt
import matplotlib.cm as cm


class Domain:
    """
    Create a domain that defines the fluid part and the solid part
    and computes the distances between these two states.

    Parameters
    ----------
    dico : a dictionary that contains the following `key:value`

        box : a dictionary that contains the following `key:value`
            x : a list of the bounds in the first direction
            y : a list of the bounds in the second direction (optional)
            z : a list of the bounds in the third direction (optional)
            label : an integer or a list of integers (length twice the number of dimensions)
                used to label each edge

        elements : TODO ............................

        'space_step' : dx where dx is the value of the space step
        'number_of_scheme' : ns where ns is the value of the number of
            elementary schemes
        0:dico0, 1:dico1, ..., (nscheme-1):dico(nscheme-1) where k:dicok
              contains the velocities of the kth stencil
              (dicok['velocities'] is the list of the velocity indices for the
               kth stencil)

    optional parameters :
        if the geometry and/or the stencil were previously generated,
        it can be used directly as following
            >>> Domain(dico, geometry = geom, stencil = sten)
        if geom is an object of the class :py:class:`pyLBM.geometry.Geometry`
        and sten an object of the class :py:class:`pyLBM.stencil.Stencil`
        In that case, dico does not need to contain the informations for generate
        the geometry and/or the stencil


    Warning
    -------
    ``the sizes of the box must be a multiple of the space step dx``

    Attributes
    ----------
    dim : number of spatial dimensions (example: 1, 2, or 3)
    bounds : a list that contains the bounds of the box
        [[x[0]min,x[0]max],...,[x[dim-1]min,x[dim-1]max]]
    dx : space step (example: 0.1, 1.e-3)
    type : type of data (example: 'float64')
    stencil : the stencil of velocities (object of the class
             :py:class:`pyLBM.stencil.Stencil`)
    N : number of points in each direction
    Na : augmented number of points (Na = N + 2*vmax in each direction)
    indbe : list of indices for the loops in the inner dommain
            indbe[k][0]:indebe[k][1] is the list of indices in the kth direction
    x : coordinates of th domain
    in_or_out : NumPy array that defines the fluid and the solid part
                fluid part : value=valin
                solid part : value=valout
    distance : NumPy array that defines the distances to the borders.
               The distance is scaled by dx and is not equal to valin only for
               the points that reach the border with the specified velocity.
               In 1D, distance[k, i] is the distance between the point x[0][i]
               and the border in the direction of the kth velocity.
               In 2D, distance[k, j, i] is the distance between the point
               (x[0][i], x[1][j]) and the border in the direction of kth
               velocity
               ...
    flag : NumPy array that defines the flag of the border reached with the
           specified velocity
           In 1D, flag[k, i] is the flag of the border reached by the point
           x[0][i] in the direction of the kth velocity
           In 2D, flag[k, j, i] is the flag of the border reached by the point
           (x[0][i], x[1][j]) in the direction of kth velocity
           ...

    Members
    -------
    visualize : Visualize the domain by creating a plot

    Examples
    --------
    see demo/examples/domain/*.py

    """
    def __init__(self, dico, geometry=None, stencil=None):
        self.type = 'float64'
        if geometry is not None:
            self.geom = geometry
        else:
            self.geom = pyLBMGeom.Geometry(dico)
        if stencil is not None:
            self.stencil = stencil
        else:
            self.stencil = pyLBMSten.Stencil(dico)
        if self.geom.dim != self.stencil.dim:
            s = 'Error in the dimension: stencil and geometry dimensions are different'
            s += 'geometry: {0:d}, stencil: {1:d}'.format(self.geom.dim, self.sten.dim)
            log.error(s)
            sys.exit()
        else:
            self.dim = self.geom.dim # spatial dimension
        self.bounds = self.geom.bounds # the box where the domain lies
        self.dx = dico['space_step'] # spatial step
        # spatial mesh
        debord = [self.dx*(self.stencil.vmax[k] - 0.5) for k in xrange(self.dim)]
        self.N = [int((self.bounds[k][1] - self.bounds[k][0] + 0.5*self.dx)/self.dx) for k in xrange(self.dim)]
        self.Na = [self.N[k] + 2*self.stencil.vmax[k] for k in xrange(self.dim)]
        self.x = np.asarray([np.linspace(self.bounds[k][0] - debord[k],
                                         self.bounds[k][1] + debord[k],
                                         self.Na[k]) for k in xrange(self.dim)])
        self.indbe = np.asarray([(self.stencil.vmax[k],
                                  self.stencil.vmax[k] + self.N[k]) for k in xrange(self.dim)])

        # distance to the borders
        self.valin = 999  # value in the fluid domain
        self.valout = -1   # value in the solid domain

        s1 = self.Na[self.dim - 1::-1]
        s2 = np.concatenate(([self.stencil.unvtot], s1))
        self.in_or_out = self.valin*np.ones(s1, dtype = self.type)
        self.distance = self.valin*np.ones(s2, dtype = self.type)
        self.flag = self.valin*np.ones(s2, dtype = 'int')

        self.__add_init(self.geom.box_label) # compute the distance and the flag for the primary box
        for elem in self.geom.list_elem: # treat each element of the geometry
            self.__add_elem(elem)

        log.info(self.__str__())

    def __str__(self):
        s = "Domain informations\n"
        s += "\t spatial dimension: {0:d}\n".format(self.dim)
        s += "\t bounds of the box: bounds = " + self.bounds.__str__() + "\n"
        s += "\t space step: dx={0:10.3e}\n".format(self.dx)
        s += "\t Number of points in each direction: N=" + self.N.__str__() + ", Na=" + self.Na.__str__() + "\n"
        return s

    def __add_init(self, label):
        if (self.dim == 1):
            vmax = self.stencil.vmax[0]
            xb, xe = self.indbe[0][:]

            self.in_or_out[:] = self.valout
            self.in_or_out[xb:xe] = self.valin

            uvx = self.stencil.uvx
            for k, vk in np.ndenumerate(uvx):
                if (vk > 0):
                    for i in xrange(vk):
                        self.distance[k, xe - 1 - i] = (i + .5)/vk
                        self.flag[k, xe - 1 - i] = label[0] # east border
                elif (vk < 0):
                    for i in xrange(-vk):
                        self.distance[k, xb + i] = -(i + .5)/vk
                        self.flag[k, xb + i] = label[1] # west border

        elif (self.dim == 2):
            vxmax, vymax = self.stencil.vmax[:2]
            xb, xe = self.indbe[0][:]
            yb, ye = self.indbe[1][:]

            self.in_or_out[:, :] = self.valout
            self.in_or_out[yb:ye,xb:xe] = self.valin

            for k in xrange(self.stencil.unvtot):
                vxk = self.stencil.unique_velocities[k].vx
                vyk = self.stencil.unique_velocities[k].vy
                if (vxk > 0):
                    for i in xrange(vxk):
                        dvik = (i + .5)/vxk
                        indbordvik = np.where(dvik < self.distance[k, yb:ye, xe - 1 - i])
                        self.distance[k, yb + indbordvik[0], xe - 1 - i] = dvik
                        self.flag[k, yb + indbordvik[0], xe - 1 - i] = label[1]
                elif (vxk < 0):
                    for i in xrange(-vxk):
                        dvik = -(i + .5)/vxk
                        indbordvik = np.where(dvik < self.distance[k, yb:ye, xb + i])
                        self.distance[k, yb + indbordvik[0], xb + i] = dvik
                        self.flag[k, yb + indbordvik[0], xb + i] = label[3]
                if (vyk > 0):
                    for i in xrange(vyk):
                        dvik = (i + .5)/vyk
                        indbordvik = np.where(dvik < self.distance[k, ye - 1 - i, xb:xe])
                        self.distance[k, ye - 1 - i, xb + indbordvik[0]] = dvik
                        self.flag[k, ye - 1 - i, xb + indbordvik[0]] = label[2]
                elif (vyk < 0):
                    for i in xrange(-vyk):
                        dvik = -(i + .5)/vyk
                        indbordvik = np.where(dvik < self.distance[k, yb + i, xb:xe])
                        self.distance[k, yb + i, xb + indbordvik[0]] = dvik
                        self.flag[k, yb + i, xb + indbordvik[0]] = label[0]
        return

    def __add_elem(self, elem):
        """
        Add an element

            - if elem.isfluid = False as a solid part. (bw=0)
            - if elem.isfluid = True as a fluid part.  (bw=1)

        """
        # compute the box around the element adding vmax safety points
        bmin, bmax = elem.get_bounds()
        xbeg = np.asarray([self.x[0][0], self.x[1][0]])

        tmp = np.array((bmin - xbeg)/self.dx - self.stencil.vmax[:self.dim], np.int)
        nmin = np.maximum(self.indbe[:, 0], tmp)
        tmp = np.array((bmax - xbeg)/self.dx + self.stencil.vmax[:self.dim] + 1, np.int)
        nmax = np.minimum(self.indbe[:, 1], tmp)

        # set the grid
        x = self.x[0][nmin[0]:nmax[0]]
        y = self.x[1][nmin[1]:nmax[1]]
        gridx = x[np.newaxis, :]
        gridy = y[:, np.newaxis]

        # local view of the arrays
        ioo_view = self.in_or_out[nmin[1]:nmax[1], nmin[0]:nmax[0]]
        dist_view = self.distance[:, nmin[1]:nmax[1], nmin[0]:nmax[0]]
        flag_view = self.flag[:, nmin[1]:nmax[1], nmin[0]:nmax[0]]

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
                condx = self.in_or_out[nmin[1] + vyk:nmax[1] + vyk, nmin[0] + vxk:nmax[0] + vxk] == self.valout
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

    def visualize(self, opt=0):
        """
        Visualize the domain by creating a plot

        optional argument ``opt`` if the spatial dimension dim is 2

        * If dim=1 or (dim=2 and opt=0)
             - plot a star on inner points and a square on outer points
             - plot the flag on the boundary (each boundary point + s[k]*unique_velocities[k] for each velocity k)
        * If dim=2 and opt=1
             - plot a imshow figure, white for inner domain and black for outer domain
        """
        fig = plt.figure(0,figsize=(8, 8))
        fig.clf()
        plt.ion()
        if (self.dim == 1):
            x = self.x[0]
            y = np.zeros(x.shape)
            vkmax = self.stencil.vmax[0]
            plt.hold(True)
            for k in xrange(self.stencil.unvtot):
                vk = self.stencil.unique_velocities[k].vx
                coul = (1.-(vkmax+vk)*0.5/vkmax, 0., (vkmax+vk)*0.5/vkmax)
                indbord = np.where(self.distance[k,:]<=1)
                #plt.scatter(x[indbord]+self.dx*self.distance[k,[indbord]]*vk, y[indbord], 1000*self.dx, c=coul, marker='^')
                for i in indbord[0]:
                    plt.text(x[i]+self.dx*self.distance[k,i]*vk, y[i],str(self.flag[k,i]),
                             fontsize=18, horizontalalignment='center',verticalalignment='center')
                    plt.plot([x[i],x[i]+self.dx*self.distance[k,i]*vk],[y[i],y[i]],c=coul)
            indin = np.where(self.in_or_out==self.valin)
            plt.scatter(x[indin],y[indin], 1000*self.dx, c='k', marker='*')
            indout = np.where(self.in_or_out==self.valout)
            plt.scatter(x[indout],y[indout], 1000*self.dx, c='k', marker='s')
        if (self.dim == 2):
            x = self.x[0][np.newaxis, :]
            y = self.x[1][:, np.newaxis]
            if (opt==0):
                plt.imshow(self.in_or_out>=0, origin='lower', cmap=cm.gray, interpolation='nearest')
            else:
                vxkmax = self.stencil.vmax[0]
                vykmax = self.stencil.vmax[1]
                plt.hold(True)
                for k in xrange(self.stencil.unvtot):
                    vxk = self.stencil.unique_velocities[k].vx
                    vyk = self.stencil.unique_velocities[k].vy
                    coul = (1.-(vxkmax+vxk)*0.5/vxkmax, (vykmax+vyk)*0.5/vykmax, (vxkmax+vxk)*0.5/vxkmax)
                    indbordy, indbordx = np.where(self.distance[k,:]<=1)
                    #plt.scatter(x[0, indbordx]+self.dx*self.distance[k, indbordy, indbordx]*vxk, y[indbordy, 0]+self.dx*self.distance[k, indbordy, indbordx]*vyk, 1000*self.dx, c=coul, marker='^')
                    for i in xrange(indbordx.shape[0]):
                        plt.text(x[0,indbordx[i]]+self.dx*self.distance[k,indbordy[i],indbordx[i]]*vxk,
                                 y[indbordy[i],0]+self.dx*self.distance[k,indbordy[i],indbordx[i]]*vyk,
                                 str(self.flag[k,indbordy[i],indbordx[i]]),
                                 fontsize=18, horizontalalignment='center',verticalalignment='center')
                        plt.plot([x[0,indbordx[i]],x[0,indbordx[i]]+self.dx*self.distance[k,indbordy[i],indbordx[i]]*vxk],
                                 [y[indbordy[i],0],y[indbordy[i],0]+self.dx*self.distance[k,indbordy[i],indbordx[i]]*vyk],c=coul)

                indiny, indinx = np.where(self.in_or_out==self.valin)
                plt.scatter(x[0, indinx], y[indiny, 0], 500*self.dx, c='k', marker='*')
                indouty, indoutx = np.where(self.in_or_out==self.valout)
                plt.scatter(x[0, indoutx], y[indouty, 0], 500*self.dx, c='k', marker='s')
        plt.title("Domain",fontsize=14)
        plt.draw()
        plt.hold(False)
        plt.ioff()
        plt.show()


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

    Ind = np.where(dom.in_or_out==0)
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
