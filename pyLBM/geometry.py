# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import sys
from math import sin, cos
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.patches import Ellipse, Polygon

import mpi4py.MPI as mpi

from elements import *

import logging

class Geometry:
    """
    Create a geometry that defines the fluid part and the solid part.

    Parameters
    ----------
    a dictionary that contains the following `key:value`
       - 'geometry'

    Attributes
    ----------
    dim        : number of spatial dimensions (example: 1, 2, or 3)
    bounds     : a list that contains the bounds of the box ((x[0]min,x[0]max),...,(x[dim-1]min,x[dim-1]max))
    bounds_tag : a dictionary that contains the tag of all the bounds and the description
    list_elem  : a list that contains each element added or deleted in the box
    list_tag   : a list that contains the tag of each element
    list_label : a list that contains the label of each border

    Members
    -------
    add_elem  : function that adds an element in the box
    visualize : function to visualize the box

    """

    def __init__(self, dico):
        try:
            boite = dico['box']
            boitex = boite.get('x', None)
            boitey = boite.get('y', None)
            boitez = boite.get('z', None)
        except KeyError:
            print "'box' key not found in the geometry definitiion. Check the input dictionnary."
            logging.error("'box' key not found in the geometry definitiion. Check the input dictionnary.")
            sys.exit()

        if boitex is None:
            print "'x' interval not found in the box definition of the geometry."
            sys.exit()
        else:
            self.bounds = [boitex]
            self.dim = 1
        if boitey is not None:
            self.bounds.append(boitey)
            self.dim += 1
            if boitez is not None:
                self.bounds.append(boitez)
                self.dim += 1

        # mpi support
        comm = mpi.COMM_WORLD
        size = comm.Get_size()
        split = mpi.Compute_dims(size, self.dim)

        self.bounds = np.asarray(self.bounds, dtype='f8')
        t = (self.bounds[:, 1] - self.bounds[:, 0])/split
        self.comm = comm.Create_cart(split, (True,)*self.dim)
        rank = self.comm.Get_rank()
        coords = self.comm.Get_coords(rank)
        coords = np.asarray(coords)
        self.bounds[:, 1] = self.bounds[:, 0] + t*(coords + 1)
        self.bounds[:, 0] = self.bounds[:, 0] + t*coords

        self.isInterface = [False]*2*self.dim
        for i in xrange(self.dim):
            voisins = self.comm.Shift(i, 1)
            if voisins[0] != rank:
                self.isInterface[i*2] = True
            if voisins[1] != rank:
                self.isInterface[i*2 + 1] = True

        print self.isInterface
        self.list_elem = []
        self.list_tag = []
        self.list_label = []

        self.next_tag = 2*self.dim

        try:
            dummylab = dico['box']['label']
        except:
            dummylab = 0
        if isinstance(dummylab, int):
            self.list_label.append([dummylab]*2*self.dim)
        else:
            self.list_label.append([loclab for loclab in dummylab])

        elem = dico.get('elements', None)
        if elem is not None:
            k = 0
            test = 1
            while (test == 1):
                elemk = elem.get(k, None)
                if elemk is not None:
                    k += 1
                    try:
                        labelk = elemk['label']
                    except:
                        labelk = 0
                    self.add_elem(elemk['element'], labelk, elemk['del'])
                else:
                    test = 0

            """
            for k in elem:
                elementk = dico[k]['element']
                try:
                    labelk = dico[k]['label']
                except:
                    labelk = 0
                self.add_elem(elementk, labelk, dico[k]['type'])
            """

    def __str__(self):
        s = "Geometry informations\n"
        s += "\t spatial dimension: dim={0:d}\n".format(self.dim)
        s += "\t bounds of the box: bounds = " + self.bounds.__str__() + "\n"
        if (len(self.list_elem) != 0):
            s += "\t List of elements added or deleted in the box\n"
            for k in xrange(len(self.list_elem)):
                s += "\t\t Element number {0:d}: ".format(k) + self.list_elem[k].__str__() + " (tag = " + self.list_tag[k].__str__() + ")\n"
        s += "\t informations for the boundary tags\n"
        # for k in xrange(self.next_tag):
        #     s += "\t\ttag {0:d}: ".format(k) + self.bounds_tag[k] + "\n"
        return s

    def add_elem(self, elem, label, bw):
        """
        add a solid or a fluid part in the domain.

        Parameters
        ----------
        elem  : form of the part to add
        label : label on the boundary of this form
        bw    : type of the form (0: solid, 1: fluid)

        Examples
        --------



        """
        elem.bw = bw

        self.list_elem.append(elem)
        # add a different tag for each bounds of the form
        self.list_tag.append([self.next_tag + k for k in xrange(elem.number_of_bounds)])

        # # for each bounds af the form add a description
        # for i in xrange(elem.number_of_bounds):
        #     self.bounds_tag[self.next_tag + i] = elem.description[i]
        self.next_tag += elem.number_of_bounds

        # don't understand what is elem.tag
        elem.tag = self.list_tag[-1]

        # set a label to the boundaries
        if isinstance(label, int):
            loclabel = [label]*elem.number_of_bounds
        else:
            loclabel = label.copy()
        self.list_label.append(loclabel)
        elem.label = loclabel

    def visualize(self, tag=False):
        plein = 'blue'
        fig = plt.figure(0,figsize=(8, 8))
        fig.clf()
        plt.ion()
        plt.hold(True)
        ax = fig.add_subplot(111)
        if (self.dim == 1):
            xmin = (float)(self.bounds[0][0])
            xmax = (float)(self.bounds[0][1])
            L = xmax-xmin
            h = L/20
            l = L/50
            plt.plot([xmin+l,xmin,xmin,xmin+l],[-h,-h,h,h],plein,lw=5)
            plt.plot([xmax-l,xmax,xmax,xmax-l],[-h,-h,h,h],plein,lw=5)
            plt.plot([xmin,xmax],[0.,0.],plein,lw=5)
            plt.text(xmax-l, -2*h, '0', fontsize=18, horizontalalignment='center',verticalalignment='center')
            plt.text(xmin+l, -2*h, '1', fontsize=18, horizontalalignment='center',verticalalignment='center')
            plt.axis('equal')
        elif (self.dim == 2):
            xmin = (float)(self.bounds[0][0])
            xmax = (float)(self.bounds[0][1])
            ymin = (float)(self.bounds[1][0])
            ymax = (float)(self.bounds[1][1])

            plt.fill([xmin,xmax,xmax,xmin], [ymin,ymin,ymax,ymax], fill=True, color=plein)
            if tag:
                plt.text(0.5*(xmin+xmax), ymin, '0', fontsize=18, horizontalalignment='center',verticalalignment='bottom')
                plt.text(xmax, 0.5*(ymin+ymax), '1', fontsize=18, horizontalalignment='right',verticalalignment='center')
                plt.text(0.5*(xmin+xmax), ymax, '2', fontsize=18, horizontalalignment='center',verticalalignment='top')
                plt.text(xmin, 0.5*(ymin+ymax), '3', fontsize=18, horizontalalignment='left',verticalalignment='center')
            else:
                plt.text(0.5*(xmin+xmax), ymin, self.list_label[0][0], fontsize=18, horizontalalignment='center',verticalalignment='bottom')
                plt.text(xmax, 0.5*(ymin+ymax), self.list_label[0][1], fontsize=18, horizontalalignment='right',verticalalignment='center')
                plt.text(0.5*(xmin+xmax), ymax, self.list_label[0][2], fontsize=18, horizontalalignment='center',verticalalignment='top')
                plt.text(xmin, 0.5*(ymin+ymax), self.list_label[0][3], fontsize=18, horizontalalignment='left',verticalalignment='center')
            #plt.axis('equal')
            plt.axis([xmin, xmax, ymin, ymax])
            comptelem = 0
            for elem in self.list_elem:
                if (elem.bw == 1):
                    coul = plein
                elif (elem.bw == 0):
                    coul = 'white'
                else:
                    coul = 'black'
                if (elem.geomtype=='Circle'):
                    ax.add_patch(Ellipse(elem.center, 2*elem.radius, 2*elem.radius, fill=True, color=coul))
                    theta = elem.center[0] + 2*elem.center[1]+10*elem.radius
                    x, y = elem.center[0] + elem.radius*cos(theta), elem.center[1] + elem.radius*sin(theta)
                    if tag:
                        plt.text(x, y, str(elem.tag[0]),
                                 fontsize=18, horizontalalignment='center',verticalalignment='center')
                    else:
                        plt.text(x, y, str(elem.label[0]),
                                 fontsize=18, horizontalalignment='center',verticalalignment='center')
                        comptelem += 1
                elif (elem.geomtype=='Parallelogram'):
                    A = [elem.point[k] for k in xrange(2)]
                    B = [A[k] + elem.v0[k] for k in xrange(2)]
                    C = [B[k] + elem.v1[k] for k in xrange(2)]
                    D = [A[k] + elem.v1[k] for k in xrange(2)]
                    ax.add_patch(Polygon([A,B,C,D], closed=True, fill=True, color=coul))
                    if tag:
                        plt.text(0.5*(A[0]+B[0]), 0.5*(A[1]+B[1]), str(elem.tag[0]),
                                 fontsize=18, horizontalalignment='center',verticalalignment='center')
                        plt.text(0.5*(A[0]+D[0]), 0.5*(A[1]+D[1]), str(elem.tag[1]),
                                 fontsize=18, horizontalalignment='center',verticalalignment='center')
                        plt.text(0.5*(C[0]+D[0]), 0.5*(C[1]+D[1]), str(elem.tag[2]),
                                 fontsize=18, horizontalalignment='center',verticalalignment='center')
                        plt.text(0.5*(B[0]+C[0]), 0.5*(B[1]+C[1]), str(elem.tag[3]),
                                 fontsize=18, horizontalalignment='center',verticalalignment='center')
                    else:
                        plt.text(0.5*(A[0]+B[0]), 0.5*(A[1]+B[1]), str(elem.label[0]),
                                 fontsize=18, horizontalalignment='center',verticalalignment='center')
                        plt.text(0.5*(A[0]+D[0]), 0.5*(A[1]+D[1]), str(elem.label[1]),
                                 fontsize=18, horizontalalignment='center',verticalalignment='center')
                        plt.text(0.5*(C[0]+D[0]), 0.5*(C[1]+D[1]), str(elem.label[2]),
                                 fontsize=18, horizontalalignment='center',verticalalignment='center')
                        plt.text(0.5*(B[0]+C[0]), 0.5*(B[1]+C[1]), str(elem.label[3]),
                                 fontsize=18, horizontalalignment='center',verticalalignment='center')
                        comptelem += 4
                elif (elem.geomtype=='Triangle'):
                    A = [elem.point[k] for k in xrange(2)]
                    B = [A[k] + elem.v0[k] for k in xrange(2)]
                    D = [A[k] + elem.v1[k] for k in xrange(2)]
                    ax.add_patch(Polygon([A,B,D], closed=True, fill=True, color=coul))
                    plt.text(0.5*(A[0]+B[0]), 0.5*(A[1]+B[1]), str(elem.tag[0]),
                             fontsize=18, horizontalalignment='center',verticalalignment='center')
                    plt.text(0.5*(A[0]+D[0]), 0.5*(A[1]+D[1]), str(elem.tag[1]),
                             fontsize=18, horizontalalignment='center',verticalalignment='center')
                    plt.text(0.5*(B[0]+D[0]), 0.5*(B[1]+D[1]), str(elem.tag[2]),
                             fontsize=18, horizontalalignment='center',verticalalignment='center')

        plt.title("Geometry",fontsize=14)
        plt.draw()
        plt.hold(False)
        plt.ioff()
        plt.show()

    def list_of_labels(self):
        """
           return the list of all the labels used in the geometry
        """
        L = np.empty(0, dtype=np.int32)
        for l in self.list_label:
            L = np.union1d(L, l)
        return L


def test_1D(number):
    """
    Test 1D-Geometry

    * ``Test_1D(0)`` for the segment (0,1)
    * ``Test_1D(1)`` for the segment (-1,2)
    """
    dim = 1
    print "\n\nTest number {0:d} in {1:d}D:".format(number, dim)
    if number == 0:
        dgeom = {'geometry':
                     {'dim': dim,
                      'box':{'x': [0, 1]},
                      }
                 }
        geom = Geometry(dgeom['geometry'])
    elif number == 1:
        dgeom = {'geometry':
                     {'dim': dim,
                      'box':{'x': [-1, 2]},
                      }
                 }
        geom = Geometry(dgeom['geometry'])
    try:
        print geom
    except:
        return 0
    geom.visualize()
    return 1

def test_2D(number):
    """
    Test 2D-Geometry

    * ``Test_2D(0)`` for the square [0,1]**2
    * ``Test_2D(1)`` for the rectangular cavity with a circular obstacle
    * ``Test_2D(2)`` for the circular cavity
    * ``Test_2D(3)`` for the square cavity with a triangular obstacle
    """
    dim = 2
    solid = 0
    fluid = 1

    dgeom = {'geometry':
                 {'dim': dim,
                  }
             }
    print "\n\nTest number {0:d} in {1:d}D:".format(number, dim)
    if number == 0:
        dgeom['geometry']['box'] = {'x': [0, 1], 'y': [0, 1]}
        geom = Geometry(dgeom['geometry'])
    elif number == 1:
        dgeom['geometry']['box'] = {'x': [0, 2], 'y': [0, 1]}
        geom = Geometry(dgeom['geometry'])
        geom.add_elem(Circle((0.5, 0.5), 0.1), 0, solid)
    elif number == 2:
        dgeom['geometry']['box'] = {'x': [0, 2], 'y': [0, 1]}
        geom = Geometry(dgeom['geometry'])
        geom.add_elem(Parallelogram((0, 0), (2, 0), (0, 1)), 0, solid)
        geom.add_elem(Parallelogram((0, 0.4), (2, 0), (0, 0.2)), 0, fluid)
        geom.add_elem(Circle((1, 0.5), 0.5), 0, fluid)
        geom.add_elem(Circle((1, 0.5), 0.2), 0, solid)
    elif number == 3:
        dgeom['geometry']['box'] = {'x': [0, 1], 'y': [0, 1]}
        geom = Geometry(dgeom['geometry'])
        geom.add_elem(Triangle((0.3, 0.3), (0.5, -0.1), (0.3, 0.5)), 0, solid)
    elif (number==4):
        dgeom['geometry']['box'] = {'x': [0, 2], 'y': [0, 1]}
        geom = Geometry(dgeom['geometry'])
        geom.add_elem(Parallelogram((0.4, 0.4), (0., 0.2), (0.2, 0.)), 0, solid)
        geom.add_elem(Parallelogram((1.4, 0.5), (0.1, 0.1), (0.1, -0.1)), 0, solid)
    try:
        print geom
        #geom.visualize()
    except:
        return 0
    geom.visualize()
    return 1

if __name__ == "__main__":
    k = 1
    compt = 0
    while k==1:
        k = test_1D(compt)
        compt += 1
    k = 1
    compt = 2
    while (k==1):
        k = test_2D(compt)
        compt += 1
