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

from logs import setLogger
log = setLogger(__name__)

class Geometry:
    """
    Create a geometry that defines the fluid part and the solid part.

    Parameters
    ----------
    dico : a dictionary that contains the following `key:value`

        box : a dictionary that contains the following `key:value`
            x : a list of the bounds in the first direction
            y : a list of the bounds in the second direction (optional)
            z : a list of the bounds in the third direction (optional)
            label : an integer or a list of integers (length twice the number of dimensions)
                used to label each edge

        elements : TODO .....................

    Attributes
    ----------
    dim : number of spatial dimensions (example: 1, 2, or 3)
    bounds : a list that contains the bounds of the box ((x[0]min,x[0]max),...,(x[dim-1]min,x[dim-1]max))
    bounds_tag : a dictionary that contains the tag of all the bounds and the description
    list_elem : a list that contains each element added or deleted in the box
    # (to remove) list_label : a list that contains the label of each border

    Members
    -------
    add_elem : function that adds an element in the box
    visualize : function to visualize the box

    Examples
    --------
    see demo/examples/geometry/*.py

    """

    def __init__(self, dico):
        self.dim = 1
        try:
            box = dico['box']

            try:
                self.bounds = [box['x']]
                boxy = box.get('y', None)
                if boxy is not None:
                    self.bounds.append(boxy)
                    self.dim += 1
                    boxz = box.get('z', None)
                    if boxz is not None:
                        self.bounds.append(boxz)
                        self.dim += 1
            except KeyError:
                log.error("'x' key not found in the box definition"
                          + " of the geometry.\n"
                          + " Check the input dictionnary.")
                sys.exit()
        except KeyError:
            log.error("'box' key not found in the geometry definition.\n"
                      + " Check the input dictionnary.")
            sys.exit()

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

        s = "*"*40
        s += "Message from geometry.py (mpi problem)"
        s += self.isInterface.__str__()
        s += "*"*40
        log.info(s)

        self.list_elem = []
        #self.list_label = []

        #self.next_tag = 2*self.dim

        try:
            dummylab = dico['box']['label']
        except:
            dummylab = 0
        if isinstance(dummylab, int):
            #self.list_label.append([dummylab]*2*self.dim)
            self.box_label = [dummylab]*2*self.dim
        else:
            #self.list_label.append([loclab for loclab in dummylab])
            self.box_label = [loclab for loclab in dummylab]

        elem = dico.get('elements', None)
        if elem is not None:
            for elemk in elem:
                #self.add_elem(elemk)
                self.list_elem.append(elemk)
            #
            # k = 0
            # test = 1
            # while (test == 1):
            #     elemk = elem.get(k, None)
            #     if elemk is not None:
            #         k += 1
            #         try:
            #             labelk = elemk['label']
            #         except:
            #             labelk = 0
            #         self.add_elem(elemk['element'], labelk, elemk['del'])
            #     else:
            #         test = 0

            """
            for k in elem:
                elementk = dico[k]['element']
                try:
                    labelk = dico[k]['label']
                except:
                    labelk = 0
                self.add_elem(elementk, labelk, dico[k]['type'])
            """
        log.info(self.__str__())


    def __str__(self):
        s = "Geometry informations\n"
        s += "\t spatial dimension: {0:d}\n".format(self.dim)
        s += "\t bounds of the box: " + self.bounds.__str__() + "\n"
        if (len(self.list_elem) != 0):
            s += "\t List of elements added or deleted in the box\n"
            for k in xrange(len(self.list_elem)):
                s += "\t\t Element number {0:d}: ".format(k) + self.list_elem[k].__str__() + "\n"
        return s

    def add_elem(self, elem):
        """
        add a solid or a fluid part in the domain.

        Parameters
        ----------
        elem : form of the part to add (or to del)

        Examples
        --------

        """

        self.list_elem.append(elem)
        # add a different tag for each bounds of the form
        #self.list_tag.append([self.next_tag + k for k in xrange(elem.number_of_bounds)])

        # # for each bounds af the form add a description
        # for i in xrange(elem.number_of_bounds):
        #     self.bounds_tag[self.next_tag + i] = elem.description[i]
        #self.next_tag += elem.number_of_bounds

        # don't understand what is elem.tag
        #elem.tag = self.list_tag[-1]

        # set a label to the boundaries
        #if isinstance(label, int):
        #    loclabel = [label]*elem.number_of_bounds
        #else:
        #    loclabel = label.copy()
        #self.list_label.append(loclabel)
        #elem.label = loclabel

    def visualize(self, viewlabel=False):
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
            if viewlabel:
                plt.text(xmax-l, -2*h, self.box_label[0], fontsize=18, horizontalalignment='center',verticalalignment='center')
                plt.text(xmin+l, -2*h, self.box_label[1], fontsize=18, horizontalalignment='center',verticalalignment='center')
            plt.axis('equal')
        elif (self.dim == 2):
            xmin = (float)(self.bounds[0][0])
            xmax = (float)(self.bounds[0][1])
            ymin = (float)(self.bounds[1][0])
            ymax = (float)(self.bounds[1][1])
            plt.fill([xmin,xmax,xmax,xmin], [ymin,ymin,ymax,ymax], fill=True, color=plein)
            if viewlabel:
                plt.text(0.5*(xmin+xmax), ymin, self.box_label[0], fontsize=18, horizontalalignment='center',verticalalignment='bottom')
                plt.text(xmax, 0.5*(ymin+ymax), self.box_label[1], fontsize=18, horizontalalignment='right',verticalalignment='center')
                plt.text(0.5*(xmin+xmax), ymax, self.box_label[2], fontsize=18, horizontalalignment='center',verticalalignment='top')
                plt.text(xmin, 0.5*(ymin+ymax), self.box_label[3], fontsize=18, horizontalalignment='left',verticalalignment='center')
            plt.axis([xmin, xmax, ymin, ymax])
            comptelem = 0
            for elem in self.list_elem:
                if elem.isfluid:
                    coul = plein
                else:
                    coul = 'white'
                elem._visualize(ax, coul, viewlabel)
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
        for l in self.box_label:
            L = np.union1d(L, l)
        for elem in self.list_elem:
            for l in elem.label:
                L = np.union1d(L, l)
        #for l in self.list_label:
        #    L = np.union1d(L, l)
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
