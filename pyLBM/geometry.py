# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import matplotlib.pyplot as plt

import mpi4py.MPI as mpi

from .elements import *
from .interface import Interface
from .logs import __setLogger
log = __setLogger(__name__)

def get_box(dico):
    """
    return the dimension and the bounds of the box defined in the dictionnary.

    Parameters
    ----------

    dico : a dictionnary

    Returns
    -------

    dim : the dimension of the box
    bounds: the bounds of the box
    """
    try:
        box = dico['box']
        try:
            bounds = [box['x']]
            dim = 1
            boxy = box.get('y', None)
            if boxy is not None:
                bounds.append(boxy)
                dim += 1
                boxz = box.get('z', None)
                if boxz is not None:
                    bounds.append(boxz)
                    dim += 1
        except KeyError:
            log.error("'x' interval not found in the box definition of the geometry.")
    except KeyError:
        log.error("'box' key not found in the geometry definition. Check the input dictionnary.")
    return dim, bounds

class Geometry:
    """
    Create a geometry that defines the fluid part and the solid part.

    Parameters
    ----------

    dico : a dictionary that contains the following `key:value`
      - box : a dictionary for the definition of the computed box
      - elements : a list of elements (optional)

    Notes
    -----

    The dictionary that defines the box should contains the following `key:value`
      - x : a list of the bounds in the first direction
      - y : a list of the bounds in the second direction (optional)
      - z : a list of the bounds in the third direction (optional)
      - label : an integer or a list of integers (length twice the number of dimensions) used to label each edge (optional)

    Attributes
    ----------

    dim : int
      number of spatial dimensions (1, 2, or 3)
    bounds : numpy array
      the bounds of the box in each spatial direction
    box_label : list of integers
      a list of the four labels for the left, right, bottom, top, front, and back edges
    list_elem : list of elements
      a list that contains each element added or deleted in the box

    Methods
    -------

    add_elem :
      function that adds an element in the box
    visualize :
      function to visualize the box
    list_of_labels :
      return a list of all the unique labels of the geometry

    Examples
    --------

    see demo/examples/geometry/
    """

    def __init__(self, dico):
        self.dim, self.globalbounds = get_box(dico)

        self.list_elem = []

        dummylab = dico['box'].get('label', 0)
        if isinstance(dummylab, int):
            self.box_label = [dummylab]*2*self.dim
        elif isinstance(dummylab, list):
            if len(dummylab) != 2*self.dim:
                log.error("The list label of the box has the wrong size (must be 2*dim)")
            self.box_label = dummylab
        else:
            log.error("The labels of the box must be an integer or a list")

        period = [False]*self.dim
        for i in xrange(self.dim):
            if self.box_label[2*i] == self.box_label[2*i+1] == -1: # work only for dim = 2
                period[i] = True

        self.interface = Interface(self.dim, period)

        self.globalbounds = np.asarray(self.globalbounds, dtype='f8')
        self.bounds = self.globalbounds.copy()

        t = (self.bounds[:, 1] - self.bounds[:, 0])/self.interface.split
        coords = self.interface.get_coords()
        self.bounds[:, 1] = self.bounds[:, 0] + t*(coords + 1)
        self.bounds[:, 0] = self.bounds[:, 0] + t*coords

        # Modify box_label if the border becomes an interface
        for i in xrange(self.dim):
            voisins = self.interface.comm.Shift(i, 1)
            if voisins[0] != mpi.PROC_NULL:
                self.box_label[2*i] = -2
            if voisins[1] != mpi.PROC_NULL:
                self.box_label[2*i + 1] = -2

        log.debug("Message from geometry.py (box_label):\n {0}".format(self.box_label))
        log.debug("Message from geometry.py (bounds):\n {0}".format(self.bounds))

        elem = dico.get('elements', None)
        if elem is not None:
            for elemk in elem:
                self.list_elem.append(elemk)
        log.debug(self.__str__())


    def __str__(self):
        s = "Geometry informations\n"
        s += "\t spatial dimension: {0:d}\n".format(self.dim)
        s += "\t bounds of the box: \n" + self.bounds.__str__() + "\n"
        if (len(self.list_elem) != 0):
            s += "\t List of elements added or deleted in the box\n"
            for k in xrange(len(self.list_elem)):
                s += "\t\t Element number {0:d}: ".format(k) + self.list_elem[k].__str__() + "\n"
        return s

    def add_elem(self, elem):
        """
        add a solid or a fluid part in the domain

        Parameters
        ----------

        elem : a geometric element to add (or to del)
        """
        self.list_elem.append(elem)

    def visualize(self, viewlabel=False):
        """
        plot a view of the geometry

        Parameters
        ----------

        viewlabel : boolean to activate the labels mark (optional)
        """
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
                # label 0 for left
                plt.text(xmin+l, -2*h, self.box_label[0], fontsize=18, horizontalalignment='center',verticalalignment='center')
                # label 1 for right
                plt.text(xmax-l, -2*h, self.box_label[1], fontsize=18, horizontalalignment='center',verticalalignment='center')
            plt.axis('equal')
        elif (self.dim == 2):
            xmin = (float)(self.bounds[0][0])
            xmax = (float)(self.bounds[0][1])
            ymin = (float)(self.bounds[1][0])
            ymax = (float)(self.bounds[1][1])
            plt.fill([xmin,xmax,xmax,xmin], [ymin,ymin,ymax,ymax], fill=True, color=plein)
            if viewlabel:
                # label 0 for left
                plt.text(xmin, 0.5*(ymin+ymax), self.box_label[0], fontsize=18, horizontalalignment='left',verticalalignment='center')
                # label 1 for right
                plt.text(xmax, 0.5*(ymin+ymax), self.box_label[1], fontsize=18, horizontalalignment='right',verticalalignment='center')
                # label 2 for bottom
                plt.text(0.5*(xmin+xmax), ymin, self.box_label[2], fontsize=18, horizontalalignment='center',verticalalignment='bottom')
                # label 3 for top
                plt.text(0.5*(xmin+xmax), ymax, self.box_label[3], fontsize=18, horizontalalignment='center',verticalalignment='top')
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
        Get the list of all the labels used in the geometry.
        """
        L = np.unique(self.box_label)
        for elem in self.list_elem:
            L = np.union1d(L, elem.label)
        return L


def test_1D(number):
    """
    Test 1D-Geometry

    * ``Test_1D(0)`` for the segment (0,1)
    * ``Test_1D(1)`` for the segment (-1,2)
    """
    if number == 0:
        dgeom = {'box':{'x': [0, 1]}}
    elif number == 1:
        dgeom = {'box':{'x': [-1, 2]}}
    else:
        dgeom = None

    if dgeom is not None:
        geom = Geometry(dgeom)
        print "\n\nTest number {0:d} in {1:d}D:".format(number, geom.dim)
        print geom
        geom.visualize()
        return 1
    else:
        return 0

def test_2D(number):
    """
    Test 2D-Geometry

    * ``Test_2D(0)`` for the square [0,1]**2
    * ``Test_2D(1)`` for the rectangular cavity with a circular obstacle
    * ``Test_2D(2)`` for the circular cavity
    * ``Test_2D(3)`` for the square cavity with a triangular obstacle
    """
    if number == 0:
        dgeom = {'box':{'x': [0, 1], 'y': [0, 1]}}
    elif number == 1:
        dgeom = {'box':{'x': [0, 2], 'y': [0, 1]},
                 'elements':[Circle((0.5, 0.5), 0.1)]
                }
    elif number == 2:
        dgeom = {'box':{'x': [0, 2], 'y': [0, 1]},
                 'elements':[Parallelogram((0, 0), (2, 0), (0, 1)),
                             Parallelogram((0, .4), (2, 0), (0, .2), isfluid=True),
                             Circle((1, .5), 0.5, isfluid=True),
                             Circle((1, .5), 0.2, isfluid=False)
                             ]
                }
    elif number == 3:
        dgeom = {'box':{'x': [0, 1], 'y': [0, 1]},
                 'elements':[Triangle((0.3, 0.3), (0.5, -0.1), (0.3, 0.5))]}
    elif (number==4):
        dgeom = {'box':{'x': [0, 2], 'y': [0, 1]},
                 'elements':[Parallelogram((0.4, 0.4), (0., 0.2), (0.2, 0.)),
                             Parallelogram((1.4, 0.5), (0.1, 0.1), (0.1, -0.1))
                            ]
                }
    else:
        dgeom = None
    if dgeom is not None:
        geom = Geometry(dgeom)
        print "\n\nTest number {0:d} in {1:d}D:".format(number, geom.dim)
        print geom
        geom.visualize()
        return 1
    else:
        return 0

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
