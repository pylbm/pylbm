from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

from six.moves import range
from six import string_types

import types
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import mpi4py.MPI as mpi

from .elements import *
from .logs import setLogger
from . import viewer
from .validate_dictionary import *

proto_box = {
    'x': (is_2_list_int_or_float,),
    'y': (type(None), is_2_list_int_or_float,),
    'z': (type(None), is_2_list_int_or_float,),
    'label': (type(None), int, is_list_int_or_string) + string_types,
}

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
    log = setLogger(__name__)
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
    return dim, np.asarray(bounds, dtype='f8')

class Geometry(object):
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

    Examples
    --------

    see demo/examples/geometry/
    """

    def __init__(self, dico):
        self.dim, self.bounds = get_box(dico)

        self.list_elem = []
        self.log = setLogger(__name__)

        dummylab = dico['box'].get('label', -1)
        if isinstance(dummylab, int):
            self.box_label = [dummylab]*2*self.dim
        elif isinstance(dummylab, list):
            if len(dummylab) != 2*self.dim:
                self.log.error("The list label of the box has the wrong size (must be 2*dim)")
            self.box_label = dummylab
        else:
            self.log.error("The labels of the box must be an integer or a list")

        self.bounds = np.asarray(self.bounds, dtype='f8')

        self.log.debug("Message from geometry.py (box_label):\n {0}".format(self.box_label))
        self.log.debug("Message from geometry.py (bounds):\n {0}".format(self.bounds))

        elem = dico.get('elements', None)
        if elem is not None:
            for elemk in elem:
                self.list_elem.append(elemk)
        self.log.debug(self.__str__())


    def __str__(self):
        s = "Geometry informations\n"
        s += "\t spatial dimension: {0:d}\n".format(self.dim)
        s += "\t bounds of the box: \n" + self.bounds.__str__() + "\n"
        if (len(self.list_elem) != 0):
            s += "\t List of elements added or deleted in the box\n"
            for k in range(len(self.list_elem)):
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

    def visualize(self,
                  viewer_app=viewer.matplotlibViewer,
                  figsize = (6,4),
                  viewlabel=False,
                  fluid_color='navy',
                  viewgrid=False,
                  alpha = 1.):
        """
        plot a view of the geometry

        Parameters
        ----------

        viewer_app : a viewer (default matplotlibViewer)
        viewlabel : boolean to activate the labels mark (default False)
        fluid_color : color for the fluid part (default blue)

        """
        view = viewer_app.Fig(dim = self.dim, figsize = figsize)
        ax = view[0]

        if self.dim == 1:
            xmin, xmax = self.bounds[0][:]
            L = xmax - xmin
            h = L/20
            l = L/50
            lpos = np.asarray([[xmin+l,xmin,xmin,xmin+l],[-h,-h,h,h]]).T
            pos = np.asarray([[xmin,xmax],[0, 0]]).T
            rpos = np.asarray([[xmax-l,xmax,xmax,xmax-l],[-h,-h,h,h]]).T
            ax.line(lpos, color=fluid_color)
            ax.line(rpos, color=fluid_color)
            ax.line(pos, color=fluid_color)
            if viewlabel:
                # label 0 for left
                ax.text(str(self.box_label[0]), [xmin+l, -2*h])
                # label 1 for right
                ax.text(str(self.box_label[1]), [xmax-l, -2*h])
            ax.axis(xmin - L/2, xmax + L/2, -10*h, 10*h)
            ax.yaxis_set_visible(False)
            ax.xaxis_set_visible(viewgrid)
        elif self.dim == 2:
            xmin, xmax = self.bounds[0][:]
            ymin, ymax = self.bounds[1][:]
            ax.polygon(np.array([[xmin, ymin],
                          [xmin, ymax],
                          [xmax, ymax],
                          [xmax, ymin]]), fluid_color, alpha=alpha)
            if viewlabel:
                # label 0 for left
                ax.text(str(self.box_label[0]), [xmin, 0.5*(ymin+ymax)])
                # label 1 for right
                ax.text(str(self.box_label[1]), [xmax, 0.5*(ymin+ymax)])
                # label 2 for bottom
                ax.text(str(self.box_label[2]), [0.5*(xmin+xmax), ymin])
                # label 3 for top
                ax.text(str(self.box_label[3]), [0.5*(xmin+xmax), ymax])
            for elem in self.list_elem:
                if elem.isfluid:
                    color = fluid_color
                    a = alpha
                else:
                    color = 'white'
                    a = 1
                elem._visualize(ax, color, viewlabel, alpha=a)
            xpercent = 0.05*(xmax-xmin)
            ypercent = 0.05*(ymax-ymin)
            ax.axis(xmin-xpercent, xmax+xpercent, ymin-ypercent, ymax+ypercent, aspect='equal')
            ax.grid(viewgrid)
        elif self.dim == 3:
            couleurs = [(.5+.5/k, .5/k, 1.-1./k) for k in range(1,11)]
            Pmin = [(float)(self.bounds[k][0]) for k in range(3)]
            Pmax = [(float)(self.bounds[k][1]) for k in range(3)]
            xmin, xm, xmax = Pmin[0], .5*(Pmin[0]+Pmax[0]), Pmax[0]
            ymin, ym, ymax = Pmin[1], .5*(Pmin[1]+Pmax[1]), Pmax[1]
            zmin, zm, zmax = Pmin[2], .5*(Pmin[2]+Pmax[2]), Pmax[2]
            ct_lab = 0
            for k in range(3):
                for x0 in [Pmin[k], Pmax[k]]:
                    XS, YS = np.meshgrid([Pmin[(k+1)%3], Pmax[(k+1)%3]],
                                         [Pmin[(k+2)%3], Pmax[(k+2)%3]])
                    ZS = x0 + np.zeros(XS.shape)
                    C = [XS, YS, ZS]
                    ax.surface(C[(2-k)%3], C[(3-k)%3], C[(1-k)%3],
                         color=couleurs[self.box_label[ct_lab]%10], alpha=min(alpha,0.5))
                    if viewlabel:
                        x = .25*np.sum(C[(2-k)%3])
                        y = .25*np.sum(C[(3-k)%3])
                        z = .25*np.sum(C[(1-k)%3])
                        ax.text(str(self.box_label[ct_lab]), [x, y, z], fontsize=18)
                    ct_lab += 1
            ax.axis(xmin,xmax,ymin,ymax,zmin,zmax, aspect='equal')
            ax.set_label("X", "Y", "Z")
            for elem in self.list_elem:
                if elem.isfluid:
                    coul = fluid_color
                    a = alpha
                else:
                    coul = [couleurs[elem.label[k]] for k in range(elem.number_of_bounds)]
                    a = 1
                elem._visualize(ax, coul, viewlabel, alpha=a)
        else:
            self.log.error('Error in geometry.visualize(): the dimension {0} is not allowed'.format(self.dim))

        ax.title = "Geometry"
        view.show()

    def list_of_labels(self):
        """
        Get the list of all the labels used in the geometry.
        """
        L = np.unique(self.box_label)
        return np.union1d(L, self.list_of_elements_labels())

    def list_of_elements_labels(self):
        """
        Get the list of all the labels used in the geometry.
        """
        L = np.empty(0)
        for elem in self.list_elem:
            L = np.union1d(L, elem.label)
        return L
