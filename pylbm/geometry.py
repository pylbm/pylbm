# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Geometry module
"""

import logging
# from textwrap import dedent
# from six import string_types
import numpy as np

from . import viewer
from .validator import validate

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_box(dico):
    """
    return the dimension and the bounds of the box defined in the dictionnary.

    Parameters
    ----------

    dico : dict

    Returns
    -------

    int
        the dimension of the box

    ndarray
        the bounds of the box

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
            err_msg = "'x' interval not found "
            err_msg += "in the box definition of the geometry."
            log.error(err_msg)
    except KeyError:
        err_msg = "'box' key not found "
        err_msg += "in the geometry definition. "
        err_msg += "Check the input dictionnary."
        log.error(err_msg)
    return dim, np.asarray(bounds, dtype='f8')


class Geometry:
    """
    Create a geometry that defines the fluid part and the solid part.

    Parameters
    ----------
    dico : dict
        dictionary that contains the following `key:value`
            - box : a dictionary for the definition of the computed box
            - elements : a list of elements (optional)
    need_validation : bool
        boolean to specify if the dictionary has to be validated (optional)

    Attributes
    ----------
    dim : int
        number of spatial dimensions (1, 2, or 3)
    bounds : ndarray
        the bounds of the box in each spatial direction
    box_label : list
        a list of the four labels for the
            left, right, bottom, top, front, and back edges
    list_elem : list
        a list that contains each element added or deleted in the box

    Notes
    -----

    The dictionary that defines the box should contains the following
    `key:value`

        - x : a list of the bounds in the first direction
        - y : a list of the bounds in the second direction (optional)
        - z : a list of the bounds in the third direction (optional)
        - label : an integer or a list of integers
            (length twice the number of dimensions)
            used to label each edge (optional)

    Examples
    --------

    see demo/examples/geometry/
    """

    def __init__(self, dico, need_validation=True):
        if need_validation:
            # pylint: disable=undefined-variable
            validate(dico, __class__.__name__)

        self.dim, self.bounds = get_box(dico)

        self.list_elem = []

        dummylab = dico['box'].get('label', -1)
        if isinstance(dummylab, int):
            self.box_label = [dummylab]*2*self.dim
        elif isinstance(dummylab, list):
            if len(dummylab) != 2*self.dim:
                err_msg = "The list label of the box has the wrong size "
                err_msg += "(must be 2*dim)"
                log.error(err_msg)
            self.box_label = dummylab
        else:
            log.error("The labels of the box must be an integer or a list")

        self.bounds = np.asarray(self.bounds, dtype='f8')

        log.debug("Message from geometry.py (box_label):\n %d", self.box_label)
        log.debug("Message from geometry.py (bounds):\n %s", self.bounds)

        elem = dico.get('elements', None)
        if elem is not None:
            for elemk in elem:
                if elemk.dim != self.dim:
                    raise ValueError(
                        "Element must have the same dimension of the box"
                    )
                self.list_elem.append(elemk)
        log.debug(self.__str__())

    def __str__(self):
        from .utils import header_string
        from .jinja_env import env
        template = env.get_template('geometry.tpl')
        return template.render(
            header=header_string('Geometry information'),
            geom=self
        )

    def __repr__(self):
        return self.__str__()

    def add_elem(self, elem):
        """
        add a solid or a fluid part in the domain

        Parameters
        ----------

        elem : Element
            a geometric element to add (or to del)

        """
        self.list_elem.append(elem)

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    # pylint: disable=too-complex
    def visualize(self,
                  viewer_app=viewer.matplotlib_viewer,
                  figsize=(6, 4),
                  viewlabel=False,
                  viewgrid=False,
                  fluid_color='navy',
                  alpha=1.):
        """
        plot a view of the geometry

        Parameters
        ----------

        viewer_app : Viewer
            a viewer (default matplotlib_viewer)
        viewlabel : boolean
            activate the labels mark (default False)
        figsize : tuple
            the size of the figure (default (6, 4))
        viewgrid : bool
            view the grid (default False)
        fluid_color : color
            color for the fluid part (default blue)
        alpha : double
            transparency between 0 and 1 (default 1)


        Returns
        -------

        object
            views

        """
        views = viewer_app.Fig(dim=self.dim, figsize=figsize)
        view = views[0]

        if self.dim == 1:
            xmin, xmax = self.bounds[0][:]
            dist = xmax - xmin
            height = dist/20
            length = dist/50
            lpos = np.asarray([[xmin + length, xmin, xmin, xmin + length],
                               [-height, -height, height, height]]).T
            pos = np.asarray([[xmin, xmax], [0, 0]]).T
            rpos = np.asarray([[xmax - length, xmax, xmax, xmax - length],
                               [-height, -height, height, height]]).T
            view.line(lpos, color=fluid_color)
            view.line(rpos, color=fluid_color)
            view.line(pos, color=fluid_color)
            if viewlabel:
                # label 0 for left
                view.text(str(self.box_label[0]), [xmin + length, -2*height])
                # label 1 for right
                view.text(str(self.box_label[1]), [xmax - length, -2*height])
            view.axis(xmin - dist/2, xmax + dist/2, -10*height, 10*height)
            view.yaxis_set_visible(False)
            view.xaxis_set_visible(viewgrid)
        elif self.dim == 2:
            xmin, xmax = self.bounds[0][:]
            ymin, ymax = self.bounds[1][:]
            view.polygon(np.array([[xmin, ymin],
                                   [xmin, ymax],
                                   [xmax, ymax],
                                   [xmax, ymin]]), fluid_color, alpha=alpha)
            if viewlabel:
                # label 0 for left
                view.text(str(self.box_label[0]), [xmin, 0.5*(ymin+ymax)])
                # label 1 for right
                view.text(str(self.box_label[1]), [xmax, 0.5*(ymin+ymax)])
                # label 2 for bottom
                view.text(str(self.box_label[2]), [0.5*(xmin+xmax), ymin])
                # label 3 for top
                view.text(str(self.box_label[3]), [0.5*(xmin+xmax), ymax])
            for elem in self.list_elem:
                if elem.isfluid:
                    color = fluid_color
                    alpha_ = alpha
                else:
                    color = 'white'
                    alpha_ = 1
                elem.visualize(
                    view, color,
                    viewlabel=viewlabel,
                    alpha=alpha_
                )
            xpercent = 0.05*(xmax-xmin)
            ypercent = 0.05*(ymax-ymin)
            view.axis(
                xmin-xpercent,
                xmax+xpercent,
                ymin-ypercent,
                ymax+ypercent,
                aspect='equal'
            )
            view.grid(viewgrid)
        elif self.dim == 3:
            couleurs = [(.5+.5/k, .5/k, 1.-1./k) for k in range(1, 11)]
            pmin = [(float)(self.bounds[k][0]) for k in range(3)]
            pmax = [(float)(self.bounds[k][1]) for k in range(3)]
            xmin, xmax = pmin[0], pmax[0]
            ymin, ymax = pmin[1], pmax[1]
            zmin, zmax = pmin[2], pmax[2]
            ct_lab = 0
            for k in range(3):
                for x0_ in [pmin[k], pmax[k]]:
                    xgrid, ygrid = np.meshgrid(
                        [pmin[(k+1) % 3], pmax[(k+1) % 3]],
                        [pmin[(k+2) % 3], pmax[(k+2) % 3]]
                    )
                    zgrid = x0_ + np.zeros(xgrid.shape)
                    coord = [xgrid, ygrid, zgrid]
                    view.surface(
                        coord[(2-k) % 3],
                        coord[(3-k) % 3],
                        coord[(1-k) % 3],
                        color=couleurs[self.box_label[ct_lab] % 10],
                        alpha=min(alpha, 0.5)
                    )
                    if viewlabel:
                        x = .25*np.sum(coord[(2-k) % 3])
                        y = .25*np.sum(coord[(3-k) % 3])
                        z = .25*np.sum(coord[(1-k) % 3])
                        view.text(
                            str(self.box_label[ct_lab]),
                            [x, y, z],
                            fontsize=18
                        )
                    ct_lab += 1
            view.axis(xmin, xmax, ymin, ymax, zmin, zmax, dim=self.dim)
            if viewlabel:
                view.set_label("X", "Y", "Z")
            for elem in self.list_elem:
                if elem.isfluid:
                    color = fluid_color
                    alpha_ = alpha
                else:
                    color = [
                        couleurs[elem.label[k]]
                        for k in range(elem.number_of_bounds)
                    ]
                    alpha_ = alpha
                elem.visualize(view, color, viewlabel, alpha=alpha_)
        else:
            err_msg = "Error in geometry.visualize(): "
            err_msg += "the dimension {:d} is not allowed".format(self.dim)
            log.error(err_msg)

        views.title = "Geometry"
        views.show()
        return views

    def list_of_labels(self):
        """
        Get the list of all the labels used in the geometry.
        """
        labels = np.unique(self.box_label)
        return np.union1d(labels, self.list_of_elements_labels())

    def list_of_elements_labels(self):
        """
        Get the list of all the labels used in the geometry.
        """
        labels = np.empty(0)
        for elem in self.list_elem:
            labels = np.union1d(labels, elem.label)
        return labels
