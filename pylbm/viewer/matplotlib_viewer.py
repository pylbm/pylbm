# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause
# FIXME: write the documentation

# pylint: disable=missing-docstring

import logging
from six.moves import range

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import

import numpy as np

# FIXME: rewrite viewer in order to use it
#        to add multiple viewers.
# from .base import Viewer

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Fig:
    """
    Class Fig

    Parameters
    ----------

    nrows : int
        number of rows (default 1)
    ncols : int
        number of columns (default 1)
    dim : int
        dimension (default 0)
    figsize : tuple
        size of the figure (default (6, 4))

    Returns
    -------

    object :
        a figure object

    Attributes
    ----------

    fig : object
        a plot figure
    dim : int
        the dimension
    """
    def __init__(self, nrows=1, ncols=1, dim=0, figsize=(6, 4)):
        self.fig = plt.figure(figsize=figsize)
        self._grid = plt.GridSpec(ncols=ncols, nrows=nrows, figure=self.fig)
        self._plot_widgets = [[None]*ncols for i in range(nrows)]
        self.dim = dim
        self.shape = (nrows, ncols)

    def fix_space(self, wspace=0.025, hspace=0.05):
        self._grid.update(wspace=wspace, hspace=hspace)

    @property
    def plot_widgets(self):
        """
        List of the associated PlotWidget instances
        """
        return tuple(self._plot_widgets)

    def __getitem__(self, idxs):
        """
        Get an axis
        """
        if isinstance(idxs, tuple):
            coords = idxs
        else:
            coords = (idxs, 0)

        if self._plot_widgets[coords[0]][coords[1]] is None:
            if self.dim < 3:
                widget = PlotWidget(
                    self.fig.add_subplot(self._grid[idxs])
                )
            else:
                widget = PlotWidget(
                    self.fig.add_subplot(
                        self._grid[idxs],
                        projection='3d'
                    )
                )
            self._plot_widgets[coords[0]][coords[1]] = widget
        return self._plot_widgets[coords[0]][coords[1]]

    # pylint: disable=attribute-defined-outside-init
    def animate(self, func, interval=50):
        """
        animate the figure

        Parameters
        ----------

        func : function
            the update function
        interval : int
            the time interval between update (default 50)
        """
        self.animation = animation.FuncAnimation(
            self.fig,
            func,
            interval=interval
        )

    @staticmethod
    def show():
        plt.show()

    def close(self):
        plt.close(self.fig)


class CLine:
    """
    matplotlib object: plot
    used to allowed update function instead of set_data
    """
    def __init__(self, x, y,
                 width=2, color='black',
                 linestyle='solid', alpha=.5,
                 label=''):
        self.x = x
        self.y = y
        self.width = width
        self.color = color
        self.style = linestyle
        self.alpha = alpha
        self.label = label

    # pylint: disable=attribute-defined-outside-init
    def add(self, axe):
        self.line = axe.plot(
            self.x, self.y,
            linestyle=self.style, alpha=self.alpha,
            color=self.color, width=self.width, label=self.label
        )[0]

    def update(self, y):
        self.y = y
        self.line.set_data(self.x, self.y)


class CScatter:
    """
    matplotlib object: scatter
    used to allowed update function instead of set_offsets
    """
    def __init__(self, x, y,
                 size=5, color='black',
                 symbol='o', alpha=.5,
                 label=''):
        self.pos = np.zeros((x.size, 2))
        self.pos[:, 0] = x
        self.pos[:, 1] = y
        self.size = size
        self.color = color
        self.symbol = symbol
        self.alpha = alpha
        self.label = label

    # pylint: disable=attribute-defined-outside-init
    def add(self, axe):
        self.line = axe.markers(
            self.pos, self.size,
            symbol=self.symbol, alpha=self.alpha,
            color=self.color, label=self.label
        )

    def update(self, y):
        self.pos[:, 1] = y
        self.line.set_offsets(self.pos)


class SImage:
    """
    matplotlib object: imshow
    """
    def __init__(self, data, cmap='gist_gray', clim=(None, None), alpha=1):
        self.data = data.T
        self.cmap = cmap
        self.clim = clim
        self.alpha = alpha

    # pylint: disable=attribute-defined-outside-init
    def add(self, axe):
        self.img = axe.image(
            self.data, cmap=self.cmap, clim=self.clim,
            alpha=self.alpha,
        )

    def update(self, data):
        self.data = data.T
        self.img.set_data(self.data)


class SContour:
    """
    matplotlib object: contour
    """
    def __init__(self, data, levels=6, colors='k'):
        self.data = data.T
        self.levels = levels
        self.colors = colors

    # pylint: disable=attribute-defined-outside-init
    def add(self, axe):
        self.contour = axe.contour(
            self.data, levels=self.levels, colors=self.colors
        )
        # axe.axis_equal()
        axe.ax.clabel(self.contour, inline=1, fontsize=6)

    def update(self, data):
        # TODO
        self.data = data.T


class SScatter:
    """
    matplotlib object: 3D scatter
    """
    def __init__(self, x, y, z, size,
                 alpha=0.5, color='black', symbol='o',
                 sampling=1):
        self.size = size
        self.sampling = sampling
        self.alpha = alpha
        self.color = color
        self.symbol = symbol
        mesh_y, mesh_x = np.meshgrid(y[::self.sampling], x[::self.sampling])
        self.x = mesh_x.flatten()
        self.y = mesh_y.flatten()
        self.pos = np.zeros((self.x.size, 3))
        self.pos[:, 0] = self.x
        self.pos[:, 1] = self.y
        self.pos[:, 2] = z[::self.sampling, ::self.sampling].flatten()

    # pylint: disable=attribute-defined-outside-init
    def add(self, axe):
        self.layer = axe.markers(
            self.pos, self.size,
            alpha=self.alpha, color=self.color,
            symbol=self.symbol
        )

    def update(self, z):
        # pylint: disable=protected-access
        self.layer._offsets3d = (
            self.x, self.y,
            z[::self.sampling, ::self.sampling].flatten()
        )


# pylint: disable=too-many-public-methods
class PlotWidget:
    """
    class PlotWidget

    contains an object plotted in the figure
    """
    def __init__(self, parent):
        self.ax = parent  # pylint: disable=invalid-name

    @property
    def title(self):
        """
        get the title of the figure
        """
        return self.ax.title.get_text()

    @title.setter
    def title(self, text):
        """
        set the title of the figure

        Parameters
        ----------

        text : string
            the title
        """
        self.ax.title.set_text(text)

    def legend(self,
               loc='upper left',
               frameon=True,
               shadow=False
               ):
        """
        set the legend of the figure

        Parameters
        ----------

        loc : string
            the location (default 'upper left')
            allowed values: 'best',
            'upper left', 'upper center', 'upper right',
            'lower left', 'lower center', 'lower right'
        frameon : bool
            activate or desactivate the frame (default True)
        shadow : bool
            activate or desactivate the shadow (default False)
        """
        self.ax.legend(loc=loc, shadow=shadow, frameon=frameon)

    def text(self, text, pos,
             color='black',
             fontsize=18, fontweight='normal',
             horizontalalignment='center',
             verticalalignment='center'
             ):
        """
        add a text widget on the figure

        Parameters
        ----------

        text : string or tuple
            the text or the texts
        pos : list or tuple
            the position (2 or 3 coordinates)
        color : string or tuple
            the color of the text (default 'black')
            could be a tuple as a RGB color
        fontsize : int
            the size of the font (default 18)
        fontweight : string
            the weight of the font (default 'normal')
        horizontalalignment : string
            the horizontal alignment (default 'center')
        verticalalignment : string
            the vertical alignment (default 'center)

        Returns
        -------

        list
            the list of the text widgets
        """
        allt = []
        if isinstance(text, str):
            text = (text,)
            pos = (pos,)
        for t, p in zip(text, pos):  # pylint: disable=invalid-name
            if len(p) == 2:
                allt.append(
                    self.ax.text(
                        p[0], p[1], t,
                        color=color,
                        fontsize=fontsize,
                        fontweight=fontweight,
                        horizontalalignment=horizontalalignment,
                        verticalalignment=verticalalignment
                    )
                )
            else:
                allt.append(
                    self.ax.text(
                        p[0], p[1], p[2], t,
                        color=color,
                        fontsize=fontsize,
                        fontweight=fontweight,
                        horizontalalignment=horizontalalignment,
                        verticalalignment=verticalalignment
                    )
                )
        return allt

    def line(self, pos,
             width=2, color='black', style='solid',
             alpha=0.5, label=''
             ):
        """
        a line widget

        Parameters
        ----------

        pos : ndarray
            the coordinates of the line (2 or 3 D)

        width : int
            the width of the line (default 2)

        color : string or tuple
            the color of the line (default 'black')
            could be a tuple as a RGB color

        style : string
            the style of the line (default 'solid')

        alpha : transparency
            the transparency of the line (default 0.5)

        label : string
            the label of the line (default '')

        Returns
        -------

        object
            a graphical object

        """
        if pos.shape[1] == 2:
            return self.ax.plot(
                pos[:, 0], pos[:, 1],
                linestyle=style, alpha=alpha,
                c=color, lw=width, label=label
            )
        elif pos.shape[1] == 3:
            return self.ax.plot(
                pos[:, 0], pos[:, 1], pos[:, 2],
                linestyle=style, alpha=alpha,
                c=color, lw=width, label=label
            )
        else:
            err_msg = "Problem of dimension in viewer.line: "
            err_msg += "pos should be a ndarray of shape (N, d) "
            err_msg += "with d=2 or d=3 "
            err_msg += "(d={:d} not allowed)".format(pos.shape[1])
            log.error(err_msg)
            return

    def plot(self, x, y, z=None,
             width=2, color='black',
             marker='', linestyle='solid', alpha=1.,
             label=''
             ):
        """
        a plot widget
        """
        if z is None:
            return self.ax.plot(
                x, y, c=color,
                lw=width, marker=marker,
                label=label,
                linestyle=linestyle, alpha=alpha
            )
        else:
            return self.ax.plot(
                x, y, z, c=color,
                lw=width, marker=marker,
                label=label,
                linestyle=linestyle, alpha=alpha
            )

    # pylint: disable=unused-argument
    def segments(self, pos, width=5, color='k', alpha=1., **kwargs):
        if pos.shape[1] == 2:
            for i in range(pos.shape[0]//2):
                self.ax.plot(
                    pos[2*i:2*i+2, 0], pos[2*i:2*i+2, 1],
                    c=color, lw=width, alpha=alpha
                )
        else:
            for i in range(pos.shape[0]//2):
                self.ax.plot(
                    pos[2*i:2*i+2, 0], pos[2*i:2*i+2, 1], pos[2*i:2*i+2, 2],
                    c=color, lw=width, alpha=alpha
                )

    def clear(self):
        self.ax.clf()

    def grid(self, visible=True, which='both', alpha=1.):
        # ISSUE with MATPLOTLIB
        # https://github.com/matplotlib/matplotlib/issues/18758
        # self.ax.grid(visible=visible, which=which, alpha=alpha)
        self.ax.grid(visible, which=which, alpha=alpha)

    def axis(self, xmin, xmax, ymin, ymax, zmin=0, zmax=0, dim=2, aspect=None):
        if (zmin == 0 and zmax == 0) or dim <= 2:
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            if aspect is not None:
                self.ax.set_aspect(aspect)
        else:
            self.ax.set_xlim3d(xmin, xmax)
            self.ax.set_ylim3d(ymin, ymax)
            self.ax.set_zlim3d(zmin, zmax)
        if dim == 1:
            self.ax.get_yaxis().set_visible(False)

    def axis_equal(self):
        self.ax.set_aspect('equal')

    def xaxis_set_visible(self, visible):
        self.ax.get_xaxis().set_visible(visible)

    def xaxis(self, major_ticks, minor_ticks=None):
        self.ax.set_xticks(major_ticks)
        if minor_ticks is not None:
            self.ax.set_xticks(minor_ticks, minor=True)

    def yaxis_set_visible(self, visible):
        self.ax.get_yaxis().set_visible(visible)

    def yaxis(self, major_ticks, minor_ticks=None):
        self.ax.set_yticks(major_ticks)
        if minor_ticks is not None:
            self.ax.set_yticks(minor_ticks, minor=True)

    def zaxis_set_visible(self, visible):
        self.ax.get_zaxis().set_visible(visible)

    def zaxis(self, major_ticks, minor_ticks=None):
        self.ax.set_zticks(major_ticks)
        if minor_ticks is not None:
            self.ax.set_zticks(minor_ticks, minor=True)

    def set_label(self, xlab, ylab, zlab=None):
        if xlab is not None:
            self.ax.set_xlabel(xlab)
        if ylab is not None:
            self.ax.set_ylabel(ylab)
        if zlab is not None:
            self.ax.set_zlabel(zlab)

    def ellipse(self, pos, radius, color, angle=0., alpha=1):
        return self.ax.add_patch(
            Ellipse(
                xy=pos,
                width=2*radius[0], height=2*radius[1],
                angle=angle*180/np.pi,
                fill=True, color=color, alpha=alpha,
                zorder=0
            )
        )

    def polygon(self, pos, color, alpha=1.):
        return self.ax.add_patch(
            Polygon(
                pos, closed=True, fill=True, color=color, alpha=alpha,
                zorder=0
            )
        )

    def surface(self, x, y, z, color, alpha=0.5):
        return self.ax.plot_surface(
            x, y, z,
            rstride=4, cstride=4, color=color,
            shade=False, alpha=alpha,
            antialiased=False, linewidth=0., zorder=0
        )

    # pylint: disable=invalid-name, too-many-locals
    def ellipse_3d(self, pos, a, b, c, color, alpha=1.):
        u = np.linspace(0, 2.*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        CS = np.outer(np.cos(u), np.sin(v))
        SS = np.outer(np.sin(u), np.sin(v))
        C = np.outer(np.ones(np.size(u)), np.cos(v))
        x = pos[0] + a[0]*CS + b[0]*SS + c[0]*C
        y = pos[1] + a[1]*CS + b[1]*SS + c[1]*C
        z = pos[2] + a[2]*CS + b[2]*SS + c[2]*C
        return self.ax.plot_surface(
            x, y, z,
            rstride=4, cstride=4,
            color=color, alpha=alpha, zorder=0
        )

    def markers(self, pos, size, color='k',
                symbol='o', alpha=1., dim=None, label=''
                ):
        if dim is None:
            dim = pos.shape[1]
        if pos.shape[1] == 2 or dim <= 2:
            return self.ax.scatter(
                pos[:, 0], pos[:, 1],
                s=size, c=color, marker=symbol,
                alpha=alpha, label=label
            )
        else:
            posx, posy, posz = pos[:, 0], pos[:, 1], pos[:, 2]
            return self.ax.scatter(
                posx, posy, posz,
                c=color, marker=symbol, s=size,
                alpha=alpha, label=label
            )

    def image(self, f, fargs=(), cmap='gist_gray', clim=(None, None), alpha=1):
        if isinstance(f, np.ndarray):
            data = f
        else:
            data = f(*fargs)
        image = self.ax.imshow(
            data, origin='lower',
            vmin=clim[0], vmax=clim[1],
            cmap=cmap, interpolation='nearest',
            alpha=alpha,
        )
        return image

    def contour(self, Z, levels=10, colors='k'):
        contour = self.ax.contour(Z, levels=levels, colors=colors)
        return contour


    @staticmethod
    def draw():
        plt.show()

    @property
    def is3d(self):
        return False

    def CurveLine(self, x, y,
                  width=2, color='black',
                  linestyle='solid', alpha=.5,
                  label=''):
        line = CLine(
            x, y,
            width=width, color=color,
            linestyle=linestyle, alpha=alpha,
            label=label
        )
        line.add(self)
        return line

    def CurveScatter(self, x, y, size=5, color='black',
                     symbol='o', alpha=.5,
                     dim=None, label=''
                     ):
        line = CScatter(
            x, y,
            size=size, color=color,
            symbol=symbol, alpha=alpha,
            label=label
        )
        line.add(self)
        return line

    def SurfaceImage(self, data, cmap='gist_gray', clim=(None, None), alpha=1):
        layer = SImage(data, cmap=cmap, clim=clim, alpha=alpha)
        layer.add(self)
        return layer

    def SurfaceContour(self, data, levels=6, colors='k'):
        layer = SContour(data, levels=levels, colors=colors)
        layer.add(self)
        return layer

    def SurfaceScatter(self, x, y, z, size, sampling=1,
                       alpha=0.5, color='black', symbol='o'
                       ):
        layer = SScatter(
            x, y, z, size, sampling=sampling,
            alpha=alpha, color=color, symbol=symbol
        )
        layer.add(self)
        return layer
