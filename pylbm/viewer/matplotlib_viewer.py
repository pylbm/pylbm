# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause
# FIXME: write the documentation
#pylint: disable=missing-docstring
from six.moves import range

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse, Polygon
from matplotlib import animation
from matplotlib.colors import hsv_to_rgb

import numpy as np

# FIXME: rewrite viewer in order to use it
#        to add multiple viewers.
#from .base import Viewer

class Fig:
    def __init__(self, nrows=1, ncols=1, dim=0, figsize=(6, 4)):
        self.fig = plt.figure(figsize=figsize)
        self._grid = plt.GridSpec(nrows, ncols)
        self._plot_widgets = []
        self.dim = dim

    def fix_space(self, wspace=0.025, hspace=0.05):
        self._grid.update(wspace=wspace, hspace=hspace)

    @property
    def plot_widgets(self):
        """List of the associated PlotWidget instances"""
        return tuple(self._plot_widgets)

    def __getitem__(self, idxs):
        """Get an axis"""
        widget = self._grid.__getitem__(idxs)
        if self.dim < 3:
            widget = PlotWidget(self.fig.add_subplot(self._grid.__getitem__(idxs)))
        else:
            widget = PlotWidget(self.fig.add_subplot(self._grid.__getitem__(idxs), projection='3d'))
        self._plot_widgets += [widget]
        return widget

    #pylint: disable=attribute-defined-outside-init
    def animate(self, func, interval=50):
        self.animation = animation.FuncAnimation(self.fig, func, interval=interval)

    @staticmethod
    def show():
        plt.show()

    def close(self):
        plt.close(self.fig)

#pylint: disable=too-many-public-methods
class PlotWidget:
    def __init__(self, parent):
        self.ax = parent #pylint: disable=invalid-name

    @property
    def title(self):
        return self.ax.get_title()

    @title.setter
    def title(self, text):
        self.ax.set_title(text)

    def legend(self, 
               loc='upper left',
               frameon=True,
               shadow=False
               ):
        self.ax.legend(loc=loc)#, shadow=shadow, frameon=frameon)

    def text(self, text, pos, fontsize=18, fontweight='normal', color='k', horizontalalignment='center', verticalalignment='center'):
        allt = []
        if isinstance(text, str):
            text = (text,)
            pos = (pos,)
        for t, p in zip(text, pos): #pylint: disable=invalid-name
            if len(p) == 2:
                allt.append(self.ax.text(p[0], p[1], t,
                                         fontsize=fontsize, fontweight=fontweight, color=color,
                                         horizontalalignment=horizontalalignment,
                                         verticalalignment=verticalalignment))
            else:
                allt.append(self.ax.text(p[0], p[1], p[2], t,
                                         fontsize=fontsize, fontweight=fontweight, color=color,
                                         horizontalalignment=horizontalalignment,
                                         verticalalignment=verticalalignment))
        return allt

    def line(self, pos, width=2, color='k'):
        return self.ax.plot(pos[:, 0], pos[:, 1], c=color, lw=width)

    def plot(self, x, y, z=None, width=2, color='k',
             label='', marker='', linestyle='-', alpha=1.):
        if z is None:
            return self.ax.plot(x, y, c=color, lw=width, marker=marker, label=label,
                                linestyle=linestyle, alpha=alpha)
        else:
            return self.ax.plot(x, y, z, c=color, lw=width, marker=marker, label=label,
                                linestyle=linestyle, alpha=alpha)

    #pylint: disable=unused-argument
    def segments(self, pos, width=5, color='k', alpha=1., **kwargs):
        if pos.shape[1] == 2:
            for i in range(pos.shape[0]//2):
                self.ax.plot(pos[2*i:2*i+2, 0], pos[2*i:2*i+2, 1], c=color, lw=width, alpha=alpha)
        else:
            for i in range(pos.shape[0]//2):
                self.ax.plot(pos[2*i:2*i+2, 0], pos[2*i:2*i+2, 1], pos[2*i:2*i+2, 2], c=color, lw=width, alpha=alpha)

    def clear(self):
        self.ax.clf()

    def grid(self, visible=True, which='both', alpha=1.):
        self.ax.grid(visible=visible, which=which, alpha=alpha)

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
            if aspect is not None:
                self.ax.set_aspect(aspect)
        if dim == 1:
            self.ax.get_yaxis().set_visible(False)

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
        return self.ax.add_patch(Ellipse(xy=pos, width=2*radius[0], height=2*radius[1],
                                         angle=angle*180/np.pi, fill=True, color=color,
                                         alpha=alpha))

    def polygon(self, pos, color, alpha=1.):
        return self.ax.add_patch(Polygon(pos, closed=True, fill=True, color=color, alpha=alpha))

    def surface(self, x, y, z, color, alpha=0.5):
        return self.ax.plot_surface(x, y, z,
                                    rstride=1, cstride=1, color=color,
                                    shade=False, alpha=alpha,
                                    antialiased=False, linewidth=.5)

    #pylint: disable=invalid-name, too-many-locals
    def ellipse_3d(self, pos, a, b, c, color, alpha=1.):
        u = np.linspace(0, 2.*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        CS = np.outer(np.cos(u), np.sin(v))
        SS = np.outer(np.sin(u), np.sin(v))
        C = np.outer(np.ones(np.size(u)), np.cos(v))
        x = pos[0] + a[0]*CS + b[0]*SS + c[0]*C
        y = pos[1] + a[1]*CS + b[1]*SS + c[1]*C
        z = pos[2] + a[2]*CS + b[2]*SS + c[2]*C
        return self.ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color[0], alpha=alpha)

    def markers(self, pos, size, color='k', symbol='o', alpha=1., dim=None):
        if dim is None:
            dim = pos.shape[1]
        if pos.shape[1] == 2 or dim <= 2:
            return self.ax.scatter(pos[:, 0], pos[:, 1], s=size, c=color, marker=symbol, alpha=alpha)
        else:
            posx, posy, posz = pos[:, 0], pos[:, 1], pos[:, 2]
            return self.ax.scatter(posx, posy, posz, s=size, c=color, marker=symbol, alpha=alpha)

    def image(self, f, fargs=(), cmap='gist_gray', clim=(None, None)):
        if isinstance(f, np.ndarray):
            data = f
        else:
            data = f(*fargs)
        image = self.ax.imshow(data, origin='lower', vmin=clim[0], vmax=clim[1], cmap=cmap, interpolation='nearest')
        return image

    @staticmethod
    def draw():
        plt.show()

    @property
    def is3d(self):
        return False
