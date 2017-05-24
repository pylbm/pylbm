# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

from __future__ import print_function
from __future__ import division
from six.moves import range

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.patches import Ellipse, Polygon
import matplotlib.animation as animation
import itertools

import numpy as np

from .base import Viewer

class Fig(object):
    def __init__(self, nrows=1, ncols=1, dim = 0):
        self.fig = plt.figure()
        self._grid = plt.GridSpec(nrows, ncols)
        self._plot_widgets = []
        self.dim = dim

    @property
    def plot_widgets(self):
        """List of the associated PlotWidget instances"""
        return tuple(self._plot_widgets)

    def __getitem__(self, idxs):
        """Get an axis"""
        pw = self._grid.__getitem__(idxs)
        if self.dim < 3:
            pw = PlotWidget(self.fig.add_subplot(self._grid.__getitem__(idxs)))
        else:
            pw = PlotWidget(self.fig.add_subplot(self._grid.__getitem__(idxs), projection='3d'))
        self._plot_widgets += [pw]
        return pw

    def animate(self, func, interval=50, fargs=None):
        self.animation = animation.FuncAnimation(self.fig, func, interval=interval, fargs = fargs)

    def show(self):
        plt.show()

class PlotWidget(object):
    def __init__(self, parent):
        self.ax = parent

    @property
    def title(self):
        return self.ax.get_title()

    @title.setter
    def title(self, text):
        self.ax.set_title(text)

    def legend(self, loc = 'upper left'):
        self.ax.legend(loc = loc)

    def text(self, text, pos, fontsize=18, color='k', horizontalalignment='center', verticalalignment='center'):
        allt = []
        if isinstance(text, str):
            text = (text,)
            pos = (pos,)
        for t, p in zip(text, pos):
            if len(p) == 2:
                allt.append(self.ax.text(p[0], p[1], t,
                    fontsize=fontsize, color=color,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment))
            else:
                allt.append(self.ax.text(p[0], p[1], p[2], t,
                    fontsize=fontsize, color=color,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment))
        return allt

    def line(self, pos, width=2, color='k'):
        return self.ax.plot(pos[:, 0], pos[:, 1], c=color, lw=width)

    def plot(self, x, y, width=2, color='k', label='', marker='', linestyle='-'):
        return self.ax.plot(x, y, c=color, lw=width, marker=marker, label=label, linestyle=linestyle)

    def segments(self, pos,  width=5, color='k', **kwargs):
        if pos.shape[1] == 2:
            for i in range(pos.shape[0]//2):
                self.ax.plot(pos[2*i:2*i+2, 0], pos[2*i:2*i+2, 1], c=color, lw=width)
        else:
            for i in range(pos.shape[0]//2):
                self.ax.plot(pos[2*i:2*i+2, 0], pos[2*i:2*i+2, 1], pos[2*i:2*i+2, 2], c=color, lw=width)

    def clear(self):
        self.ax.clf()

    def axis(self, xmin, xmax, ymin, ymax, zmin=0, zmax=0, dim=2):
        if (zmin == 0 and zmax == 0) or dim == 2:
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
        else:
            self.ax.set_xlim3d(xmin, xmax)
            self.ax.set_ylim3d(ymin, ymax)
            self.ax.set_zlim3d(zmin, zmax)

    def set_label(self, xlab, ylab, zlab=None):
        self.ax.set_xlabel(xlab)
        self.ax.set_ylabel(ylab)
        if zlab is not None:
            self.ax.set_zlabel(zlab)

    def ellipse(self, pos, radius, color, angle=0.):
        return self.ax.add_patch(Ellipse(xy = pos, width = 2*radius[0], height = 2*radius[1], angle=angle*180/np.pi, fill=True, color=color))

    def polygon(self, pos, color):
        return self.ax.add_patch(Polygon(pos, closed=True, fill=True, color=color))

    def surface(self, X, Y, Z, color):
        return self.ax.plot_surface(X, Y, Z,
            rstride=1, cstride=1, color=color,
            shade=False, alpha=0.5,
            antialiased=False, linewidth=.5)

    def ellipse_3D(self, pos, a, b, c, color):
        u = np.linspace(0, 2.*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        CS = np.outer(np.cos(u), np.sin(v))
        SS = np.outer(np.sin(u), np.sin(v))
        C = np.outer(np.ones(np.size(u)), np.cos(v))
        x = pos[0] + a[0]*CS + b[0]*SS + c[0]*C
        y = pos[1] + a[1]*CS + b[1]*SS + c[1]*C
        z = pos[2] + a[2]*CS + b[2]*SS + c[2]*C
        return self.ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color[0])

    def markers(self, pos, size, color='k', symbol='o'):
        if pos.shape[1] == 2:
            return self.ax.scatter(pos[:, 0], pos[:, 1], size, c=color, marker=symbol)
        else:
            posx, posy, posz = pos[:,0], pos[:,1], pos[:,2]
            return self.ax.scatter(posx, posy, posz, s=size, c=color, marker=symbol)

    def image(self, f, fargs=(), cmap='gist_gray', clim=[None, None]):
        if isinstance(f, np.ndarray):
            data = f
        else:
            data = f(*fargs)
        image = self.ax.imshow(data, origin='lower', vmin=clim[0], vmax=clim[1], cmap=cmap, interpolation='nearest')
        return image

    def draw(self):
        plt.show()

    @property
    def is3d(self):
        return False

if __name__ == '__main__':
    import numpy as np
    f = Fig(2, 2)
    ax = f[0, :]
    x = np.linspace(0, 2*np.pi, 100)
    ax.plot(x, np.sin(x))
    ax = f[1, 0]
    y = np.linspace(0, 2*np.pi, 100)
    x = x[np.newaxis, :]
    y = y[:, np.newaxis]

    image = ax.imshow(np.sin(x)*np.sin(y))

    t = 0
    def update(frame_number):
        image.set_data(np.sin(x+frame_number)*np.sin(y))
        print(frame_number)

    f.animate(update)
    plt.show()
    print(f.plot_widgets)
