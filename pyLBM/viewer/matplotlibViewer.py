# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.patches import Ellipse, Polygon
import matplotlib.animation as animation
import itertools

import numpy as np

from .base import Viewer

class Fig:
    def __init__(self, nrows=1, ncols=1):
        self.fig = plt.figure()
        self._grid = plt.GridSpec(nrows, ncols)
        self._plot_widgets = []

    @property
    def plot_widgets(self):
        """List of the associated PlotWidget instances"""
        return tuple(self._plot_widgets)

    def __getitem__(self, idxs):
        """Get an axis"""
        pw = self._grid.__getitem__(idxs)
        pw = PlotWidget(self.fig.add_subplot(self._grid.__getitem__(idxs)))
        self._plot_widgets += [pw]
        return pw

    def animate(self, func, interval=50):
        self.animation = animation.FuncAnimation(self.fig, func, interval=interval)

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

    def text(self, text, pos, fontsize=18, color='k', dim=2, horizontalalignment='center', verticalalignment='center'):
        allt = []
        if isinstance(text, str):
            text = (text,)
            pos = (pos,)
        for t, p in zip(text, pos):
            allt.append(self.ax.text(p[0], p[1], t,
                fontsize=fontsize, color=color,
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment))
        return allt

    def line(self, pos, width=2, color='k'):
        return self.ax.plot(pos[:, 0], pos[:, 1], c=color, lw=width)

    def plot(self, x, y, width=2, color='k', label='', marker=''):
        return self.ax.plot(x, y, c=color, lw=width, marker=marker, label=label)

    def segments(self, pos,  width=5, color='k', **kwargs):
        for i in range(pos.shape[0]/2):
            self.ax.plot(pos[2*i:2*i+2, 0], pos[2*i:2*i+2, 1], c=color, lw=width)

    def clear(self):
        self.ax.clf()

    def axis(self, xmin, xmax, ymin, ymax, zmin=0, zmax=0):
        self.ax.axis([xmin, xmax, ymin, ymax])

    def ellipse(self, pos, radius, color):
        return self.ax.add_patch(Ellipse(pos, 2*radius[0], 2*radius[1], fill=True, color=color))

    def polygon(self, pos, color):
        return self.ax.add_patch(Polygon(pos, closed=True, fill=True, color=color))

    def ellipse_3D(self, pos, radius, color):
        u = np.linspace(0, 2.*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = pos[0] + radius*np.outer(np.cos(u), np.sin(v))
        y = pos[1] + radius*np.outer(np.sin(u), np.sin(v))
        z = pos[2] + radius*np.outer(np.ones(np.size(u)), np.cos(v))
        return self.plot_surface(x, y, z, rstride=4, cstride=4, color=color)

        #return self.ax.add_patch(Ellipse(pos, 2*radius[0], 2*radius[1], fill=True, color=color))

    def markers(self, pos, size, color='k', symbol='o'):
        return self.ax.scatter(pos[:, 0], pos[:, 1], size, c=color, marker=symbol)

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
        print frame_number

    f.animate(update)
    plt.show()
    print f.plot_widgets
