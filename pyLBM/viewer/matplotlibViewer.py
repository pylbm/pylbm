# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon

from .base import Viewer

class MatplotlibViewer(Viewer):
    def __init__(self, figsize=(10, 10)):
        self.figsize = figsize
        self.figure()

    def figure(self, figsize=None):
        if figsize is None:
            fig = plt.figure(figsize=self.figsize)
        else:
            fig = plt.figure(figsize=figsize)
        self.ax = fig.add_subplot(111)

    def title(self, t, **kwargs):
        self.ax.set_title(t, **kwargs)

    def text(self, t, pos, fontsize=18, color='k', **kwargs):
        self.ax.text(pos[0], pos[1], t, fontsize=fontsize, **kwargs)

    def line(self, pos,  width=5, color='k', **kwargs):
        self.ax.plot(pos[:, 0], pos[:, 1], c=color, lw=width, **kwargs)

    def segments(self, pos,  width=5, color='k', **kwargs):
        for i in range(pos.shape[0]/2):
            self.ax.plot(pos[2*i:2*i+2, 0], pos[2*i:2*i+2, 1], c=color, lw=width, **kwargs)

    def image(self, data, *args, **kwargs):
        self.ax.imshow(data, *args, **kwargs)

    def clear(self):
        self.ax.clf()

    def axis(self, xmin, xmax, ymin, ymax, zmin=0, zmax=0):
        self.ax.axis([xmin, xmax, ymin, ymax])

    def ellipse(self, pos, radius, color, **kwargs):
        self.ax.add_patch(Ellipse(pos, 2*radius[0], 2*radius[1], fill=True, color=color, **kwargs))

    def polygon(self, pos, color, **kwargs):
        self.ax.add_patch(Polygon(pos, closed=True, fill=True, color=color, **kwargs))

    def markers(self, pos, size, color='k', symbol='o', **kwargs):
        self.ax.scatter(pos[:, 0], pos[:, 1], size, c=color, marker=symbol)

    def draw(self):
        plt.show()

    @property
    def is3d(self):
        return False
