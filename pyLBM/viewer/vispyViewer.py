# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

from vispy import scene, app
from vispy.geometry import Rect
import mpi4py.MPI as mpi
import numpy as np
import sys

from .base import Viewer

class VispyViewer(Viewer):
    def __init__(self, figsize=(600, 600)):
        if mpi.COMM_WORLD.Get_rank() == 0:
            self.canvas = scene.SceneCanvas(size=figsize, keys=dict(escape=self.close, F11=self.toggle_fs))
            self.view = self.canvas.central_widget.add_view()
            self.init_imshow = True
            self.plot = None
            self.parent = self.view.scene
            self.visuals = []
            self.visualsText = []
            self.init = True

    def toggle_fs(self):
        self.canvas.fullscreen = not self.canvas.fullscreen

    def close(self):
        self.canvas.close()
        sys.exit()

    def text(self, t, pos, fontsize=18, color='r', **kwargs):
        """
        add text on the figure
        """
        text = scene.visuals.Text(t, font_size=fontsize, color=color,
                               pos=pos,
                               anchor_x='center', anchor_y='center',
                               **kwargs)
        self.visualsText.append(text)

    def line(self, pos, width=5, color='w', **kwargs):
        """
        add line on the figure
        """
        l = scene.visuals.LinePlot(pos, color=color, width=width,
                                   marker_size=0, **kwargs)
        self.visuals.append(l)

    def segments(self, pos, width=5, color='w', **kwargs):
        """
        add segments on the figure
        """
        l = scene.visuals.LinePlot(pos, color=color, width=width,
                                   marker_size=0, connect='segments', **kwargs)
        self.visuals.append(l)

    def draw(self):
        """
        show figure
        """
        if mpi.COMM_WORLD.Get_rank() == 0:
            # @self.plot.events.connect
            # def set_events(event):
            #     print type(event)
            if self.init:
                for v in self.visuals[::-1]:
                    self.view.add(v)
                for v in self.visualsText:
                    self.view.add(v)
                self.init = False
            app.process_events()
            self.canvas.update()
            self.canvas.show()
            app.run()

    def imshow(self, f, fargs=()):
        if isinstance(f, np.ndarray):
            data = f
        else:
            data = f(*fargs)

        if mpi.COMM_WORLD.Get_rank() == 0:
            if self.init_imshow:
                self.plot = scene.visuals.Image(data, cmap = 'grays', parent=self.view.scene)
                self.init_imshow = False
                self.canvas.size = data.shape
                self.canvas.show()
            else:
                self.plot.set_data(data)

            # rotation
            tr = scene.transforms.AffineTransform()
            tr.rotate(180, (1, 1, 0))
            self.plot.transform = tr

            self.view.camera = scene.PanZoomCamera(Rect(0, 0, data.shape[0], data.shape[1]))

    def title(self, title):
        if mpi.COMM_WORLD.Get_rank() == 0:
            self.canvas.title = title

    def clear(self):
        pass

    def axis(self, xmin, xmax, ymin, ymax, zmin=0, zmax=0):
        self.view.camera = scene.PanZoomCamera(Rect(xmin, ymin, xmax-xmin, ymax-ymin))

    def ellipse(self, pos, radius, color, **kwargs):
        e = scene.visuals.Ellipse(pos, radius=radius, color=color, **kwargs)
        self.visuals.append(e)

    def polygon(self, pos, color, **kwargs):
        p = scene.visuals.Polygon(pos, color=color, **kwargs)
        self.visuals.append(p)

    def markers(self, pos, size, color='white', symbol='o', **kwargs):
        m = scene.visuals.Markers()
        m.set_data(pos, symbol, size, face_color=color)
        self.visuals.append(m)

    def is3d(self):
        """
        3d support of the viewer
        """
        pass
