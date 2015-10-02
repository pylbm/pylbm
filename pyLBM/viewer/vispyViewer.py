# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import sys
import mpi4py.MPI as mpi
import numpy as np

try:
    import vispy
    minor = int(vispy.__version__.split('.')[1])
    if  minor < 5:
        raise ImportError

    from vispy import scene, app
    from vispy.geometry import Rect
    import vispy.plot as vp
except ImportError:
    print """
Vispy import error

To use vispy backend please install the development version

https://github.com/vispy/vispy
"""
    sys.exit()

from .base import Viewer

class Fig(scene.SceneCanvas):
    def __init__(self, nrows=1, ncols=1, bgcolor='w', size=(800, 600), show=True):
        super(Fig, self).__init__(bgcolor=bgcolor, keys=dict(escape=self.close, F11=self.toggle_fs),
                                  show=show, size=size)
        self._grid = self.central_widget.add_grid()
        self._grid._default_class = PlotWidget
        self._plot_widgets = []
        self.nrows = nrows
        self.ncols = ncols
        self.iframe = 0

    @property
    def plot_widgets(self):
        """List of the associated PlotWidget instances"""
        return tuple(self._plot_widgets)

    def __getitem__(self, idxs):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        if len(idxs) == 1:
            idxs = idxs + (slice(0, self.ncols),)

        new_idxs = ()
        end = [self.nrows, self.ncols]
        for ii, idx in enumerate(idxs):
            if isinstance(idx, slice) and idx.stop is None:
                new_idxs += (slice(idx.start, end[ii], idx.step),)
            else:
                new_idxs += (idx,)

        pw = self._grid.__getitem__(new_idxs)
        self._plot_widgets += [pw]
        return pw

    def toggle_fs(self):
        self.fullscreen = not self.fullscreen

    def close(self):
        sys.exit()

    def update_func(self, event):
        self._func(self.iframe)
        app.process_events()
        self.update()
        self.iframe += 1

    def animate(self, func, interval=50):
        self._func = func
        self.timer = app.Timer(interval=interval*0.001, connect=self.update_func, start=True)
        app.run()

    def show(self):
        app.run()

class PlotWidget(vp.PlotWidget):
    def __init__(self, *args, **kwargs):

        vp.PlotWidget.__init__(self, border_color='k', *args, **kwargs)
        self._title = ''

    def _set_camera(self, cls, *args, **kwargs):
        self._camera_set = True
        self.camera = cls(*args, **kwargs)
        self.camera.set_range(margin=1)

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, text):
        self._title = text
        bounds = self.get_scene_bounds()
        x = .5*(bounds[0][1]-bounds[0][1])
        y = bounds[1][1]+1
        self.add(self.text(text, [x, y, 0.], color='k'))
        #self._set_camera(scene.PanZoomCamera, aspect=1)

    def legend(self, loc=None):
        pass

    def image(self, f, fargs=(), cmap='cubehelix', clim='auto'):
        if isinstance(f, np.ndarray):
            data = f
        else:
            data = f(*fargs)

        image = scene.visuals.Image(data, cmap=cmap, clim=clim)
        self.add(image)
        self._set_camera(scene.PanZoomCamera, aspect=1)
        return image

    def text(self, t, pos, fontsize=18, color='r', dim=2):
        """
        add text on the figure
        """
        text = scene.visuals.Text(t, font_size=fontsize, color=color,
                               pos=pos,
                               anchor_x='center', anchor_y='center')
        self.add(text)
        if dim in [1, 2]:
            self._set_camera(scene.PanZoomCamera)
        elif dim == 3:
            self._set_camera(scene.TurntableCamera)
        return text

    def line(self, pos, width=5, color='w'):
        """
        add line on the figure
        """
        l = scene.visuals.LinePlot(pos, color=color, width=width,
                                   marker_size=0)
        self.add(l)
        self._set_camera(scene.PanZoomCamera)
        return l

    def tube(self, pos, radius=.01):
        t = scene.visuals.Tube(pos, radius)
        self.add(t)
        self._set_camera(scene.TurntableCamera, fov=10)
        return t

    def segments(self, pos, width=5, color='w'):
        """
        add segments on the figure
        """
        l = scene.visuals.LinePlot(pos, color=color, width=width,
                                   marker_size=0, connect='segments')
        self.add(l)
        self._set_camera(scene.PanZoomCamera)
        print 'segments', l.bounds('visual', 0)
        return l

    def ellipse(self, pos, radius, color):
        e = scene.visuals.Ellipse(pos, radius=radius, color=color)
        self.add(e)
        self._set_camera(scene.PanZoomCamera)
        return e

    def polygon(self, pos, color):
        p = scene.visuals.Polygon(pos, color=color)
        self.add(p)
        self._set_camera(scene.PanZoomCamera)
        return p

    def markers(self, pos, size, color='white', symbol='o'):
        m = scene.visuals.Markers()
        m.set_data(pos, symbol, size, face_color=color)
        self.add(m)
        self._set_camera(scene.PanZoomCamera)
        return m

    def axis(self, xmin, xmax, ymin, ymax, zmin=0, zmax=0):
        self._set_camera = scene.PanZoomCamera(Rect(xmin, ymin, xmax-xmin, ymax-ymin))

class VispyViewer(Viewer):
    def __init__(self, figsize=(600, 600), animate=False):
        self.canvas = scene.SceneCanvas(size=figsize, keys=dict(escape=self.close, F11=self.toggle_fs))
        self.view = self.canvas.central_widget.add_view()
        self.visuals = []
        self.visualsText = []
        self.animate = animate

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
            for v in self.visuals[::-1]:
                self.view.add(v)
            for v in self.visualsText:
                self.view.add(v)
            self.visuals = []
            self.visualsText = []
            app.process_events()
            self.canvas.update()
            self.canvas.show()
            if not self.animate:
                app.run()

    def imshow(self, f, fargs=(), cmap='grays', **kwargs):
        if isinstance(f, np.ndarray):
            data = f
        else:
            data = f(*fargs)

        image = scene.visuals.Image(data, cmap=cmap, **kwargs)
        # rotation
        tr = scene.transforms.AffineTransform()
        tr.rotate(180, (1, 1, 0))
        image.transform = tr

        self.view.camera = scene.PanZoomCamera(Rect(0, 0, data.shape[0], data.shape[1]))
        self.visuals.append(image)
        self.canvas.size = data.shape

        return image

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

    # def imshow(self, data, cmap='grays', **kwargs):
    #     i = scene.visuals.Image(data)#, cmap=cmap)
    #     # rotation
    #     tr = scene.transforms.AffineTransform()
    #     tr.rotate(180, (1, 1, 0))
    #     i.transform = tr
    #
    #     self.view.camera = scene.PanZoomCamera(Rect(0, 0, data.shape[0], data.shape[1]))
    #
    #     self.visuals.append(i)

    def is3d(self):
        """
        3d support of the viewer
        """
        pass

if __name__ == '__main__':
    import numpy as np
    f = Fig(2, 2)
    ax = f[0, :]
    x = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.asarray([x, np.sin(x)]).T)
    ax = f[1, 0]
    y = np.linspace(0, 2*np.pi, 100)
    x = x[np.newaxis, :]
    y = y[:, np.newaxis]

    image = ax.image(np.sin(x)*np.sin(y))

    t = 0
    def update(frame_number):
        image.set_data(np.sin(x+frame_number)*np.sin(y))
        print frame_number

    f.animate(update)
    f.show()
    print f.plot_widgets
