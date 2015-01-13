# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import abc

import matplotlib.pyplot as plt
import vtk

class Viewer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def add_text(self, t, x, y, z=0, **kwargs):
        """
        add text on the figure
        """
        return

    @abc.abstractmethod
    def add_line(self, x, y, z, **kwargs):
        """
        add line on th figure
        """
        return

    @abc.abstractmethod
    def draw(self):
        """
        show figure
        """
        return

    @abc.abstractmethod
    def clear(self):
        return

    @abc.abstractmethod
    def axis(self, xmin, xmax, ymin, ymax, zmin, zmax):
        return

    @abc.abstractproperty
    def is3d(self):
        """
        3d support of the viewer
        """
        return

class MatplotlibViewer(Viewer):
    def __init__(self, figsize=(10, 10), fontsize=20):
        self.figsize = figsize
        self.fontsize = fontsize

    def figure(self, title=None):
        fig = plt.figure(figsize=self.figsize)
        self.ax = fig.add_subplot(111)

        if title is not None:
            self.ax.set_title(title)
        self.ax.hold(True)

    def add_text(self, t, x, y, z=0, **kwargs):
        self.ax.text(x, y, t, kwargs, fontsize=self.fontsize)

    def add_line(self, x, y, *args, **kwargs):
        self.ax.plot(x, y, *args, **kwargs)

    def clear(self):
        self.ax.clf()

    def axis(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.ax.axis([xmin, xmax, ymin, ymax])

    def draw(self):
        plt.show()

    @property
    def is3d(self):
        return False

class VtkViewer(Viewer):
    def __init__(self, title = None, figsize=(300, 300), fontsize=20):
        self.ren = []
        self.iren = []
        self.figsize = figsize
        self.fontsize = fontsize

    def figure(self, title=None):
        self.ren.append(vtk.vtkRenderer())

        renwin = vtk.vtkRenderWindow()
        renwin.SetSize(self.figsize[0], self.figsize[1])
        #renwin.SetSize(self.figsize)
        renwin.SetWindowName(title)
        renwin.AddRenderer(self.ren[-1])
        self.iren.append(vtk.vtkRenderWindowInteractor())
        self.iren[-1].SetRenderWindow(renwin)
        renwin.Render()
        self.iren[-1].GetInteractorStyle().SetCurrentStyleToTrackballActor()
        self.iren[-1].GetInteractorStyle().SetCurrentStyleToTrackballCamera()

    def add_text(self, t, x, y, z=0, **kwargs):
        textActor = vtk.vtkTextActor3D()
        textActor.GetTextProperty().SetFontSize(self.fontsize);
        textActor.SetInput(str(t))
        p = 5
        textActor.SetPosition(p*self.fontsize*x, p*self.fontsize*y, p*self.fontsize*z)
        self.ren[-1].AddViewProp(textActor)

    def add_line(self, x, y, *args, **kwargs):
        pass

    def axis(self, xmin, xmax, ymin, ymax, zmin, zmax):
        pass

    def clear(self):
        pass

    def draw(self):
        for r in self.ren:
            r.ResetCamera()
        self.iren[-1].Start();

    @property
    def is3d(self):
        return True
