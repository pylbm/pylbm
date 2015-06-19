import numpy as np
from pyevtk.vtk import VtkFile, VtkRectilinearGrid
import mpi4py.MPI as mpi
import os

from .logs import setLogger

class VTKFile:
    def __init__(self, filename, path='', npx=1, npy=1, npz=1):
        self._para = False
        prefix = ''
        if mpi.COMM_WORLD.Get_size() != 1:
            self._para = True
            prefix = '_{0}'.format(mpi.COMM_WORLD.Get_rank())

        if not os.path.exists(path):
            os.mkdir(path)

        self.path = path
        self.filename = filename
        self.vtkfile = VtkFile(path + '/' + filename + prefix, VtkRectilinearGrid)
        self.end = np.zeros(3, dtype=np.int)
        self.x = None
        self.y = None
        self.z = None
        self.scalars = {}
        self.vectors = {}
        self._init_grid = False

        self.npx = npx
        self.npy = npy
        self.npz = npz

        self.log = setLogger(__name__)

    def set_grid(self, x, y=None, z=None):
        if self._init_grid:
            self.log.warning("vtk grid redefined.")

        self.x = x
        self.y = y
        self.z = z
        self.dim = 1
        self.end[0] = x.size - 1
        if y is not None:
            self.dim = 2
            self.end[1] = y.size - 1
        if z is not None:
            self.dim = 3
            self.end[2] = z.size - 1

        self.vtkfile.openGrid(start = (0,)*3, end = self.end.tolist())
        self.vtkfile.openPiece(start = (0,)*3, end = self.end.tolist())
        self._init_grid = True

    def add_scalar(self, name, f, *fargs):
        if not self._init_grid:
            self.log.error("""You must define the grid before to add scalar
                            (see set_grid method)""")

        if isinstance(f, np.ndarray):
            data = f
        else:
            data = f(*fargs)

        if np.all(np.asarray(data.shape) != (self.end + 1)[:self.dim]):
            self.log.error("""The shape of the scalar data {0} ({1})
                            must have the shape of the grid ({2})""".format(
                            name, data.shape, self.end + 1
                            ))
        self.scalars[name] = data.ravel(order='F')

    def add_vector(self, name, f, *fargs):
        if not self._init_grid:
            self.log.error("""You must define the grid before to add scalar
                            (see set_grid method)""")

        if isinstance(f, list):
            data = f
        else:
            data = f(*fargs)

        for d in data:
            if np.all(np.asarray(d.shape) != (self.end + 1)[:self.dim]):
                self.log.error("""The shape of each component
                            of the vector data {0}
                            must have the shape of the grid ({1})""".format(
                            name, self.end + 1
                            ))
        tdata = ()
        for d in data:
            tdata += (d.ravel(order='F'),)

        if self.dim == 2:
            tdata += (np.zeros(tdata[0].shape),)
        self.vectors[name] = tdata

    def save(self):
        if not self._init_grid:
            self.log.error("""You must define the grid before save data
                            (see set_grid method)""")

        if len(self.scalars) == 0 and len(self.vectors) == 0:
            self.log.error("""You must provide scalar or vector data
                            to save the vtkfile""")

        # Point data
        self.vtkfile.openData("Point", scalars = self.scalars.keys(),
                               vectors = self.vectors.keys())

        for k, v in self.scalars.items():
            self.vtkfile.addData(k, v)
        for k, v in self.vectors.items():
            self.vtkfile.addData(k, v)

        self.vtkfile.closeData("Point")

        # Coordinates of cell vertices
        self.vtkfile.openElement("Coordinates")
        self.vtkfile.addData("x_coordinates", self.x);
        if self.y is not None:
            self.vtkfile.addData("y_coordinates", self.y);
        if self.z is not None:
            self.vtkfile.addData("z_coordinates", self.z);
        else:
            self.vtkfile.addData("z_coordinates", np.zeros(self.x.shape));
        self.vtkfile.closeElement("Coordinates");

        self.vtkfile.closePiece()
        self.vtkfile.closeGrid()

        for k, v in self.scalars.items():
            self.vtkfile.appendData(data = v)
        for k, v in self.vectors.items():
            self.vtkfile.appendData(data = v)
        if self.y is not None and self.z is not None:
            self.vtkfile.appendData(self.x).appendData(self.y).appendData(self.z)
        elif self.y is not None:
            self.vtkfile.appendData(self.x).appendData(self.y)
        else:
            self.vtkfile.appendData(self.x)

        if self._para and mpi.COMM_WORLD.Get_rank() == 0:
            self._write_pvd()

        self.vtkfile.save()

    def _write_pvd(self):
        pvd = """<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
<Collection>
"""
        for i in xrange(mpi.COMM_WORLD.Get_size()):
            pvd += "<DataSet part=\"{0}\" file=\"./{1}_{0}.vtr\"/>\n".format(i, self.filename)
        pvd +="""</Collection>
</VTKFile>
"""
        f = open(self.path + '/' + self.filename + '.pvd', 'w')
        f.write(pvd)
        f.close()

def write_collection(filename, path, ntime):
    size = mpi.COMM_WORLD.Get_size()
    rank = mpi.COMM_WORLD.Get_rank()

    pvd = """<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
<Collection>
"""
    for nt in xrange(ntime):
        for i in xrange(size):
            pvd += "<DataSet timestep=\"{2}\" part=\"{0}\" file=\"./{1}_{2}_{0}.vtr\"/>\n".format(i, filename, nt)
    pvd +="""</Collection>
</VTKFile>
"""
    f = open(path + '/' + filename + '.pvd', 'w')
    f.write(pvd)
    f.close()
