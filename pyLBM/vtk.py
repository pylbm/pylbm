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

        if np.all(np.asarray(data.shape) != self.end + 1):
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
            if np.all(np.asarray(d.shape) != self.end + 1):
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

        if self._para:
            self._write_pvtr()

        self.vtkfile.save()

    def _write_pvtr(self):
        dim = 1
        xglob = mpi.COMM_WORLD.gather(self.x.size)
        if self.y is not None:
            dim = 2
            yglob = mpi.COMM_WORLD.gather(self.y.size)
        else:
            yglob = [0]
        if self.z is not None:
            dim = 3
            zglob = mpi.COMM_WORLD.gather(self.z.size)
        else:
            zglob = [0]

        if mpi.COMM_WORLD.Get_rank() == 0:
            gsize = [0]*(2*dim)
            gsize[1] = np.sum(np.asarray(yglob))
            if self.y is not None:
                gsize[3] = np.sum(np.asarray(yglob))
            if self.z is not None:
                gsize[5] = np.sum(np.asarray(zglob))

            pvtr = """<?xml version=\"1.0\"?>
<VTKFile type=\"PRectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">
<PRectilinearGrid WholeExtent=\"{0}\" GhostLevel=\"0\">
<PCoordinates>
<PDataArray NumberOfComponents="1" type="Float64" Name="x_coordinates"/>
<PDataArray NumberOfComponents="1" type="Float64" Name="y_coordinates"/>
<PDataArray NumberOfComponents="1" type="Float64" Name="z_coordinates"/>
</PCoordinates>
<PPointData>
""".format(' '.join(map(str, gsize)), dim)

            for s in self.scalars.keys():
                pvtr += "<PDataArray type=\"Float64\" Name=\"{0}\"/>\n".format(s)
            for v in self.vectors.keys():
                pvtr += "<PDataArray type=\"Float64\" Name=\"{0}\"/>\n".format(v)

            pvtr += "</PPointData>"

            z = [0]*2
            ind = 0
            for k in xrange(self.npz):
                y = [0]*2
                z[1] += zglob[k] - 1
                for j in xrange(self.npy):
                    x = [0]*2
                    y[1] += yglob[j] - 1
                    for i in xrange(self.npx):
                        x[1] += xglob[i] - 1
                        tmp = x + y + z
                        pvtr += """
<Piece Extent=\"{0}\" Source=\"./{1}\"/>
""".format(' '.join(map(str, tmp[:dim*2])), self.filename+'_'+str(ind)+'.vtr')
                        x[0] = x[1] + 1
                        ind += 1
                    y[0] = y[1] + 1
                z[0] = z[1] + 1

            pvtr += """
</PRectilinearGrid>
</VTKFile>
"""
            f = open(self.path + '/' + self.filename + '.pvtr', 'w')
            f.write(pvtr)
            f.close()
