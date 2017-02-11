from six.moves import range
import numpy as np
import h5py
import mpi4py.MPI as mpi
import os

from .logs import setLogger

class H5File(object):
    def __init__(self, mpi_topo, filename, path='', timestep=0, init_xdmf=False):
        self.timestep = timestep
        prefix = '_{}'.format(timestep)
        self.path = path
        self.filename = filename + prefix
        self.h5filename = filename + prefix + '.h5'

        if mpi.COMM_WORLD.Get_rank()==0:
            if not os.path.exists(path):
                os.mkdir(path)

            self.h5file = h5py.File(path + '/' + self.h5filename, "w")
            
        # All the processes wait for the creation of the output directory
        mpi.COMM_WORLD.Barrier()

        self.mpi_topo = mpi_topo
        self.scalars = {}
        self.vectors = {}
        self._init_grid = True
        self._init_xdmf = init_xdmf

        self.log = setLogger(__name__)

    def set_grid(self, x, y=None, z=None):
        if not self._init_grid:
            self.log.warning("h5 grid redefined.")

        self.origin = [x[0]]
        self.dx = [x[1] - x[0]]
        if x is not None:
            self.origin.append(y[0])
            self.dx.append(y[1] - y[0])
        if z is not None:
            self.origin.append(z[0])
            self.dx.append(z[1] - z[0])

        self.dim = len(self.origin)
        self.n = [0]*self.dim
        self.region = []
        self.global_size = []
        coords = [x]
        if y is not None:
            coords.append(y)
        if z is not None:
            coords.append(z)

        for i in range(len(coords)):
            sub = [False]*len(coords)
            sub[i] = True
            comm = self.mpi_topo.cartcomm.Sub(sub)
            self.n[i] = comm.allreduce(coords[i].size, op=mpi.SUM)
            self.region.append(np.asarray(comm.gather(coords[i].size, root=0)))
            self.global_size.append(np.sum(self.region[i]))
            coords[i] = np.asarray(comm.gather(coords[i], root=0))

        if mpi.COMM_WORLD.Get_rank() == 0:
            self.region = np.asarray(self.region, dtype=np.int)
            print(np.zeros((self.dim, 1)), self.region)
            self.region = np.concatenate((np.zeros((self.dim, 1), dtype=np.int), np.cumsum(self.region, axis=1)), axis=1)
            print(self.region, self.global_size)
            for i in range(self.dim):
                dset = self.h5file.create_dataset("x_{}".format(i), [self.global_size[i]], dtype=np.double)
                dset[:] = coords[i].flatten()

    def _get_slice(self, rank):
        ind = []
        buffer_size = []
        mpi_coords = self.mpi_topo.cartcomm.Get_coords(rank)
        for i in range(self.dim):
            ind.append(slice(self.region[i][mpi_coords[i]], self.region[i][mpi_coords[i]+1]))
            buffer_size.append(self.region[i][mpi_coords[i]+1] - self.region[i][mpi_coords[i]])
        return ind, buffer_size

    def _set_dset(self, dset, comm,  data, id, with_id=False):
        ind, buffer_size = self._get_slice(0)
        ind = tuple(ind)
        if with_id:
            ind = ind + (id,)
        dset[ind] = data

        for i in range(1, comm.Get_size()):
            ind, buffer_size = self._get_slice(i)
            ind = tuple(ind)
            if with_id:
                ind = ind + (id,)
            rcv_buffer = np.empty(buffer_size)
            comm.Recv([rcv_buffer, mpi.DOUBLE], source=i, tag=id)
            dset[ind] = rcv_buffer
        
    def add_scalar(self, name, f, *fargs):
        if isinstance(f, np.ndarray):
            data = f
        else:
            data = f(*fargs)

        comm = self.mpi_topo.cartcomm
        if comm.Get_rank() == 0:
            dset = self.h5file.create_dataset(name, self.global_size, dtype=np.double)
            self._set_dset(dset, comm, data, 0)
            self.scalars[name] = self.h5filename + ":/" + name
        else:
            comm.Isend([np.ascontiguousarray(data), mpi.DOUBLE], dest=0, tag=0)

    def add_vector(self, name, f, *fargs):
        if isinstance(f, list):
            data = f
        else:
            data = f(*fargs)

        comm = self.mpi_topo.cartcomm
        if comm.Get_rank() == 0:
            dset = self.h5file.create_dataset(name, self.global_size + [self.dim], dtype=np.double)
            for id, d in enumerate(data):
                self._set_dset(dset, comm, d, id, with_id=True)
            self.vectors[name] = self.h5filename + ":/" + name
        else:
            for id, d in enumerate(data):
                comm.Isend([np.ascontiguousarray(d), mpi.DOUBLE], dest=0, tag=id)

    def save(self):
        comm = self.mpi_topo.cartcomm
        if comm.Get_rank() == 0:
            self.h5file.close()
            self.xdmf_file = open(self.path + '/' + self.filename + '.xdmf', "w")
            self.xdmf_file.write("""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">
 <Domain>           
            """)
            self.xdmf_file.write("""
            <Grid Name="Structured Grid" GridType="Uniform">
                <Topology TopologyType="2DRectMesh" NumberOfElements="{0}"/>
                <Geometry GeometryType="VXVY">
            """.format(' '.join(map(str, self.global_size))))
            for i in range(self.dim-1, -1, -1):
                self.xdmf_file.write("""
                <DataItem Format="HDF" Dimensions="{0}">
                    {1}:/x_{2} 
                </DataItem>
                """.format(self.global_size[i], self.filename + '.h5', i))

            self.xdmf_file.write("</Geometry>\n")

            for k, v in self.scalars.items():
                self.xdmf_file.write("""
                <Attribute Name="{0}" AttributeType="Scalar">
                <DataItem Format="HDF" Dimensions="{1}">
                {2} 
                </DataItem>
                </Attribute>
                """.format(k, ' '.join(map(str, self.global_size)), v))

            for k, v in self.vectors.items():
                self.xdmf_file.write("""
                <Attribute Name="{0}" AttributeType="Vector">
                <DataItem Format="HDF" Dimensions="{1} {2}">
                {3} 
                </DataItem>
                </Attribute>
                """.format(k, ' '.join(map(str, self.global_size)), self.dim, v))

            self.xdmf_file.write("</Grid>\n</Domain>\n</Xdmf>\n")