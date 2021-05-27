# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
HDF5 module
"""
import os
import logging
from six.moves import range
import numpy as np
import h5py
import mpi4py.MPI as mpi

log = logging.getLogger(__name__) #pylint: disable=invalid-name

class H5File:
    """
    class to manage hfd5 and xdmf file.
    """
    def __init__(self, mpi_topo, filename, path='', timestep=None, init_xdmf=False):
        if timestep is not None:
            prefix = '_{}'.format(timestep)
        else:
            prefix = ''
        self.path = path
        name, ext = os.path.splitext(filename)
        self.filename = name + prefix
        self.h5filename = name + prefix + '.h5'

        self.origin = None
        self.dx = None
        self.dim = None
        self.n = None
        self.region = None
        self.global_size = None
        self.xdmf_file = None

        if mpi.COMM_WORLD.Get_rank() == 0:
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

    def set_grid(self, x, y=None, z=None):
        """
        create the hdf5 coordinate.

        Parameters
        ----------

        x : ndarray
            x-coordinate

        y : ndarray
            y-coordinate
            default is None

        z : ndarray
            z-coordinate
            default is None

        """
        if not self._init_grid:
            log.warning("h5 grid redefined.")

        self.origin = [x[0]]
        self.dx = [x[1] - x[0]]
        coords = [x]
        if y is not None:
            self.origin.append(y[0])
            self.dx.append(y[1] - y[0])
            coords.append(y)
        if z is not None:
            self.origin.append(z[0])
            self.dx.append(z[1] - z[0])
            coords.append(z)

        self.dim = len(self.origin)
        self.n = [0]*self.dim
        self.region = []
        self.global_size = []

        # get the region own by each processes
        # the global size
        # and the global coords
        for i in range(self.dim):
            sub = [False]*self.dim
            sub[i] = True
            comm = self.mpi_topo.cartcomm.Sub(sub)
            self.n[i] = comm.allreduce(coords[i].size, op=mpi.SUM)
            self.region.append(comm.gather(coords[i].size, root=0))
            self.global_size.append(np.sum(np.asarray(self.region[i])))
            coords[i] = np.asarray(comm.gather(coords[i], root=0))
            comm.Free()

        if mpi.COMM_WORLD.Get_rank() == 0:
            for i in range(len(self.region)):
                self.region[i].insert(0, 0)
                for j in range(1, len(self.region[i])):
                    self.region[i][j] += self.region[i][j-1]
            #self.region = np.asarray(self.region, dtype=np.int)
            #self.region = np.concatenate((np.zeros((self.dim, 1), dtype=np.int), np.cumsum(self.region, axis=1)), axis=1)
            #print(self.region)
            for i in range(self.dim):
                dset = self.h5file.create_dataset("x_{}".format(i), [self.global_size[i]], dtype=np.double)
                dset[:] = np.concatenate(coords[i])

    def _get_slice(self, rank):
        """
        get the part own by the processor rank and
        the size of the data in each direction
        """
        ind = []
        buffer_size = []
        mpi_coords = self.mpi_topo.cartcomm.Get_coords(rank)
        # print('coords', rank, mpi_coords)
        for i in range(self.dim):
            ind.append(slice(self.region[i][mpi_coords[i]], self.region[i][mpi_coords[i]+1]))
            buffer_size.append(self.region[i][mpi_coords[i]+1] - self.region[i][mpi_coords[i]])
        return ind[::-1], buffer_size[::-1]

    def _set_dset(self, dset, comm, data, index=0, with_index=False):
        """
        Merge data from multiple sub domains into a dataset.

        Parameters
        ----------

        dset : dataset
            hdf5 dataset where to store the data

        comm : comm
            mpi communicator

        data : array
            data on the sub-domain

        index : int
            use to store an index of a vector

        with_index : bool
            if we store a vector component

        """
        ind, buffer_size = self._get_slice(0)
        ind = tuple(ind)
        if with_index:
            ind = ind + (index,)
        dset[ind] = data.T

        for i in range(1, comm.Get_size()):
            ind, buffer_size = self._get_slice(i)
            #print(i, ind, buffer_size)
            ind = tuple(ind)
            if with_index:
                ind = ind + (index,)
            rcv_buffer = np.empty(buffer_size)
            comm.Recv([rcv_buffer, mpi.DOUBLE], source=i, tag=index)
            dset[ind] = rcv_buffer

    def add_scalar(self, name, f, *fargs):
        """
        store a scalar field.

        Parameters
        ----------

        name: string
            the name of the dataset entry

        f : array or function
            data to store

        fargs: tuple
            arguments of f if f is a function

        """
        if isinstance(f, np.ndarray):
            data = f
        else:
            data = f(*fargs)

        comm = self.mpi_topo.cartcomm
        if comm.Get_rank() == 0:
            dset = self.h5file.create_dataset(name, self.global_size[::-1], dtype=np.double)
            self._set_dset(dset, comm, data)
            self.scalars[name] = self.h5filename + ":/" + name
        else:
            comm.Send([np.ascontiguousarray(data.T, dtype=np.double), mpi.DOUBLE], dest=0, tag=0)

    def add_vector(self, name, f, *fargs):
        """
        store a vector field.

        Parameters
        ----------

        name: string
            the name of the dataset entry

        f : array or function
            data to store

        fargs: tuple
            arguments of f if f is a function

        """
        if isinstance(f, list):
            datas = f
        else:
            datas = f(*fargs)

        comm = self.mpi_topo.cartcomm
        if comm.Get_rank() == 0:
            dset = self.h5file.create_dataset(name, self.global_size[::-1] + [3], dtype=np.double)
            for i, data in enumerate(datas):
                self._set_dset(dset, comm, data, i, with_index=True)
            self.vectors[name] = self.h5filename + ":/" + name
        else:
            for i, data in enumerate(datas):
                comm.Send([np.ascontiguousarray(data.T), mpi.DOUBLE], dest=0, tag=i)

    def save(self):
        """
        save the hdf5 and the xdmf files.
        """
        comm = self.mpi_topo.cartcomm
        if comm.Get_rank() == 0:
            self.h5file.close()
            self.xdmf_file = open(self.path + '/' + self.filename + '.xdmf', "w")
            self.xdmf_file.write("""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf>
 <Domain>           
            """)
            if self.dim == 2:
                self.xdmf_file.write("""
                <Grid Name="Structured Grid" GridType="Uniform">
                    <Topology TopologyType="2DRectMesh" NumberOfElements="{0}"/>
                    <Geometry GeometryType="VXVY">
                """.format(' '.join(map(str, self.global_size))))
            else:
                self.xdmf_file.write("""
                <Grid Name="Structured Grid" GridType="Uniform">
                    <Topology TopologyType="3DRectMesh" NumberOfElements="{0}"/>
                    <Geometry GeometryType="VXVYVZ">
                """.format(' '.join(map(str, self.global_size))))
            for i in range(self.dim):
                self.xdmf_file.write("""
                <DataItem Format="HDF" Dimensions="{0}">
                    {1}:/x_{2} 
                </DataItem>
                """.format(self.global_size[i], self.filename + '.h5', i))

            self.xdmf_file.write("</Geometry>\n")

            for k, v in self.scalars.items():
                self.xdmf_file.write("""
                <Attribute Name="{0}" AttributeType="Scalar" Center="Node">
                <DataItem Format="HDF" Dimensions="{1}">
                {2} 
                </DataItem>
                </Attribute>
                """.format(k, ' '.join(map(str, self.global_size[::-1])), v))

            for k, v in self.vectors.items():
                self.xdmf_file.write("""
                <Attribute Name="{0}" AttributeType="Vector" Center="Node">
                <DataItem Format="HDF" Dimensions="{1} {2}">
                {3} 
                </DataItem>
                </Attribute>
                """.format(k, ' '.join(map(str, self.global_size[::-1])), self.dim, v))

            self.xdmf_file.write("</Grid>\n</Domain>\n</Xdmf>\n")
            self.xdmf_file.close()
