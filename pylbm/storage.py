# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Storage module
"""

import copy
import logging
from six.moves import range

import numpy as np
import sympy as sp
import mpi4py.MPI as mpi

from .generator import For
from .monitoring import monitor

log = logging.getLogger(__name__) # pylint: disable=invalid-name


class Array:
    """
    This class defines the storage of the moments and
    distribution functions in pylbm.

    It sets the storage in memory and how to access.

    Parameters
    ----------
    nv: int
        number of velocities
    gspace_size: list
        number of points in each direction including the fictitious point
    vmax: list
        the size of the fictitious points in each direction
    sorder: list
        the order of nv, nx, ny and nz
        Default is None which mean [nv, nx, ny, nz]
    mpi_topo: MpiTopology
        the mpi topology
    dtype: type
        the type of the array. Default is numpy.double
    gpu_support : bool
        true if GPU is needed

    Attributes
    ----------
    array
    nspace
    nv
    shape
    size

    """
    #pylint: disable=too-many-locals
    def __init__(self, nv, gspace_size, vmax, sorder=None,
                 mpi_topo=None, dtype=np.double, gpu_support=False):
        self.comm = mpi.COMM_WORLD
        self.sorder = sorder

        self.gspace_size = gspace_size
        self.dim = len(gspace_size)
        self.consm = {}
        self.gpu_support = gpu_support

        if mpi_topo is not None:
            self.mpi_topo = mpi_topo

            self.region = mpi_topo.get_region(*gspace_size)
            self.region_size = []
            for i in range(self.dim):
                self.region[i][0] -= vmax[i]
                self.region[i][1] += vmax[i]
                self.region_size.append(self.region[i][1] - self.region[i][0])
        else:
            self.region_size = gspace_size

        if sorder is None:
            ind = [i for i in range(self.dim + 1)]
        else:
            if len(sorder) != self.dim + 1:
                log.error("storage order must have the same length of the spatial dimension + 1.")
            ind = copy.deepcopy(sorder)

        self.index = copy.copy(ind)
        self.vmax = vmax

        tmpshape = [nv] + self.region_size
        shape = [0]*len(tmpshape)
        for i in range(self.dim + 1):
            shape[ind[i]] = int(tmpshape[i])
        self.array_cpu = np.zeros((shape), dtype=dtype)
        self.array = self.array_cpu

        if self.gpu_support:
            try:
                import pyopencl as cl
                import pyopencl.array #pylint: disable=unused-variable
                from .context import queue
            except ImportError:
                raise ImportError("Please install loo.py")
            self.array = cl.array.to_device(queue, self.array_cpu)

        self.swaparray = np.transpose(self.array_cpu, self.index)

        if mpi_topo is not None:
            self._set_subarray()

        # if self.gpu_support:
        #     self.generate()

    def __getitem__(self, key):
        if self.gpu_support:
            self.array_cpu[...] = self.array.get()
        if isinstance(key, sp.Symbol):
            return self.swaparray[self.consm[key]]
        return self.swaparray[key]

    def __setitem__(self, key, values):
        if isinstance(key, sp.Symbol):
            self.swaparray[self.consm[key]] = values
        else:
            self.swaparray[key] = values
        if self.gpu_support:
            try:
                import pyopencl as cl
                import pyopencl.array #pylint: disable=unused-variable
                from .context import queue
            except ImportError:
                raise ImportError("Please install loo.py")
            self.array = cl.array.to_device(queue, self.array_cpu)

    def _in(self, key):
        ind = []
        for vmax in self.vmax:
            ind.append(slice(vmax, -vmax))
        ind = np.asarray(ind)

        if self.gpu_support:
            self.array_cpu[...] = self.array.get()
        if isinstance(key, (sp.Symbol, sp.IndexedBase)):
            return self.swaparray[self.consm[key]][tuple(ind)]
        return self.swaparray[key][tuple(ind)]


    def set_conserved_moments(self, consm):
        """
        add conserved moments information to have a direct access.

        Parameters
        ----------
        consm : dict
            set the name and the location of the conserved moments.
            The format is

            key : the conserved moment (sympy symbol or string)

            value : list of 2 integers

                first item : the scheme number

                second item : the index of the conserved moment in this scheme

        """
        for k, v in consm.items():
            self.consm[k] = v

    @property
    def nspace(self):
        """
        the space size.
        """
        return self.swaparray.shape[1:]

    @property
    def nv(self):
        """
        the number of velocities.
        """
        return self.swaparray.shape[0]

    @property
    def shape(self):
        """
        the shape of the array that stores the data.
        """
        return self.array.shape

    @property
    def size(self):
        """
        the size of the array that stores the data.
        """
        return self.array.size

    #pylint: disable=too-many-locals
    def _set_subarray(self):
        """
        Create the neigbors and the subarrays to update interfaces
        between each processes.

        """
        nspace = list(self.nspace)
        nv = self.nv
        dim = self.dim
        vmax = self.vmax

        def swap(array_in):
            array_out = [0]*(dim+1)
            for i in range(dim+1):
                array_out[self.index[i]] = array_in[i]
            return array_out

        sizes = swap([nv] + nspace)

        rank = self.mpi_topo.cartcomm.Get_rank()
        coords = self.mpi_topo.cartcomm.Get_coords(rank)

        self.neighbors = []
        direction = []
        for i in range(dim):
            direction = copy.copy(coords)
            direction[i] -= 1
            self.neighbors.append(self.mpi_topo.cartcomm.Get_cart_rank(direction))
            direction = copy.copy(coords)
            direction[i] += 1
            self.neighbors.append(self.mpi_topo.cartcomm.Get_cart_rank(direction))

        self.send_tag = [0, 1, 2, 3, 4, 5]
        self.recv_tag = [1, 0, 3, 2, 5, 4]

        self.send_type = []
        self.recv_type = []

        for d in range(dim): #pylint: disable=invalid-name
            subsizes = [nv] + nspace
            subsizes[d+1] = vmax[d]
            subsizes = swap(subsizes)

            sstart = [0]*(dim+1)
            sstart[d+1] = vmax[d]
            sstart = swap(sstart)
            rstart = [0]*(dim+1)

            self.send_type.append(mpi.DOUBLE.Create_subarray(sizes, subsizes, sstart))
            self.recv_type.append(mpi.DOUBLE.Create_subarray(sizes, subsizes, rstart))

            log.info("[%d] send to %d with tag %d subarray:%s", rank, self.neighbors[2*d], self.send_tag[2*d], (sizes, subsizes, sstart))
            log.info("[%d] recv from %d with tag %d subarray:%s", rank, self.neighbors[2*d], self.recv_tag[2*d], (sizes, subsizes, rstart))

            sstart = [0]*(dim+1)
            sstart[d+1] = nspace[d] - 2*vmax[d]
            sstart = swap(sstart)
            rstart = [0]*(dim+1)
            rstart[d+1] = nspace[d] - vmax[d]
            rstart = swap(rstart)

            self.send_type.append(mpi.DOUBLE.Create_subarray(sizes, subsizes, sstart))
            self.recv_type.append(mpi.DOUBLE.Create_subarray(sizes, subsizes, rstart))

            log.info("[%d] send to %d with tag %d subarray:%s", rank, self.neighbors[2*d+1], self.send_tag[2*d+1], (sizes, subsizes, sstart))
            log.info("[%d] recv from %d with tag %d subarray:%s", rank, self.neighbors[2*d+1], self.recv_tag[2*d+1], (sizes, subsizes, rstart))

        for send, recv in zip(self.send_type, self.recv_type):
            send.Commit()
            recv.Commit()

    #pylint: disable=possibly-unused-variable
    @monitor
    def update(self):
        """
        update ghost points on the interface with the datas of the neighbors.
        """
        if self.gpu_support:
            # FIXME: move the generated code outside for loopy
            from .symbolic import call_genfunction

            dim = len(self.nspace)
            nx = self.nspace[0]
            if dim > 1:
                ny = self.nspace[1]
            if dim > 2:
                nz = self.nspace[2]
            nv = self.nv

            f = self.array

            args = locals()
            call_genfunction(self.generator.module.update_x, args)

            if dim > 1:
                call_genfunction(self.generator.module.update_y, args)
            if dim > 2:
                call_genfunction(self.generator.module.update_z, args)

        else:
            for d in range(self.dim): #pylint: disable=invalid-name
                req = []

                req.append(self.comm.Irecv([self.array, self.recv_type[2*d]], source=self.neighbors[2*d], tag=self.recv_tag[2*d]))
                req.append(self.comm.Irecv([self.array, self.recv_type[2*d + 1]], source=self.neighbors[2*d + 1], tag=self.recv_tag[2*d + 1]))

                req.append(self.comm.Isend([self.array, self.send_type[2*d]], dest=self.neighbors[2*d], tag=self.send_tag[2*d]))
                req.append(self.comm.Isend([self.array, self.send_type[2*d + 1]], dest=self.neighbors[2*d + 1], tag=self.send_tag[2*d + 1]))

                mpi.Request.Waitall(req)

    #pylint: disable=too-many-locals
    def generate(self, generator):
        """
        generate periodic conditions functions for loo.py backend.
        """
        self.generator = generator
        def set_order(array, remove_index=None):
            out = [-1]*len(self.sorder)
            for i, s in enumerate(self.sorder):
                out[s] = array[i]
            if remove_index:
                out.pop(self.sorder[remove_index])
            return out

        nx, ny, nz, nv = sp.symbols('nx, ny, nz, nv', integer=True)
        shape = set_order([nv, nx, ny, nz])

        i = sp.Idx('i', (0, nx))
        j = sp.Idx('j', (0, ny))
        k = sp.Idx('k', (0, nz))
        s = sp.Idx('s', (0, nv))

        fi = sp.IndexedBase('f', shape) #pylint: disable=invalid-name
        f_store = sp.Matrix([fi[set_order([s, 0, j, k])], fi[set_order([s, nx-1, j, k])]])
        f_load = sp.Matrix([fi[set_order([s, nx-2, j, k])], fi[set_order([s, 1, j, k])]])
        iloop = set_order([s, i, j, k], remove_index=1)
        generator.add_routine(('update_x', For(iloop, sp.Eq(f_store, f_load))))

        if len(self.sorder) > 2:
            f_store = sp.Matrix([fi[set_order([s, i, 0, k])], fi[set_order([s, i, ny-1, k])]])
            f_load = sp.Matrix([fi[set_order([s, i, ny-2, k])], fi[set_order([s, i, 1, k])]])
            iloop = set_order([s, i, j, k], remove_index=2)
            generator.add_routine(('update_y', For(iloop, sp.Eq(f_store, f_load))))

        if len(self.sorder) > 3:
            f_store = sp.Matrix([fi[set_order([s, i, j, 0])], fi[set_order([s, i, j, nz-1])]])
            f_load = sp.Matrix([fi[set_order([s, i, j, nz-2])], fi[set_order([s, i, j, 1])]])
            iloop = set_order([s, i, j, k], remove_index=3)
            generator.add_routine(('update_z', For(iloop, sp.Eq(f_store, f_load))))

class SOA(Array):
    """
    This class defines a structure of arrays to store the
    unknowns of the lattice Boltzmann schemes.

    Parameters
    ----------
    nv: int
        number of velocities
    gspace_size: list
        number of points in each direction including the fictitious point
    vmax: list
        the size of the fictitious points in each direction
    mpi_topo: MpiTopology
        the mpi topology
    dtype: type
        the type of the array. Default is numpy.double
    gpu_support: bool
        True if GPU is needed

    Attributes
    ----------
    array
    nspace
    nv
    shape
    size

    """
    def __init__(self, nv, gspace_size, vmax, mpi_topo, dtype=np.double, gpu_support=False):
        sorder = [i for i in range(len(gspace_size) + 1)]
        Array.__init__(self, nv, gspace_size, vmax, sorder, mpi_topo, dtype, gpu_support=gpu_support)

    def reshape(self):
        """
        reshape.
        """
        return self.array

class AOS(Array):
    """
    This class defines an array of structures to store the
    unknowns of the lattice Boltzmann schemes.

    Parameters
    ----------
    nv: int
        number of velocities
    gspace_size: list
        number of points in each direction including the fictitious point
    vmax: list
        the size of the fictitious points in each direction
    mpi_topo: MpiTopology
        the mpi topology
    dtype: type
        the type of the array. Default is numpy.double
    gpu_support: bool
        True if GPU is needed

    Attributes
    ----------
    array
    nspace
    nv
    shape
    size

    """
    def __init__(self, nv, gspace_size, vmax, mpi_topo, dtype=np.double, gpu_support=False):
        sorder = [len(gspace_size)] + [i for i in range(len(gspace_size))]
        Array.__init__(self, nv, gspace_size, vmax, sorder, mpi_topo, dtype, gpu_support=gpu_support)

    def reshape(self):
        """
        reshape
        """
        return self.array.reshape((np.prod(self.nspace), self.nv))
