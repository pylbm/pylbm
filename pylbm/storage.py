from __future__ import print_function
from six.moves import range
import numpy as np
import sympy as sp
import copy
import mpi4py.MPI as mpi

from .logs import setLogger
from .generator import generator

class Array(object):
    """
    This class defines the storage of the moments and
    distribution functions in pylbm.

    It sets the storage in memory and how to access.

    Parameters
    ----------
    nv: int
        number of velocities
    gspace_size: list of int
        number of points in each direction including the fictitious point
    vmax: list of int
        the size of the fictitious points in each direction
    sorder: list of int
        the order of nv, nx, ny and nz
        Default is None which mean [nv, nx, ny, nz]
    mpi_topo:
        the mpi topology
    dtype: type
        the type of the array. Default is numpy.double

    Attributes
    ----------
    array
    nspace
    nv
    shape
    size

    """
    def __init__(self, nv, gspace_size, vmax, sorder=None, mpi_topo=None, dtype=np.double, gpu_support=False):
        self.log = setLogger(__name__)
        self.comm = mpi.COMM_WORLD
        self.sorder = sorder

        self.gspace_size = gspace_size
        self.dim = len(gspace_size)

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
                self.log.error("storage order must have the same length of the spatial dimension + 1.")
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
            import pyopencl as cl
            import pyopencl.array
            from .context import queue

            self.array = cl.array.to_device(queue, self.array_cpu)

        self.swaparray = np.transpose(self.array_cpu, self.index)

        if mpi_topo is not None:
            self._set_subarray()

        if self.gpu_support:
            self.generate()

    def __getitem__(self, key):
        if self.gpu_support:
            self.array_cpu[...] = self.array.get()
        if isinstance(key, sp.Symbol):
            return self.swaparray[self.consm[key]]
        return self.swaparray[key]

    def __setitem__(self, key, values):
        if isinstance(key , sp.Symbol):
            self.swaparray[self.consm[key]] = values
        else:
            self.swaparray[key] = values
        if self.gpu_support:
            import pyopencl as cl
            import pyopencl.array
            from .context import queue
            
            self.array = cl.array.to_device(queue, self.array_cpu)            

    def _in(self, key):
        ind = []
        for vmax in self.vmax:
            ind.append(slice(vmax, -vmax))
        ind = np.asarray(ind)

        if self.gpu_support:
            self.array_cpu[...] = self.array.get()
        if isinstance(key, sp.Symbol):
            return self.swaparray[self.consm[key]][tuple(ind)]
        return self.swaparray[key][tuple(ind)]


    def set_conserved_moments(self, consm, nv_ptr):
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

        nv_ptr : list of int
            store the location of the schemes

        """
        self.consm = {}
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
            direction = copy.copy(coords); direction[i] -= 1
            self.neighbors.append(self.mpi_topo.cartcomm.Get_cart_rank(direction))
            direction = copy.copy(coords); direction[i] += 1
            self.neighbors.append(self.mpi_topo.cartcomm.Get_cart_rank(direction))

        self.sendTag = [0, 1, 2, 3, 4, 5]
        self.recvTag = [1, 0, 3, 2, 5, 4]

        self.sendType = []
        self.recvType = []

        for d in range(dim):
            subsizes = [nv] + nspace; subsizes[d+1] = vmax[d]
            subsizes = swap(subsizes)
        
            sstart = [0]*(dim+1); sstart[d+1] = vmax[d]
            sstart = swap(sstart)
            rstart = [0]*(dim+1)

            self.sendType.append(mpi.DOUBLE.Create_subarray(sizes, subsizes, sstart))
            self.recvType.append(mpi.DOUBLE.Create_subarray(sizes, subsizes, rstart))

            self.log.info("[{0}] send to {1} with tag {2} subarray:{3}".format(rank, self.neighbors[2*d], self.sendTag[2*d], (sizes, subsizes, sstart)))
            self.log.info("[{0}] recv from {1} with tag {2} subarray:{3}".format(rank, self.neighbors[2*d], self.recvTag[2*d], (sizes, subsizes, rstart)))

            sstart = [0]*(dim+1); sstart[d+1] = nspace[d] - 2*vmax[d]
            sstart = swap(sstart)
            rstart = [0]*(dim+1); rstart[d+1] = nspace[d] - vmax[d]
            rstart = swap(rstart)

            self.sendType.append(mpi.DOUBLE.Create_subarray(sizes, subsizes, sstart))
            self.recvType.append(mpi.DOUBLE.Create_subarray(sizes, subsizes, rstart))

            self.log.info("[{0}] send to {1} with tag {2} subarray:{3}".format(rank, self.neighbors[2*d+1], self.sendTag[2*d+1], (sizes, subsizes, sstart)))
            self.log.info("[{0}] recv from {1} with tag {2} subarray:{3}".format(rank, self.neighbors[2*d+1], self.recvTag[2*d+1], (sizes, subsizes, rstart)))

        for s, r in zip(self.sendType, self.recvType):
            s.Commit()
            r.Commit()

    def update(self):
        """
        update ghost points on the interface with the datas of the neighbors.
        """
        if self.gpu_support:
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
            call_genfunction(generator.module.update_x, args)

            if dim > 1:
                call_genfunction(generator.module.update_y, args)
            if dim > 2:
                call_genfunction(generator.module.update_z, args)

        else:
            for d in range(self.dim):
                req = []

                req.append(self.comm.Irecv([self.array, self.recvType[2*d]], source = self.neighbors[2*d], tag=self.recvTag[2*d]))
                req.append(self.comm.Irecv([self.array, self.recvType[2*d + 1]], source = self.neighbors[2*d + 1], tag=self.recvTag[2*d + 1]))

                req.append(self.comm.Isend([self.array, self.sendType[2*d]], dest = self.neighbors[2*d], tag=self.sendTag[2*d]))
                req.append(self.comm.Isend([self.array, self.sendType[2*d + 1]], dest = self.neighbors[2*d + 1], tag=self.sendTag[2*d + 1]))

                mpi.Request.Waitall(req)

    def generate(self):
        """
        generate periodic conditions functions for loo.py backend.
        """
        import sympy as sp
        from .generator import generator, For, If

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

        fi = sp.IndexedBase('f', shape)
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
    gspace_size: list of int
        number of points in each direction including the fictitious point
    vmax: list of int
        the size of the fictitious points in each direction
    mpi_topo:
        the mpi topology
    dtype: type
        the type of the array. Default is numpy.double

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
    gspace_size: list of int
        number of points in each direction including the fictitious point
    vmax: list of int
        the size of the fictitious points in each direction
    mpi_topo:
        the mpi topology
    dtype: type
        the type of the array. Default is numpy.double

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

class Array_in(Array):
    def __init__(self, array):
        self.log = setLogger(__name__)
        self.vmax = array.vmax
        self.consm = array.consm
        self.gspace_size = array.gspace_size
        self.mpi_topo = array.mpi_topo
        self.gpu_support = array.gpu_support
        self.sorder = array.sorder

        self.ind = [slice(None)]
        for vmax in array.vmax:
            self.ind.append(slice(vmax, -vmax))
        self.ind = np.asarray(self.ind)
        self.swaparray = array.swaparray[tuple(self.ind)]
        self.array_cpu = array.array_cpu[tuple(self.ind[array.sorder])]
        self.array = array.array
        self.region = self.mpi_topo.get_region(*self.gspace_size)

    def get_global_array(self):
        lx = self.mpi_topo.get_lx(*self.gspace_size)
        region = self.mpi_topo.get_region(*self.gspace_size)

        #if self.mpi_topo.cartcomm.Get_rank() == 0:
        n = [l[-1] for l in lx]
        garray = np.empty([self.nv] + n)
        
        
        nglobal = np.asarray([self.nv] + self.gspace_size, dtype=np.int)
        nlocal = self.swaparray.shape
        #start = [0] + [r[0] for r in region]
        start = [0]*3

        print('global', nglobal, 'local', nlocal, 'start', start)
        newtype = mpi.DOUBLE.Create_subarray(nglobal, nlocal, start)
        newtype.Commit()
        recvbuf = [garray, newtype]

        self.mpi_topo.cartcomm.Gather([np.ascontiguousarray(self.swaparray), mpi.DOUBLE], recvbuf, root=0)
        return garray


def get_directions(dim):
    """
    Return an array with all the directions around.

    Parameters
    ----------

    dim : int
      number of spatial dimensions (1, 2, or 3)

    Examples
    --------

    >>> get_directions(1)
    array([[-1],
       [ 0],
       [ 1]])
    >>> get_directions(2)
    array([[-1, -1],
       [-1,  0],
       [-1,  1],
       [ 0, -1],
       [ 0,  0],
       [ 0,  1],
       [ 1, -1],
       [ 1,  0],
       [ 1,  1]], dtype=int32)

    """
    a = np.array([-1, 0, 1])

    if dim == 1:
        directions = a[:, np.newaxis]
    elif dim == 2:
        a = a[np.newaxis, :]

        directions = np.empty((9, 2), dtype=np.int32)
        directions[:, 0] = np.repeat(a, 3, axis=1).flatten()
        directions[:, 1] = np.repeat(a, 3, axis=0).flatten()
    elif dim == 3:
        a = a[np.newaxis, :]

        directions = np.empty((27, 3), dtype=np.int32)
        directions[:, 0] = np.repeat(a, 9, axis=1).flatten()
        directions[:, 1] = np.repeat(np.repeat(a, 3, axis=0), 3).flatten()
        directions[:, 2] = np.repeat(a, 9, axis=0).flatten()

    return directions

def get_tags(dim):
    tag = np.arange((3)**dim).reshape((3,)*dim)
    if dim == 1:
        return tag, tag[::-1]
    if dim == 2:
        return tag, tag[::-1, ::-1]
    if dim == 3:
        return tag, tag[::-1, ::-1, ::-1]



if __name__ == '__main__':
    nrep = 100
    nx, ny, nz, nv = 2, 3, 5, 4
    f = SOA(nv, [nx, ny], [1, 1])
    tt = np.arange(f.size).reshape(f.shape)

    a1 = Array(nv, [nx, ny], [1, 1])
    a1[1:3, :2, 1:] = 1
    print(a1.shape)
    #print a1.array
    print(a1.swaparray.shape)
    print(a1.swaparray)

    a2 = Array(nv, [nx, ny], inv=1, inspace=[2, 0])
    #a2[1:3, :2, 1:] = 1
    print(a2.shape)
    print(a2.swaparray.shape)

    #print a2.array
    print(a2.swaparray.flags)
    print(a2.array.flags)
    #print a2.swaparray
    # import time
    # t = time.time()
    # for i in xrange(nrep):
    #     f[:] = tt
    # print time.time() - t
    #
    # import time
    # t = time.time()
    # for i in xrange(nrep):
    #     f[3]
    # print time.time() - t
    #
    g = AOS(nv, [nx, ny])
    #
    # import time
    # t = time.time()
    # for i in xrange(nrep):
    #     g[:] = tt
    # print time.time() - t
    #
    # import time
    # t = time.time()
    # for i in xrange(nrep):
    #     g[3]
    # print time.time() - t
    # #print g[3, 1, 1:]
    # #print g[1]
    #
    # # g = AOS(3, [10, 10])
    # # g[1] = 1.
    # # print g[:]
    # # print g.array[:]
