import numpy as np
import copy
import mpi4py.MPI as mpi

from .logs import setLogger

class Array:
    """
    This class defines the storage of the moments and
    distribution functions in pyLBM.

    It sets the storage in memory and how to access.
    """
    def __init__(self, nv, nspace, vmax, inv=None, inspace=None, dtype=np.double, cartcomm = None):
        self.log = setLogger(__name__)
        self.comm = mpi.COMM_WORLD
        self.cartcomm = cartcomm

        if inv is None:
            ind = [0]
        else:
            ind = [inv]

        if inspace is None:
            ind += [i + 1 for i in range(len(nspace))]
        else:
            ind += inspace

        self.index = copy.copy(ind)
        self.vmax = vmax

        tmpshape = [nv] + nspace
        shape = [0]*len(tmpshape)
        for i in range(len(shape)):
            shape[ind[i]] = tmpshape[i]
        self.array = np.zeros((shape), dtype=dtype)

        dim = len(nspace)
        self.swaparray = self.array
        for i in range(dim + 1):
            if i != ind[i]:
                j = ind.index(i)
                self.swaparray = self.swaparray.swapaxes(i, ind[i])
                ind[j] = ind[i]
                ind[i] = i

        self._set_subarray()

    def __getitem__(self, key):
        return self.swaparray[key]

    def __setitem__(self, key, values):
        self.swaparray[key] = values

    @property
    def nspace(self):
        return self.swaparray.shape[1:]

    @property
    def nv(self):
        return self.swaparray.shape[0]

    @property
    def shape(self):
        return self.array.shape

    @property
    def size(self):
        return self.array.size

    def _set_subarray(self):
        """
        Create the neigbors and the subarrays to update interfaces
        between each processes.

        Parameters
        ----------

        vmax : list
          the maximal velocity in norm for each spatial direction.

        """

        rank = self.cartcomm.Get_rank()
        coords = self.cartcomm.Get_coords(rank)

        nspace = self.nspace
        nv = self.nv
        dim = len(self.nspace)
        vmax = self.vmax

        # set nloc without the ghost points
        nloc = [i - 2*v for i, v in zip(nspace, vmax)]
        nswap = (nv,) + nspace
        n = [0]*(dim+1)
        for i in range(dim+1):
            n[self.index[i]] = nswap[i]
        # set the size and the start indices
        # for the send and receive messages
        start_send = []
        start_recv = []
        msize = []
        stag, rtag = get_tags(dim)
        for i in xrange(dim):
            start_send.append([vmax[i], vmax[i], nspace[i]-2*vmax[i]])
            start_recv.append([0, vmax[i], nspace[i]-vmax[i]])
            msize.append([vmax[i], nloc[i], vmax[i]])
        start_send = np.asarray(start_send)
        start_recv = np.asarray(start_recv)
        msize = np.asarray(msize)

        # set the neighbors of the domain and their subarrays
        # for the send and receive messages
        self.neighbors = []
        self.sendType = []
        self.sendTag = []
        self.recvType = []
        self.recvTag = []
        directions = get_directions(dim)
        rows = np.arange(dim)
        for d in directions:
            if not np.all(d == 0):
                try:
                    neighbor = self.cartcomm.Get_cart_rank(coords + d)
                    self.neighbors.append(neighbor)

                    msswap = [nv] + list(msize[rows, d+1])
                    ssswap = [0] + list(start_send[rows, d+1])
                    srswap = [0] + list(start_recv[rows, d+1])

                    ms = [0]*(dim+1)
                    ss = [0]*(dim+1)
                    sr = [0]*(dim+1)
                    for i in range(dim+1):
                        ms[self.index[i]] = msswap[i]
                        ss[self.index[i]] = ssswap[i]
                        sr[self.index[i]] = srswap[i]

                    self.sendType.append(mpi.DOUBLE.Create_subarray(n, ms, ss))
                    self.recvType.append(mpi.DOUBLE.Create_subarray(n, ms, sr))
                    self.sendTag.append(stag[tuple(d+1)])
                    self.recvTag.append(rtag[tuple(d+1)])
                    self.log.info("[{0}] send to {1} with tag {2} subarray:{3}".format(rank, neighbor, self.sendTag[-1], (n, ms, ss)))
                    self.log.info("[{0}] recv from {1} with tag {2} subarray:{3}".format(rank, neighbor, self.recvTag[-1], (n, ms, sr)))
                except mpi.Exception:
                    pass

        for s, r in zip(self.sendType, self.recvType):
            s.Commit()
            r.Commit()

    def update(self):
        """
        update ghost points on the interface with the datas of the neighbors.
        """
        req = []
        for i in xrange(len(self.recvType)):
            req.append(self.comm.Irecv([self.array, self.recvType[i]], source = self.neighbors[i], tag=self.recvTag[i]))

        for i in xrange(len(self.sendType)):
            req.append(self.comm.Isend([self.array, self.sendType[i]], dest = self.neighbors[i], tag=self.sendTag[i]))

        mpi.Request.Waitall(req)

class SOA(Array):
    """
    This class defines a structure of arrays to store the
    unknowns in the lattice Boltzmann schemes.

    Parameters
    ----------

    nv: int
        number of velocities
    nspace: list of int
        number of points in each direction
    dtype: type
        the type of the array. Default is numpy.double

    Attributes
    ----------

    array: NumPy array
        the array that stores the solution
    nspace: list of int
        the space dimension
    nv: int
        the number of velocities
    shape:
        the shape of array
    size:
        the total size of the array

    """
    def __init__(self, nv, nspace, vmax, dtype=np.double, cartcomm=None):
        Array.__init__(self, nv, nspace, vmax, dtype=dtype, cartcomm=cartcomm)

    def reshape(self):
        return self.array

class AOS(Array):
    """
    This class defines an array of structures to store the
    unknowns in the lattice Boltzmann schemes.

    Parameters
    ----------

    nv: int
        number of velocities
    nspace: list of int
        number of points in each direction
    dtype: type
        the type of the array. Default is numpy.double

    Attributes
    ----------

    array: NumPy array
        the array that stores the solution
    nspace: list of int
        the space dimension
    nv: int
        the number of velocities
    shape:
        the shape of array
    size:
        the total size of the array

    """
    def __init__(self, nv, nspace, vmax, dtype=np.double, cartcomm=None):
        ls = len(nspace)
        Array.__init__(self, nv, nspace, vmax, inv=ls,
                       inspace=[i for i in range(ls)],
                       dtype=dtype, cartcomm=cartcomm)

    def reshape(self):
        return self.array.reshape((np.prod(self.nspace), self.nv))


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
    f = SOA(nv, [nx, ny])
    tt = np.arange(f.size).reshape(f.shape)

    a1 = Array(nv, [nx, ny])
    a1[1:3, :2, 1:] = 1
    print a1.shape
    #print a1.array
    print a1.swaparray.shape
    print a1.swaparray

    a2 = Array(nv, [nx, ny], inv=1, inspace=[2, 0])
    #a2[1:3, :2, 1:] = 1
    print a2.shape
    print a2.swaparray.shape

    #print a2.array
    print a2.swaparray.flags
    print a2.array.flags
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
