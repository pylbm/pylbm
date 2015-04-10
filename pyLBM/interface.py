# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np
import mpi4py.MPI as mpi
from argparse import ArgumentParser

from .options import options
from .logs import setLogger

class Interface:
    """
    Interface construction using a MPI topology.

    Parameters
    ----------

    dim : int
      number of spatial dimensions (1, 2, or 3)
    period : list
      boolean list that specifies if a direction is periodic or not.
      Its size is dim.

    Attributes
    ----------

    dim : int
      number of spatial dimensions (1, 2, or 3)
    comm : MPI communicator
      the communicator of the topology
    split : tuple
      number of processes in each direction
    neighbors : list
      list of the neighbors where we have to send and to receive messages
    sendType : list
      list of subarrays that defines the part of data to be send
    sendTag : list
      list of tags for the send messages
    recvType : list
      list of subarrays that defines the part of data to update during a receive message
    recvTag : list
      list of tags for the receive messages

    Methods
    -------

    set_options :
      defines command line options.
    get_coords :
      return the coords of the process in the MPI topology.
    set_subarray :
      create subarray for the send and receive message
    update :
      update a numpy array according to the subarrays and the topology.

    """
    def __init__(self, dim, period, comm=mpi.COMM_WORLD):
        self.dim = dim
        self.set_options()

        self.comm = comm
        # if npx, npy and npz are all the default value (1)
        # then Compute_dims performs the splitting of the domain
        if self.npx == self.npy == self.npz == 1:
            size = comm.Get_size()
            split = mpi.Compute_dims(size, self.dim)
        else:
            split = (self.npx, self.npy, self.npz)

        self.split = np.asarray(split[:self.dim])
        self.cartcomm = comm.Create_cart(self.split, period)

        self.log = setLogger(__name__)

    def get_coords(self):
        """
        return the coords of the process in the MPI topology
        as a numpy array.
        """
        rank = self.cartcomm.Get_rank()
        return np.asarray(self.cartcomm.Get_coords(rank))

    def set_options(self):
        """
        defines command line options.
        """
        self.npx = int(options().npx)
        self.npy = int(options().npy)
        self.npz = int(options().npz)

    def set_subarray(self, n, vmax, nv_on_beg=False):
        """
        Create the neigbors and the subarrays to update interfaces
        between each processes.

        Parameters
        ----------

        n : list
          shape of the moment and distribution arrays
        vmax : list
          the maximal velocity in norm for each spatial direction.
        nv_on_beg : boolean
          True if the LBM velocities are set on the beginning of the moment
          and distribution arrays and False otherwise.

        """

        rank = self.cartcomm.Get_rank()
        coords = self.cartcomm.Get_coords(rank)

        # set nloc without the ghost points
        if nv_on_beg:
            nloc = [i - 2*v for i, v in zip(n[1:], vmax)]
            nv = n[0]
            nn = n[1:]
        else:
            nloc = [i - 2*v for i, v in zip(n[:-1], vmax)]
            nv = n[-1]
            nn = n[:-1]

        # set the size and the start indices
        # for the send and receive messages
        start_send = []
        start_recv = []
        msize = []
        stag, rtag = get_tags(self.dim)
        for i in xrange(self.dim):
            start_send.append([vmax[i], vmax[i], nn[i]-2*vmax[i]])
            start_recv.append([0, vmax[i], nn[i]-vmax[i]])
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
        directions = get_directions(self.dim)
        rows = np.arange(self.dim)
        for d in directions:
            if not np.all(d == 0):
                try:
                    neighbor = self.cartcomm.Get_cart_rank(coords + d)
                    self.neighbors.append(neighbor)

                    if nv_on_beg:
                        ms = [nv] + list(msize[rows, d[::-1]+1])
                        ss = [0] + list(start_send[rows, d[::-1]+1])
                        sr = [0] + list(start_recv[rows, d[::-1]+1])
                    else:
                        ms = list(msize[rows, d[::-1]+1]) + [nv]
                        ss = list(start_send[rows, d[::-1]+1]) + [0]
                        sr = list(start_recv[rows, d[::-1]+1]) + [0]

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

    def update(self, f):
        """
        update ghost points on the interface with the datas of the neighbors.
        """
        req = []

        for i in xrange(len(self.recvType)):
            req.append(self.comm.Irecv([f, self.recvType[i]], source = self.neighbors[i], tag=self.recvTag[i]))

        for i in xrange(len(self.sendType)):
            req.append(self.comm.Isend([f, self.sendType[i]], dest = self.neighbors[i], tag=self.sendTag[i]))

        mpi.Request.Waitall(req)

    def get_full(self, f, domain):
        globalsizes = domain.Ng
        subsizes = np.array(globalsizes/self.split, dtype='i')
        starts = np.zeros(self.dim + 1)

        rank = self.cartcomm.Get_rank()
        size = self.cartcomm.Get_size()
        coords = self.cartcomm.Get_coords(rank)

        for i in xrange(self.dim):
            if coords[i] + 1 == self.split[i]:
                subsizes[i] += globalsizes[i]%self.split[i]

        globalsizes = globalsizes[::-1]
        subsizes = np.ascontiguousarray(subsizes[::-1])

        Subsizes = np.empty(self.dim, dtype='i')

        self.comm.Allreduce([subsizes, mpi.INT], [Subsizes, mpi.INT], mpi.MAX)

        ns = f.shape[-1]
        newtype = None
        if rank == 0:
            subarray = mpi.DOUBLE.Create_subarray(globalsizes + [ns], list(Subsizes) + [ns], starts)
            newtype = subarray.Create_resized(0, 8)
            newtype.Commit()

        globalArray = None

        counts = 0
        displs = 0
        if rank == 0:
            counts = np.array([1]*size)
            disp = 0
            displs = []
            for i in xrange(self.split[1]):
                for j in xrange(self.split[0]):
                    displs.append(disp)
                    disp += subsizes[1]*ns
                disp += ((subsizes[0]-1)*globalsizes[1]+1)*ns
            globalArray = np.zeros(globalsizes + [ns])


        copyArray = np.empty(list(Subsizes) + [ns])
        copyArray[:subsizes[0], :subsizes[1], :] = f[1:-1, 1:-1, :]
        self.comm.Gatherv([copyArray, mpi.DOUBLE], [globalArray, (counts, displs), newtype], 0)

        return globalArray

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
