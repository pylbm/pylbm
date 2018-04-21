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

class MPI_topology(object):
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
        # if npx, npy and npz are all set to the default value (1)
        # then Compute_dims performs the splitting of the domain
        if self.npx == self.npy == self.npz == 1:
            size = comm.Get_size()
            split = mpi.Compute_dims(size, self.dim)
        else:
            split = (self.npx, self.npy, self.npz)

        self.split = np.asarray(split[:self.dim])
        self.cartcomm = comm.Create_cart(self.split, period)
        self.region = None
        self.lx = None
        self.log = setLogger(__name__)

    def get_lx_(self, n, axes=0):
        lx = [0]
        np = self.cartcomm.Get_topo()[0][axes]
        for i in range(np):
            lx.append(lx[-1] + n//np + ((n % np) > i))
        return lx

    def get_lx(self, nx, ny=None, nz=None):
        lx = [self.get_lx_(nx, 0)]
        if ny is not None:
            lx.append(self.get_lx_(ny, 1))
        if nz is not None:
            lx.append (self.get_lx_(nz, 2))
        return lx

    def get_coords(self):
        """
        return the coords of the process in the MPI topology
        as a numpy array.
        """
        rank = self.cartcomm.Get_rank()
        return np.asarray(self.cartcomm.Get_coords(rank))

    def get_region(self, nx, ny=None, nz=None):
        lx = self.get_lx(nx, ny, nz)

        coords = self.get_coords()
        region = []
        for i in range(coords.size):
            region.append([lx[i][coords[i]], 
                           lx[i][coords[i] + 1]
                         ])
        return region

    def set_options(self):
        """
        defines command line options.
        """
        self.npx = int(options().npx)
        self.npy = int(options().npy)
        self.npz = int(options().npz)

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
