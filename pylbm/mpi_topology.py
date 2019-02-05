# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Module which implements a Cartesian MPI topology
"""

import numpy as np
import mpi4py.MPI as mpi

from .options import options

class MpiTopology:
    """
    Interface construction using a MPI topology.

    Parameters
    ----------

    dim : int
      number of spatial dimensions (1, 2, or 3)
    comm : comm
      the default MPI communicator
    period : list
      boolean list that specifies if a direction is periodic or not.
      Its size is dim.

    Attributes
    ----------

    dim : int
      number of spatial dimensions (1, 2, or 3)
    comm : comm
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

    def get_region_indices_(self, n, axis=0):
        """
        1D region indices owned by each sub domain.

        Parameters
        ----------

        n : int
            number of total discrete points for a given axis
        axis : int
            axis used in the MPI topology

        Returns
        -------

        list
            list of regions owned by each processes for a given axis

        """
        region_indices = [0]
        nproc = self.cartcomm.Get_topo()[0][axis]
        for i in range(nproc):
            region_indices.append(region_indices[-1] + n//nproc + ((n % nproc) > i))
        return region_indices

    def get_region_indices(self, nx, ny=None, nz=None):
        """
        Region indices owned by each sub domain.

        Parameters
        ----------

        nx : int
            number of total discrete points in x direction
        ny : int
            number of total discrete points in y direction
            default is None
        nz : int
            number of total discrete points in z direction
            default is None

        Returns
        -------

        list
            list of regions owned by each processes

        """
        region_indices = [self.get_region_indices_(nx, 0)]
        if ny is not None:
            region_indices.append(self.get_region_indices_(ny, 1))
        if nz is not None:
            region_indices.append(self.get_region_indices_(nz, 2))
        return region_indices

    def get_coords(self):
        """
        return the coords of the process in the MPI topology
        as a numpy array.
        """
        rank = self.cartcomm.Get_rank()
        return np.asarray(self.cartcomm.Get_coords(rank))

    def get_region(self, nx, ny=None, nz=None):
        """
        Region indices owned by the sub domain.

        Parameters
        ----------

        nx : int
            number of total discrete points in x direction
        ny : int
            number of total discrete points in y direction
            default is None
        nz : int
            number of total discrete points in z direction
            default is None

        Returns
        -------

        list
            region owned by the process

        """

        region_indices = self.get_region_indices(nx, ny, nz)

        coords = self.get_coords()
        region = []
        for i in range(coords.size):
            region.append([region_indices[i][coords[i]],
                           region_indices[i][coords[i] + 1]
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

    Returns
    -------

    ndarray
        all the possible directions with a stencil of 1

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
    common_direction = np.array([-1, 0, 1])

    if dim == 1:
        directions = common_direction[:, np.newaxis]
    elif dim == 2:
        common_direction = common_direction[np.newaxis, :]

        directions = np.empty((9, 2), dtype=np.int32)
        directions[:, 0] = np.repeat(common_direction, 3, axis=1).flatten()
        directions[:, 1] = np.repeat(common_direction, 3, axis=0).flatten()
    elif dim == 3:
        common_direction = common_direction[np.newaxis, :]

        directions = np.empty((27, 3), dtype=np.int32)
        directions[:, 0] = np.repeat(common_direction, 9, axis=1).flatten()
        directions[:, 1] = np.repeat(np.repeat(common_direction, 3, axis=0), 3).flatten()
        directions[:, 2] = np.repeat(common_direction, 9, axis=0).flatten()

    return directions
