import numpy as np

class SOA:
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
    def __init__(self, nv, nspace, dtype=np.double):
        self.array = np.zeros(([nv] + nspace), dtype=dtype)

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, values):
        self.array[key] = values

    @property
    def nspace(self):
        return self.array.shape[1:]

    @property
    def nv(self):
        return self.array.shape[0]

    @property
    def shape(self):
        return self.array.shape

    @property
    def size(self):
        return self.array.size

    def reshape(self):
        return self.array.reshape((self.nv, np.prod(self.nspace)))

class AOS:
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
    def __init__(self, nv, nspace, dtype=np.double):
        self.array = np.zeros((nspace + [nv]), dtype=dtype)

        dim = len(nspace)
        self.swaparray = self.array
        for i in range(dim):
            self.swaparray = self.swaparray.swapaxes(i, dim)

    def __getitem__(self, key):
        return self.swaparray[key]

    def __setitem__(self, key, values):
        self.swaparray[key] = values

    @property
    def nspace(self):
        return self.array.shape[:-1]

    @property
    def nv(self):
        return self.array.shape[-1]

    @property
    def shape(self):
        return self.array.shape

    @property
    def size(self):
        return self.array.size

    def reshape(self):
        return self.array.reshape((np.prod(self.nspace), self.nv))

if __name__ == '__main__':
    # nrep = 100
    # nx, ny, nv = 1000, 1000, 9
    # f = SOA(nv, [nx, ny])
    # tt = np.arange(f.size).reshape(f.shape)
    #
    # import time
    # t = time.time()
    # for i in xrange(nrep):
    #     f[:] = tt
    # print time.time() - t
    #
    # import time
    # t = time.time()
    # for i in xrange(nrep):
    #     f[3:5, 1::2, 1:]
    # print time.time() - t
    #
    # g = AOS(nv, [nx, ny])
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
    #     g[3:5, 1::2, 1:]
    # print time.time() - t
    # #print g[3, 1, 1:]
    # #print g[1]

    g = AOS(3, [10, 10])
    g[1] = 1.
    print g[:]
    print g.array[:]
