import numpy as np
import mpi4py.MPI as mpi
from argparse import ArgumentParser

class Interface:
    def __init__(self, dim, period):
        self.dim = dim
        self.set_options()

        comm = mpi.COMM_WORLD
        if self.npx == self.npy == self.npz == 1:
            size = comm.Get_size()
            split = mpi.Compute_dims(size, self.dim)
        else:
            split = (self.npx, self.npy, self.npz)
        self.split = split[:self.dim]
        self.comm = comm.Create_cart(self.split, period)

    def get_coords(self):
        rank = self.comm.Get_rank()
        return np.asarray(self.comm.Get_coords(rank))

    def set_options(self):
        parser = ArgumentParser()
        parser.add_argument("-npx", dest="npx", default=1,
                             help="Set the number of processes in x direction")
        parser.add_argument("-npy", dest="npy", default=1,
                             help="Set the number of processes in y direction")
        parser.add_argument("-npz", dest="npz", default=1,
                             help="Set the number of processes in z direction")
        args = parser.parse_args()
        self.npx = int(args.npx)
        self.npy = int(args.npy)
        self.npz = int(args.npz)

    def set_subarray(self, n, vmax, nv_on_beg=False):
        rank = self.comm.Get_rank()
        coords = self.comm.Get_coords(rank)

        if nv_on_beg:
            nloc = [i - 2*v for i, v in zip(n[1:], vmax)]
            nv = n[0]
        else:
            nloc = [i - 2*v for i, v in zip(n[:-1], vmax)]
            nv = n[-1]

        start_send = []
        start_recv = []
        msize = []
        stag = np.arange((3)**self.dim).reshape((3,)*self.dim)
        rtag = stag[::-1, ::-1]
        for i in xrange(self.dim):
            start_send.append([vmax[i], vmax[i], n[i]-2*vmax[i]])
            start_recv.append([0, vmax[i], n[i]-vmax[i]])
            msize.append([vmax[i], nloc[i], vmax[i]])
        start_send = np.asarray(start_send)
        start_recv = np.asarray(start_recv)
        msize = np.asarray(msize)

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
                    neighbor = self.comm.Get_cart_rank(coords + d)
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
                    self.sendTag.append(stag[d[0]+1, d[1]+1])
                    self.recvTag.append(rtag[d[0]+1, d[1]+1])
                    print "[{0}] send to {1} with tag {2} subarray:{3}".format(rank, neighbor, self.sendTag[-1], (n, ms, ss))
                    print "[{0}] recv from {1} with tag {2} subarray:{3}".format(rank, neighbor, self.recvTag[-1], (n, ms, sr))
                except mpi.Exception:
                    pass

        for s, r in zip(self.sendType, self.recvType):
            s.Commit()
            r.Commit()

    def update(self, f):
        req = []

        for i in xrange(len(self.recvType)):
            req.append(mpi.COMM_WORLD.Irecv([f, self.recvType[i]], source = self.neighbors[i], tag=self.recvTag[i]))

        for i in xrange(len(self.sendType)):
            req.append(mpi.COMM_WORLD.Isend([f, self.sendType[i]], dest = self.neighbors[i], tag=self.sendTag[i]))

        mpi.Request.Waitall(req)

def get_directions(dim):
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

if __name__ == '__main__':
    dim = 2
    period = [False]*dim
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()

    n = [10, 10, 2]
    vmax = [2, 2]

    f = rank*np.ones(n)

    print rank, f[:, :, 0]

    comm = mpi.COMM_WORLD
    size = comm.Get_size()
    split = mpi.Compute_dims(size, dim)
    comm = comm.Create_cart(split, period)

    i = Interface(dim, comm, n, vmax)

    i.update(f)

    print rank, f[:, :, 0].transpose()
