
class interface:
    def __init__(self, dim, comm, n, vmax, nv_on_beg=False):
        self.com = com
        self.dim = dim
        self.sendType = []
        self.recvType = []

        rank = self.comm.Get_rank()
        coords = self.comm.Get_coords(rank)
        direction = np.empty((3,)*dim)

        start = []
        size = []
        for i in xrange(dim):
            start.append([vmax[i], vmax[i], n[i]-1])
            size.append([vmax[i], n[i], vmax[i]])

        if nv_on_beg:
            nv = n[0]
        else:
            nv = n[-1]

        if dim == 2:
            for i, vi in enumerate([-1, 0, 1]):
                for j, vj in enumerate([-1, 0, 1]):
                    if i != 0 and j != 0:
                    neighbor = comm.Get_cart_rank(coords + np.array([vi, vj]))
                    self.neighbors.append(neighbor)
                    if nv_on_beg:
                        ss = [nv, size[j], size[i]]
                        st = [nv, start[j], start[i]]
                    else:
                        ss = [size[j], size[i], nv]
                        st = [start[j], start[i], nv]

                    self.sendType.append(mpi.DOUBLE.Create_subarray(n, ss, st))
