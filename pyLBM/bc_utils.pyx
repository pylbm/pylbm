#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True



def test1(double[:, :, ::1] f, double[:, ::1] feq, double[:] distance, int[:, ::1] indices, int k, int ksym, int vx, int vy):
    cdef:
        int i, ix, iy
        int n = distance.size

    for i in xrange(n):
        ix = indices[1, i]
        iy = indices[0, i]
        s = distance[i]

        if s < .5:
            s *= 2
            f[iy, ix, k] = s*f[iy + vy, ix + vx, ksym] + (1.-s)*f[iy + 2*vy, ix + 2*vx, ksym] + feq[i, k] - feq[i, ksym]
        else:
            s /= 2
            f[iy, ix, k] = s*f[iy + vy, ix + vx, ksym] + (1.-s)*f[iy + vy, ix + vx, k] + feq[i, k] - feq[i, ksym]

def test2(double[:, :, ::1] f, double[:] distance, int[:, ::1] indices, int k, int ksym, int vx, int vy):
    cdef:
        int i, ix, iy
        int n = distance.size
        double s

    for i in xrange(n):
        ix = indices[1, i]
        iy = indices[0, i]
        s = distance[i]

        if s < .5:
            s *= 2
            f[iy, ix, k] = s*f[iy + vy, ix + vx, ksym] + (1.-s)*f[iy + 2*vy, ix + 2*vx, ksym]
        else:
            s /= 2
            f[iy, ix, k] = s*f[iy + vy, ix + vx, ksym] + (1.-s)*f[iy + vy, ix + vx, k]

def test3feq(double[:, :, ::1] f, double[:, ::1] feq, double[:] s, int[:] ix, int[:] iy, int k, int ksym, int vx, int vy):
    cdef:
        int i
        int n = s.size

    for i in xrange(n):
        f[iy[i], ix[i], k] = s[i]*f[iy[i] + vy, ix[i] + vx, ksym] + (1.-s[i])*f[iy[i] + 2*vy, ix[i] + 2*vx, ksym] + feq[i, k] - feq[i, ksym]

def test3(double[:, :, ::1] f,double[:] s, int[:] ix, int[:] iy, int k, int ksym, int vx, int vy):
    cdef:
        int i
        int n = s.size

    for i in xrange(n):
        f[iy[i], ix[i], k] = s[i]*f[iy[i] + vy, ix[i] + vx, ksym] + (1.-s[i])*f[iy[i] + 2*vy, ix[i] + 2*vx, ksym]

def test4feq(double[:, :, ::1] f, double[:, ::1] feq, double[:] s, int[:] ix, int[:] iy, int k, int ksym, int vx, int vy):
    cdef:
        int i
        int n = s.size

    for i in xrange(n):
        f[iy[i], ix[i], k] = s[i]*f[iy[i] + vy, ix[i] + vx, ksym] + (1.-s[i])*f[iy[i] + vy, ix[i] + vx, k] + feq[i, k] - feq[i, ksym]

def test4(double[:, :, ::1] f,double[:] s, int[:] ix, int[:] iy, int k, int ksym, int vx, int vy):
    cdef:
        int i
        int n = s.size

    for i in xrange(n):
        f[iy[i], ix[i], k] = s[i]*f[iy[i] + vy, ix[i] + vx, ksym] + (1.-s[i])*f[iy[i] + vy, ix[i] + vx, k]
