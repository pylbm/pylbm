from __future__ import print_function
from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

from past.builtins import cmp
from six.moves import range
import numpy as np
from math import sqrt
from textwrap import dedent

from .utils import itemproperty
from .geometry import get_box
from .logs import setLogger
from . import viewer
from future.utils import with_metaclass

def permute_in_place(a):
    """
    Function that returns an iterator of all the permutations of a list

    Parameters
    ----------

    a: list

    Returns
    -------

    Return successive permutations of elements in the list a

    The set of all the permutations is not created in the memory,
    so it can just be used in a loop.

    It can be used as permutations() of the itertools package but avoids the
    multiple occurences of a same output list.

    Examples
    --------

    .. code::python
        >>> import itertools
        >>> for k in itertools.permutations([0, 0, 1]):
        ...     print k
        ...
        (0, 0, 1)
        (0, 1, 0)
        (0, 0, 1)
        (0, 1, 0)
        (1, 0, 0)
        (1, 0, 0)

        >>> for k in permute_in_place([0, 0, 1]):
        ...     print k
        ...
        [0, 0, 1]
        [0, 1, 0]
        [1, 0, 0]
    """
    a.sort()
    yield list(a)

    if len(a) <= 1:
        return

    first = 0
    last = len(a)
    while 1:
        i = last - 1

        while 1:
            i = i - 1
            if a[i] < a[i + 1]:
                j = last - 1
                while not (a[i] < a[j]):
                    j = j - 1
                a[i], a[j] = a[j], a[i] # swap the values
                r = a[i + 1:last]
                r.reverse()
                a[i + 1:last] = r
                yield list(a)
                break
            if i == first:
                a.reverse()
                return

class Singleton(type):
    _instances = {}
    def __call__(self, *args, **kwargs):
        key = (self, args, str(kwargs))
        if key not in self._instances:
            self._instances[key] = super(Singleton, self).__call__(*args, **kwargs)
        return self._instances[key]

class Velocity(with_metaclass(Singleton, object)):
    """
    Create a velocity.

    Parameters
    ----------

    dim : int, optional
         The dimension of the velocity.
    num : int, optional
         The number of the velocity in the numbering convention of Lattice-Boltzmann scheme.
    vx : int, optional
         The x component of the velocity vector.
    vy : int, optional
         The y component of the velocity vector.
    vz : int, optional
         The z component of the velocity vector.

    Attributes
    ----------

    dim : int
        The dimension of the velocity.
    num
        The number of the velocity in the numbering convention of Lattice-Boltzmann scheme.
    vx : int
        The x component of the velocity vector.
    vy : int
        The y component of the velocity vector.
    vz : int
        The z component of the velocity vector.
    v : list

    Examples
    --------

    Create a velocity with the dimension and the number

    >>> v = Velocity(dim = 1, num = 2)
    >>> v
    velocity 2
     vx: -1

    Create a velocity with a direction

    >>> v = Velocity(vx=1, vy=1)
    >>> v
    velocity 5
     vx: 1
     vy: 1

    Notes
    ------

    .. plot:: codes/Velocities.py

    """
    _d = 1e3
    _R2 = np.array([[[5, 6, 4], [_d, _d, 2], [2, 5, 3]],
                    [[3, _d, _d], [_d, -1, _d], [1, _d, _d]],
                    [[6, 7, 7], [_d, _d, 0], [1, 4, 0]]], dtype=np.int)

    def __init__(self, dim=None, num=None, vx=None, vy=None, vz=None):
        self.log = setLogger(__name__)
        self.dim = dim
        self.num = num

        self.vx = vx
        self.vy = vy
        self.vz = vz

        if dim is None:
            if vz is not None:
                self.dim = 3
            elif vy is not None:
                self.dim = 2
            elif vx is not None:
                self.dim = 1
            else:
                raise ValueError("The parameters could not be all None when creating a velocity")

        if num is None:
            self._set_num()
        if vx is None:
            self._set_coord()

    @property
    def v(self):
        """
        velocity
        """
        return [self.vx, self.vy, self.vz][:self.dim]

    def __str__(self):
        s = '(%d: %d'%(self.num, self.vx)
        if self.vy is not None:
            s += ', %d'%self.vy
        if self.vz is not None:
            s += ', %d'%self.vz
        s += ')'
        return s

    def __repr__(self):
        return self.__str__()

    def get_symmetric(self, axis=None):
        """
        return the symmetric velocity.

        Parameters
        ----------

        axis : the axis of the symmetry, optional
          (None involves the symmetric with the origin,
          0 with the x axis, 1 with the y axis, and 2 with the z axis)

        Returns
        -------

        The symmetric of the velocity

        """
        if axis is not None and axis >= self.dim:
            self.log.error("axis must be less than the dimension of the velocity (axis:{0}, dim:{1})".format(axis, self.dim))
            raise ValueError

        svx = -self.vx
        svy = None if self.vy is None else -self.vy
        svz = None if self.vz is None else -self.vz

        if axis is None:
            return Velocity(vx=svx, vy=svy, vz=svz)
        if axis == 0:
            return Velocity(vx=self.vx, vy=svy, vz=svz)
        if axis == 1:
            return Velocity(vx=svx, vy=self.vy, vz=svz)
        if axis == 2:
            return Velocity(vx=svx, vy=svy, vz=self.vz)

    def set_symmetric(self):
        """
        create the symetric velocity.
        """
        self.numsym = [self.get_symmetric().num,]
        for i in range(self.dim):
            self.numsym.append(self.get_symmetric(i).num)

    def _set_num(self):
        if self.dim == 1:
            avx = abs(self.vx)
            self.num = (2*avx) - (1 if self.vx>0 else 0)
            return
        elif self.dim == 2:
            avx = abs(self.vx)
            avy = abs(self.vy)
            T1 = cmp(self.vx, 0)
            T2 = cmp(self.vy, 0)
            T3 = cmp(avx, avy)
            p = (2*max(avx, avy) - 1)
            p *= p
            q = 8*min(avx, avy)*abs(T3)
            r = self._R2[T1 + 1, T2 + 1, T3 + 1]
            self.num = int(p + q + r)
            return
        elif self.dim == 3:
            count = 0
            sign = [1, -1]
            for k in range(100):
                for i in range(k + 1):
                    for j in range(i + 1):
                        for (kk, ii, jj) in permute_in_place([k, i, j]):
                            for pmk in sign[0: kk + 1]: # loop over + and - if kk > 0
                                for pmi in sign[0:ii + 1]: # loop over + and - if ii > 0
                                    for pmj in sign[0:jj + 1]: # loop over + and - if jj > 0
                                        if self.vx == pmk*kk and self.vy == pmi*ii and self.vz == pmj*jj:
                                            self.num = count
                                            return
                                        else:
                                            count +=1
        self.log.error("The number of the velocity {0} is not found".format(self.__str__()))

    def _set_coord(self):
        if self.dim == 1:
            n = self.num + 1
            self.vx = int((1 - 2*(n % 2))*(n/2))
            return
        elif self.dim == 2:
            n = (int)((sqrt(self.num)+1)/2)
            p = self.num - (2*n-1)**2
            if (p<4):
                Lx, Ly = [n, 0, -n, 0], [0, n, 0, -n]
                vx, vy = Lx[p], Ly[p]
            elif (p<8):
                Lx, Ly = [n, -n, -n, n], [n, n, -n, -n]
                vx, vy = Lx[p-4], Ly[p-4]
            else:
                k, l = n, p/8
                Lx, Ly = [k, l, -l, -k, -k, -l, l, k], [l, k, k, l, -l, -k, -k, -l]
                vx, vy = Lx[p%8], Ly[p%8]
            self.vx = int(vx)
            self.vy = int(vy)
            return
        elif self.dim == 3:
            count = 0
            sign = [1, -1]
            for k in range(100):
                for i in range(k + 1):
                    for j in range(i + 1):
                        for (kk, ii, jj) in permute_in_place([k, i, j]):
                            for pmk in sign[0:kk + 1]: # loop over + and - if kk > 0
                                for pmi in sign[0:ii + 1]: # loop over + and - if ii > 0
                                    for pmj in sign[0:jj + 1]: # loop over + and - if jj > 0
                                        if self.num == count:
                                            self.vx = int(pmk*kk)
                                            self.vy = int(pmi*ii)
                                            self.vz = int(pmj*jj)
                                            return
                                        else:
                                            count +=1
        self.log.error("The velocity number {0} cannot be computed".format(self.num))

class OneStencil(object):
    """
    Create a stencil of a LBM scheme.

    Parameters
    ----------

    v : list
      the list of the velocities of that stencil
    nv : int
      size of the list
    num2index : list of integers
      link between the velocity number and its position in the unique
      velocities array

    Attributes
    ----------

    v : list
      the list of the velocities of that stencil
    nv : int
      size of the list v
    num2index : list of integers
      link between the velocity number and its position in the unique
      velocities array
    num
      the numbering of the velocities
    vx
      the x component of the velocities
    vy
      the y component of the velocities
    vz
    """

    def __init__(self, v, nv, num2index, nv_ptr):
        self.v = v
        self.nv = nv
        self.num2index = np.asarray(num2index) + nv_ptr
        self.nv_ptr = nv_ptr

    @property
    def num(self):
        """
        the numbering of the velocities.
        """
        vectorize = np.vectorize(lambda obj: obj.num)
        return vectorize(self.v)

    @property
    def vx(self):
        """
        the x component of the velocities.
        """
        vectorize = np.vectorize(lambda obj: obj.vx)
        return vectorize(self.v)

    @property
    def vy(self):
        """
        the y component of the velocities.
        """
        vectorize = np.vectorize(lambda obj: obj.vy)
        return vectorize(self.v)

    @property
    def vz(self):
        """
        the z component of the velocities.
        """
        vectorize = np.vectorize(lambda obj: obj.vz)
        return vectorize(self.v)


class Stencil(list):
    """
    Create the stencil of velocities used by the scheme(s).

    The numbering of the velocities follows the convention for 1D and 2D.

    Parameters
    ----------

    dico : a dictionary that contains the following `key:value`
      - dim : the value of the spatial dimension (1, 2 or 3)
      - schemes : a list of the dictionaries that contain the key:value velocities

          [{'velocities':[...]}, {'velocities':[...]}, {'velocities':[...]}, ...]

    Attributes
    ----------

    dim : int
        the spatial dimension (1, 2 or 3).
    unique_velocities : NumPy array
        array of all velocities involved in the stencils.
        Each unique velocity appeared only once.
    uvx : NumPy array
        the x component of the unique velocities.
    uvy : NumPy array
        the y component of the unique velocities.
    uvz : NumPy array
        the z component of the unique velocities.
    unum : NumPy array
        the numbering of the unique velocities.
    vmax : int
        the maximal velocity in norm for each spatial direction.
    vmin : int
        the minimal velocity in norm for each spatial direction.
    nstencils : int
        the number of elementary stencils.
    nv : list of integers
        the number of velocities for each elementary stencil.
    v : list of velocities
        list of all the velocities for each elementary stencil.
    vx : NumPy array
        the x component of the velocities for the stencil k.
    vy : NumPy array
        the y component of the velocities for the stencil k.
    vz : NumPy array
        the z component of the velocities for the stencil k.
    num : NumPy array
        the numbering of the velocities for the stencil k.
    nv_ptr : list of integers
        used to obtain the list of the velocities involved in a stencil.
        For instance, the list for the kth stencil is
            v[nv_ptr[k]:nv_ptr[k+1]]
    unvtot : int
        the number of unique velocities involved in the stencils.

    Notes
    -----

    The velocities for each schemes are defined as a Python list.

    Examples
    --------

    >>> s = Stencil({'dim': 1,
                 'schemes':[{'velocities': range(9)}, ],
                    })
    >>> s
    Stencil informations
      * spatial dimension: 1
      * maximal velocity in each direction: [4 None None]
      * minimal velocity in each direction: [-4 None None]
      * Informations for each elementary stencil:
            stencil 0
            - number of velocities:  9
            - velocities: (0: 0), (1: 1), (2: -1), (3: 2), (4: -2), (5: 3), (6: -3), (7: 4), (8: -4),

    >>> s = Stencil({'dim': 2,
                     'schemes':[{'velocities':range(9)},
                                {'velocities':range(50)},
                               ],
                    })
    >>> s
    Stencil informations
      * spatial dimension: 2
      * maximal velocity in each direction: [4 3 None]
      * minimal velocity in each direction: [-3 -3 None]
      * Informations for each elementary stencil:
            stencil 0
            - number of velocities:  9
            - velocities: (0: 0, 0), (1: 1, 0), (2: 0, 1), (3: -1, 0), (4: 0, -1), (5: 1, 1), (6: -1, 1), (7: -1, -1), (8: 1, -1),
            stencil 1
            - number of velocities: 50
            - velocities: (0: 0, 0), (1: 1, 0), (2: 0, 1), (3: -1, 0), (4: 0, -1), (5: 1, 1), (6: -1, 1), (7: -1, -1), (8: 1, -1), (9: 2, 0), (10: 0, 2), (11: -2, 0), (12: 0, -2), (13: 2, 2), (14: -2, 2), (15: -2, -2), (16: 2, -2), (17: 2, 1), (18: 1, 2), (19: -1, 2), (20: -2, 1), (21: -2, -1), (22: -1, -2), (23: 1, -2), (24: 2, -1), (25: 3, 0), (26: 0, 3), (27: -3, 0), (28: 0, -3), (29: 3, 3), (30: -3, 3), (31: -3, -3), (32: 3, -3), (33: 3, 1), (34: 1, 3), (35: -1, 3), (36: -3, 1), (37: -3, -1), (38: -1, -3), (39: 1, -3), (40: 3, -1), (41: 3, 2), (42: 2, 3), (43: -2, 3), (44: -3, 2), (45: -3, -2), (46: -2, -3), (47: 2, -3), (48: 3, -2), (49: 4, 0),

    get the x component of the unique velocities

    >>> s.uvx
    array([ 0,  1,  0, -1,  0,  1, -1, -1,  1,  2,  0, -2,  0,  2, -2, -2,  2,
            2,  1, -1, -2, -2, -1,  1,  2,  3,  0, -3,  0,  3, -3, -3,  3,  3,
            1, -1, -3, -3, -1,  1,  3,  3,  2, -2, -3, -3, -2,  2,  3,  4])

    get the y component of the velocity for the second stencil

    >>> s.vy[1]
    array([ 0,  0,  1,  0, -1,  1,  1, -1, -1,  0,  2,  0, -2,  2,  2, -2, -2,
            1,  2,  2,  1, -1, -2, -2, -1,  0,  3,  0, -3,  3,  3, -3, -3,  1,
            3,  3,  1, -1, -3, -3, -1,  2,  3,  3,  2, -2, -3, -3, -2,  0])
    """
    def __init__(self, dico):
        super(Stencil, self).__init__()
        self.log = setLogger(__name__)
        # get the dimension of the stencil (given in the dictionnary or computed
        # through the geometrical box)
        self.dim = dico.get('dim', None)

        box = dico.get('box', None)
        if box is not None:
            dbox, bounds = get_box(dico)

        if self.dim is None:
            self.dim = dbox

        if box is not None and dbox != self.dim:
            self.log.warning(dedent("""\
                             you define a scheme with dimension {0} and
                             a box with dimension {1}.
                             They must be the same.""".format(self.dim, dbox)))

        # get the schemes
        try:
            v_index = []
            schemes = dico['schemes']
            if not isinstance(schemes, list):
                self.log.error("The entry 'schemes' must be a list.")

            for s in schemes:
                # get the list of the velocities of each stencil
                v_index.append(np.asarray(s['velocities']))
        except:
            self.log.error("unable to read the scheme.")
        self.nstencils = len(v_index)

        # get the unique velocities involved in the stencil
        unique_indices = np.empty(0, dtype=np.int32)
        for vi in v_index:
            unique_indices = np.union1d(unique_indices, vi)

        self.unique_velocities = np.asarray([Velocity(dim=self.dim, num=i) for i in unique_indices])

        self.v = []
        self.nv = []
        self.nv_ptr = [0]
        num = self.unum
        for vi in v_index:
            ypos = np.searchsorted(num, vi)
            self.v.append(self.unique_velocities[ypos])
            lvi = len(vi)
            self.nv.append(lvi)
            self.nv_ptr.append(self.nv_ptr[-1] + lvi)
        self.nv_ptr = np.asarray(self.nv_ptr)

        # get the index in the v[k] of the num velocity
        self.num2index = []
        for k in range(self.nstencils):
            num = self.num[k]
            nmax = np.max(num)
            tmp = -1000 + np.zeros(nmax + 1, dtype=np.int32)
            tmp[num] = np.arange(num.size)
            #self.num2index.extend(tmp[tmp>=0])
            self.num2index.extend(num)

        # get the index in the v[k] of the num velocity (unique)
        unum = self.unum
        self.unum2index = -1000 + np.zeros(np.max(unum) + 1, dtype=np.int32)
        self.unum2index[unum] = np.arange(unum.size)

        for k in range(self.nstencils):
            self.append(OneStencil(self.v[k], self.nv[k], self.num2index[k], self.nv_ptr[k]))

        self.log.debug(self.__str__())
        self.is_symmetric()

    def unvtot(self):
        """the number of unique velocities involved in the stencils."""
        return self.unique_velocities.size

    unvtot = property(unvtot)

    @property
    def vmax(self):
        """the maximal velocity in norm for each spatial direction."""
        tmp = np.asarray([self.uvx, self.uvy, self.uvz])
        return np.max(tmp[:self.dim], axis=1)

    @property
    def vmin(self):
        """the minimal velocity in norm for each spatial direction."""
        tmp = np.asarray([self.uvx, self.uvy, self.uvz])
        return np.min(tmp[:self.dim], axis=1)

    @property
    def uvx(self):
        """the x component of the unique velocities."""
        vectorize = np.vectorize(lambda obj: obj.vx)
        return vectorize(self.unique_velocities)

    @itemproperty
    def vx(self, k):
        """vx[k] the x component of the velocities for the stencil k."""
        vectorize = np.vectorize(lambda obj: obj.vx)
        return vectorize(self[k])

    @property
    def uvy(self):
        """the y component of the unique velocities."""
        vectorize = np.vectorize(lambda obj: obj.vy)
        return vectorize(self.unique_velocities)

    @itemproperty
    def vy(self, k):
        """vy[k] the y component of the velocities for the stencil k."""
        vectorize = np.vectorize(lambda obj: obj.vy)
        return vectorize(self[k])

    @property
    def uvz(self):
        """the z component of the unique velocities."""
        vectorize = np.vectorize(lambda obj: obj.vz)
        return vectorize(self.unique_velocities)

    @itemproperty
    def vz(self, k):
        """vz[k] the z component of the velocities for the stencil k."""
        vectorize = np.vectorize(lambda obj: obj.vz)
        return vectorize(self[k])

    @property
    def unum(self):
        """the numbering of the unique velocities."""
        vectorize = np.vectorize(lambda obj: obj.num)
        return vectorize(self.unique_velocities)

    @itemproperty
    def num(self, k):
        """num[k] the numbering of the velocities for the stencil k.
        """
        vectorize = np.vectorize(lambda obj: obj.num)
        return vectorize(self.v[k])

    def get_all_velocities(self, ischeme=None):
        """
        get all the velocities for all the stencils in one array
        """
        if ischeme is None:
            size = self.nv_ptr[-1]
            allv = np.empty((size, self.dim), dtype='int')
            for iv, v in enumerate(self):
                vx = self.vx[iv]
                vy = self.vy[iv]
                vz = self.vz[iv]
                allv[self.nv_ptr[iv]:self.nv_ptr[iv+1], :] = np.asarray([vx, vy, vz][:self.dim]).T
            return allv
        else:
            vx = self.vx[ischeme]
            vy = self.vy[ischeme]
            vz = self.vz[ischeme]
            return np.asarray([vx, vy, vz][:self.dim]).T

    def get_symmetric(self, axis=None):
        """
        get the symetrics velocities.
        """
        ksym = np.empty(self.nv_ptr[-1], dtype=np.int32)

        k = 0
        for v in self.v:
            for vk in v:
                num = vk.get_symmetric(axis).num
                n = self.get_stencil(k)
                index = self.num2index[self.nv_ptr[n]:self.nv_ptr[n+1]].index(num) + self.nv_ptr[n]
                ksym[k] = index
                k += 1

        return ksym

    def get_stencil(self, k):
        n = 0
        while k >= self.nv_ptr[n+1]:
            n += 1
        return n

    def __str__(self):
        s = "Stencil informations\n"
        s += "\t * spatial dimension: {0:1d}\n".format(self.dim)
        s += "\t * maximal velocity in each direction: "
        s += str(self.vmax)
        s += "\n\t * minimal velocity in each direction: "
        s += str(self.vmin)
        s += "\n\t * Informations for each elementary stencil:\n"
        for k in range(self.nstencils):
            s += "\t\tstencil {0:1d}\n".format(k)
            s += "\t\t - number of velocities: {0:2d}\n".format(self.nv[k])
            s += "\t\t - velocities: "
            for v in self.v[k]:
                s += v.__str__() + ', '
            s += '\n'
        return s

    def __repr__(self):
        return self.__str__()

    def is_symmetric(self):
        """
        check if all the velocities have their symetric.
        """
        for n in range(self.nstencils):
            a = True
            A = np.array(self.num[n])
            for k in range(self.nv[n]):
                v = self.v[n][k]
                if np.all(np.where(A == v.get_symmetric().num, False, True)):
                    self.log.warning("The velocity " + v.__str__() + " has no symmetric velocity in the stencil {0:d}".format(n))
                    a = False
            if a:
                self.log.info("The stencil {0} is symmetric".format(n))
            else:
                self.log.warning("The stencil {0} is not symmetric".format(n))


    def visualize(self, viewer_mod=viewer.matplotlibViewer, k=None, unique_velocities=False):
        """
        plot the velocities

        Parameters
        ----------

        viewer : package used to plot the figure (could be matplotlib, ...)
            see viewer for more information
        k : list of stencil index to plot
            if None plot all stencils
        unique_velocities : if True plot the unique velocities

        """
        # if self.dim == 3 and not viewer.is3d:
        #     #raise ValueError("viewer doesn't support 3D visualization")
        #     self.log.error("viewer doesn't support 3D visualization")

        xmin = xmax = 0
        ymin = ymax = 0
        zmin = zmax = 0


        if unique_velocities:
            view = viewer_mod.Fig(figsize = (5, 5))
            ax = view[0]
            #ax.title = "unique_velocities"

            vectorize = np.vectorize(lambda txt, vx, vy, vz: ax.text(str(txt), [vx, vy, vz]))

            vx = self.uvx
            vy = vz = 0
            if self.dim >= 2:
                vy = self.uvy
            if self.dim == 3:
                vz = self.uvz

            pos = np.zeros((vx.size, 3))
            pos[:, 0] = vx
            pos[:, 1] = vy
            pos[:, 2] = vz

            ax.text(list(map(str, self.unum)), pos[:,:max(2,self.dim)])

            xmin, xmax = np.min(vx) - 1, np.max(vx) + 1
            ymin, ymax = np.min(vy) - 1, np.max(vy) + 1
            zmin, zmax = np.min(vz) - 1, np.max(vz) + 1
            ax.title = "Stencil of the unique velocities"
            if self.dim == 1:
                ax.axis(xmin, xmax, ymin, ymax)
                ax.xaxis(np.arange(xmin, xmax+1))
                ax.yaxis_set_visible(False)
            if self.dim == 2:
                ax.axis(xmin, xmax, ymin, ymax, aspect='equal')
                ax.xaxis(np.arange(xmin, xmax+1))
                ax.yaxis(np.arange(ymin, ymax+1))
            if self.dim == 3:
                ax.axis(xmin, xmax, ymin, ymax, zmin, zmax, self.dim, aspect='equal')
                ax.xaxis(np.arange(xmin, xmax+1))
                ax.yaxis(np.arange(ymin, ymax+1))
                ax.zaxis(np.arange(zmin, zmax+1))
            # if self.dim == 3:
            #     ax.axis(xmin, xmax, ymin, ymax, zmin, zmax)
            # else:
            #     ax.axis(xmin, xmax, ymin, ymax)
            #     ax.xaxis(np.arange(xmin, xmax+1))
            #     ax.yaxis_set_visible(False)
            ax.grid(visible=True, which='major', alpha=0.5)

        else:
            if k is None:
                lv = list(range(self.nstencils))
            elif isinstance(k, int):
                lv = [k]
            else:
                lv = k

            view = viewer_mod.Fig(len(lv), 1, dim = self.dim, figsize = (5, 5*len(lv)))
            view.fix_space(wspace=0.25, hspace=0.25)

            for ii, i in enumerate(lv):
                ax = view[ii]

                vx = self.vx[i]
                vy = vz = 0
                if self.dim >= 2:
                    vy = self.vy[i]
                if self.dim == 3:
                    vz = self.vz[i]

                pos = np.zeros((vx.size, 3))
                pos[:, 0] = vx
                pos[:, 1] = vy
                pos[:, 2] = vz

                xmin, xmax = np.min(vx) - 1, np.max(vx) + 1
                ymin, ymax = np.min(vy) - 1, np.max(vy) + 1
                zmin, zmax = np.min(vz) - 1, np.max(vz) + 1
                ax.title = "Stencil {0:d}".format(ii)
                if self.dim == 1:
                    ax.axis(xmin, xmax, ymin, ymax)
                    ax.xaxis(np.arange(xmin, xmax+1))
                    ax.yaxis_set_visible(False)
                    ax.plot([xmin, xmax], [0, 0], color = 'orange', alpha = 0.25)
                if self.dim == 2:
                    ax.axis(xmin, xmax, ymin, ymax, aspect='equal')
                    ax.xaxis(np.arange(xmin, xmax+1))
                    ax.yaxis(np.arange(ymin, ymax+1))
                    ax.plot([xmin, xmax], [0, 0], color = 'orange', alpha = 0.25)
                    ax.plot([0, 0], [ymin, ymax], color = 'orange', alpha = 0.25)
                if self.dim == 3:
                    ax.axis(xmin, xmax, ymin, ymax, zmin, zmax, self.dim, aspect='equal')
                    ax.xaxis(np.arange(xmin, xmax+1))
                    ax.yaxis(np.arange(ymin, ymax+1))
                    ax.zaxis(np.arange(zmin, zmax+1))
                    ax.plot([xmin, xmax], [0, 0], [0, 0], color = 'orange', alpha = 0.25)
                    ax.plot([0, 0], [ymin, ymax], [0, 0], color = 'orange', alpha = 0.25)
                    ax.plot([0, 0], [0, 0], [zmin, zmax], color = 'orange', alpha = 0.25)

                ax.text(list(map(str, self.num[i])), pos[:,:max(2,self.dim)], fontsize = 12, color = 'navy', fontweight='bold')
                ax.grid(visible=True, which='major', alpha=0.25)

        view.show()

if __name__ == '__main__':
    """
    d = {'dim': 3,
         'number_of_schemes': 3,
         0:{'velocities': range(19)},
         1:{'velocities': range(27)},
         2:{'velocities': [5, 39, 2]},
         }

    s = Stencil(d)

    v = viewer.MatplotlibViewer()

    print s.vx[0]
    print s.vy[0]
    print s.vz[0]

    print s.unum

    s.visualize(v)

    """
    d = {'dim': 2,
         'number_of_schemes': 3,
         0:{'velocities': list(range(5))},
         1:{'velocities': [0,2,4,5,1]},
         2:{'velocities': list(range(13))},
         }

    s = Stencil(d)

    #v = viewer.MatplotlibViewer()

    #for i in xrange(5):
    #    print s.get_index(1, i)

    print(s.vx[0])
    print(s.vy[0])
    print(s.vz[0])

    print(s.unum)

    #s.visualize(v, k=2)
