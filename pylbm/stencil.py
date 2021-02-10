# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause
"""
This module provides the classes to manage the velocities of
lattice Boltzmann schemes.
"""

from math import sqrt
import logging
import numpy as np

from .utils import itemproperty
from .geometry import get_box
from . import viewer
from .validator import validate

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


def cmp(a, b):
    """
    cmp function like in python 2.
    """
    return (a > b) - (a < b)


def permute_in_place(iterable):
    """
    Function that returns an iterator of all the permutations of a list

    Parameters
    ----------

    iterable: list

    Yield
    -----

    list
        successive permutations of elements in the list iterable

    Note
    ----

    The set of all the permutations is not created in the memory,
    so it can just be used in a loop.

    It can be used as permutations() of the itertools package but avoids the
    multiple occurrences of a same output list.

    Examples
    --------

    .. code::python
        >>> import itertools
        >>> for k in itertools.permutations([0, 0, 1]):
        ...     print(k)
        ...
        (0, 0, 1)
        (0, 1, 0)
        (0, 0, 1)
        (0, 1, 0)
        (1, 0, 0)
        (1, 0, 0)

        >>> for k in permute_in_place([0, 0, 1]):
        ...     print(k)
        ...
        [0, 0, 1]
        [0, 1, 0]
        [1, 0, 0]
    """
    iterable.sort()
    yield list(iterable)

    if len(iterable) <= 1:
        return

    first = 0
    last = len(iterable)
    while 1:
        i = last - 1

        while 1:
            i = i - 1
            if iterable[i] < iterable[i + 1]:
                j = last - 1
                while not iterable[i] < iterable[j]:
                    j = j - 1
                # swap the values
                iterable[i], iterable[j] = iterable[j], iterable[i]
                part = iterable[i + 1:last]
                part.reverse()
                iterable[i + 1:last] = part
                yield list(iterable)
                break
            if i == first:
                iterable.reverse()
                return


class Singleton(type):
    """
    Singleton metaclasss
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        key = (cls, args, str(kwargs))
        if key not in cls._instances:
            cls._instances[key] = super(
                Singleton, cls
            ).__call__(*args, **kwargs)
        return cls._instances[key]


class Velocity:
    """
    Create a velocity.

    Parameters
    ----------

    dim : int, optional
         The dimension of the velocity.
    num : int, optional
         The number of the velocity in the numbering convention
         of Lattice-Boltzmann scheme.
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
        The number of the velocity in the numbering convention
        of Lattice-Boltzmann scheme.
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

    >>> v = Velocity(dim=1, num=2)
    >>> v
    (2: -1)

    Create a velocity with a direction

    >>> v = Velocity(vx=1, vy=1)
    >>> v
    (5: 1, 1)

    Notes
    ------

    .. plot:: codes/Velocities.py

    """
    __metaclass__ = Singleton
    _d = 1e3
    _R2 = np.array(
        [
            [
                [5, 6, 4], [_d, _d, 2], [2, 5, 3]
            ],
            [
                [3, _d, _d], [_d, -1, _d], [1, _d, _d]
            ],
            [
                [6, 7, 7], [_d, _d, 0], [1, 4, 0]
            ]
        ],
        dtype=int
    )

    def __init__(self, dim=None, num=None, vx=None, vy=None, vz=None):
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
                err_msg = "The parameters could not be all None "
                err_msg += "when creating a velocity"
                log.error(err_msg)

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

    @property
    def v_full(self):
        """
        velocity filled with 0 in 1d and 2d
        """
        v_filled = [0]*3
        v_filled[:self.dim] = self.v
        return v_filled

    def __str__(self):
        output = '({:d}: {:d}'.format(self.num, self.vx)
        if self.vy is not None:
            output += ', {:d}'.format(self.vy)
        if self.vz is not None:
            output += ', {:d}'.format(self.vz)
        output += ')'
        return output

    def __repr__(self):
        return self.__str__()

    # pylint: disable=invalid-unary-operand-type
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

        Velocity
            The symmetric of the velocity

        Raises
        ------

        ValueError
            if axis is not None and axis < 0 or axis >= dim

        """
        if axis is not None and (axis >= self.dim or axis < 0):
            err_msg = "axis must be less than the dimension of the "
            err_msg += "velocity (axis={:d}, dim={:d})".format(axis, self.dim)
            log.error(err_msg)
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
        return Velocity(vx=svx, vy=svy, vz=self.vz)

    # pylint: disable=invalid-name, too-many-locals
    # pylint: disable=too-many-nested-blocks, too-complex
    def _set_num(self):
        # computes the number of the velocity
        # the coordinates being given
        if self.dim == 1:
            avx = abs(self.vx)
            self.num = (2*avx) - (1 if self.vx > 0 else 0)
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
                            # loop over + and - if kk > 0
                            for pmk in sign[0: kk + 1]:
                                # loop over + and - if ii > 0
                                for pmi in sign[0:ii + 1]:
                                    # loop over + and - if jj > 0
                                    for pmj in sign[0:jj + 1]:
                                        if self.vx == pmk*kk and \
                                           self.vy == pmi*ii and \
                                           self.vz == pmj*jj:
                                            self.num = count
                                            return
                                        else:
                                            count += 1
        log.error("The number of the velocity %s is not found", self.__str__())

    # pylint: disable=too-many-locals, too-many-branches, too-complex
    def _set_coord(self):
        # computes the coordinates of the velocity
        # the number being given
        if self.dim == 1:
            n = self.num + 1
            self.vx = int((1 - 2*(n % 2))*(n/2))
            return
        elif self.dim == 2:
            n = (int)((sqrt(self.num)+1)/2)
            p = self.num - (2*n-1)**2
            if p < 4:
                Lx, Ly = [n, 0, -n, 0], [0, n, 0, -n]
                vx, vy = Lx[p], Ly[p]
            elif p < 8:
                Lx, Ly = [n, -n, -n, n], [n, n, -n, -n]
                vx, vy = Lx[p-4], Ly[p-4]
            else:
                k, l = n, p/8
                Lx = [k, l, -l, -k, -k, -l, l, k]
                Ly = [l, k, k, l, -l, -k, -k, -l]
                vx, vy = Lx[p % 8], Ly[p % 8]
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
                            # loop over + and - if kk > 0
                            for pmk in sign[0:kk + 1]:
                                # loop over + and - if ii > 0
                                for pmi in sign[0:ii + 1]:
                                    # loop over + and - if jj > 0
                                    for pmj in sign[0:jj + 1]:
                                        if self.num == count:
                                            self.vx = int(pmk*kk)
                                            self.vy = int(pmi*ii)
                                            self.vz = int(pmj*jj)
                                            return
                                        else:
                                            count += 1
        log.error("The velocity number %d cannot be computed", self.num)


class OneStencil:
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
    num
      the numbering of the velocities
    vx
      the x component of the velocities
    vy
      the y component of the velocities
    vz
    """

    def __init__(self, v, nv):
        self.v = v
        self.nv = nv

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
        - schemes : a list of dictionaries that contain
                    the key:value velocities

        .. code-block:: python

            [
                {
                    'velocities': [...]
                },
                {
                    'velocities': [...]
                },
                {
                    'velocities': [...]
                },
                ...
            ]

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
    vmax_full : int
        the maximal velocity in norm for the 3 spatial direction
        even in dim 1 or 2.
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
    ...              'schemes':[{'velocities': list(range(9))},],
    ...             })
    >>> s
    +---------------------+
    | Stencil information |
    +---------------------+
        - spatial dimension: 1
        - minimal velocity in each direction: [-4]
        - maximal velocity in each direction: [4]
        - information for each elementary stencil:
            stencil 0
                - number of velocities: 9
                - velocities
                    (0: 0)
                    (1: 1)
                    (2: -1)
                    (3: 2)
                    (4: -2)
                    (5: 3)
                    (6: -3)
                    (7: 4)
                    (8: -4)
    >>> s = Stencil({'dim': 2,
    ...              'schemes':[
                                   {'velocities': list(range(9))},
    ...                            {'velocities': list(range(49))},
    ...                        ],
    ...            })
    >>> s
    +---------------------+
    | Stencil information |
    +---------------------+
        - spatial dimension: 2
        - minimal velocity in each direction: [-3 -3]
        - maximal velocity in each direction: [3 3]
        - information for each elementary stencil:
            stencil 0
                - number of velocities: 9
                - velocities
                    (0: 0, 0)
                    (1: 1, 0)
                    (2: 0, 1)
                    (3: -1, 0)
                    (4: 0, -1)
                    (5: 1, 1)
                    (6: -1, 1)
                    (7: -1, -1)
                    (8: 1, -1)
            stencil 1
                - number of velocities: 49
                - velocities
                    (0: 0, 0)
                    (1: 1, 0)
                    (2: 0, 1)
                    (3: -1, 0)
                    (4: 0, -1)
                    (5: 1, 1)
                    (6: -1, 1)
                    (7: -1, -1)
                    (8: 1, -1)
                    (9: 2, 0)
                    (10: 0, 2)
                    (11: -2, 0)
                    (12: 0, -2)
                    (13: 2, 2)
                    (14: -2, 2)
                    (15: -2, -2)
                    (16: 2, -2)
                    (17: 2, 1)
                    (18: 1, 2)
                    (19: -1, 2)
                    (20: -2, 1)
                    (21: -2, -1)
                    (22: -1, -2)
                    (23: 1, -2)
                    (24: 2, -1)
                    (25: 3, 0)
                    (26: 0, 3)
                    (27: -3, 0)
                    (28: 0, -3)
                    (29: 3, 3)
                    (30: -3, 3)
                    (31: -3, -3)
                    (32: 3, -3)
                    (33: 3, 1)
                    (34: 1, 3)
                    (35: -1, 3)
                    (36: -3, 1)
                    (37: -3, -1)
                    (38: -1, -3)
                    (39: 1, -3)
                    (40: 3, -1)
                    (41: 3, 2)
                    (42: 2, 3)
                    (43: -2, 3)
                    (44: -3, 2)
                    (45: -3, -2)
                    (46: -2, -3)
                    (47: 2, -3)
                    (48: 3, -2)

    get the x component of the unique velocities

    >>> s.uvx
    array([ 0,  1,  0, -1,  0,  1, -1, -1,  1,  2,  0, -2,  0,  2, -2, -2,  2,
            2,  1, -1, -2, -2, -1,  1,  2,  3,  0, -3,  0,  3, -3, -3,  3,  3,
            1, -1, -3, -3, -1,  1,  3,  3,  2, -2, -3, -3, -2,  2,  3])

    get the y component of the velocity for the second stencil

    >>> s.vy[1]
    array([ 0,  0,  1,  0, -1,  1,  1, -1, -1,  0,  2,  0, -2,  2,  2, -2, -2,
            1,  2,  2,  1, -1, -2, -2, -1,  0,  3,  0, -3,  3,  3, -3, -3,  1,
            3,  3,  1, -1, -3, -3, -1,  2,  3,  3,  2, -2, -3, -3, -2])

    """
    def __init__(self, dico, need_validation=True):
        super(Stencil, self).__init__()

        if need_validation:
            validate(dico, __class__.__name__) #pylint: disable=undefined-variable

        # get the dimension of the stencil
        # (given in the dictionnary or computed
        #  through the geometrical box)
        self.dim = self.extract_dim(dico)

        # get the schemes
        schemes_velocities = []
        schemes = dico['schemes']
        for scheme in schemes:
            schemes_velocities.append(np.asarray(scheme['velocities']))
        self.nstencils = len(schemes_velocities)

        # get the unique velocities involved in the stencil
        unique_indices = np.empty(0, dtype=np.int32)
        for velocities in schemes_velocities:
            unique_indices = np.union1d(unique_indices, velocities)
        self.unique_velocities = np.asarray(
            [Velocity(dim=self.dim, num=i) for i in unique_indices]
        )

        self.v = []
        self.nv = []
        self.nv_ptr = [0]
        num = self.unum
        for velocities in schemes_velocities:
            ypos = np.searchsorted(num, velocities)
            self.v.append(self.unique_velocities[ypos])
            size = len(velocities)
            self.nv.append(size)
            self.nv_ptr.append(self.nv_ptr[-1] + size)
        self.nv_ptr = np.asarray(self.nv_ptr)

        # get the index in the v[k] of the num velocity
        self.num2index = []
        for k in range(self.nstencils):
            num = self.num[k]
            self.num2index.extend(num)

        # get the index in the v[k] of the num velocity (unique)
        unum = self.unum
        self.unum2index = -1000 + np.zeros(np.max(unum) + 1, dtype=np.int32)
        self.unum2index[unum] = np.arange(unum.size)

        for k in range(self.nstencils):
            self.append(OneStencil(self.v[k], self.nv[k]))

        log.debug(self.__str__())

        # check if all the schemes are symmetric
        self.is_symmetric()

    @staticmethod
    def extract_dim(dico):
        """Extract the dimension from the dictionary"""
        dim = dico.get('dim', None)

        if not dim:
            dim, _ = get_box(dico)

        return dim

    @property
    def unvtot(self):
        """the number of unique velocities involved in the stencils."""
        return self.unique_velocities.size

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
    def vmax_full(self):
        """
        the maximal velocity in norm for each spatial direction.
        all the three dimensions are considered
        even if dim = 1 or 2
        """
        tmp = np.array([0, 0, 0])
        tmp[:self.dim] = self.vmax
        return tmp

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

    def get_all_velocities(self, scheme_id=None):
        """
        get all the velocities for all the stencils in one array

        Parameters
        ----------
        scheme_id: int
            specify for which scheme we want all velocities
            if None, return the velocities for all the schemes

        Returns
        -------

        ndarray
            all velocities of a scheme or of all the schemes

        """
        if scheme_id is None:
            size = self.nv_ptr[-1]
            all_velocities = np.empty((size, self.dim), dtype='int')
            for vind in range(len(self)):
                vx = self.vx[vind]
                vy = self.vy[vind]
                vz = self.vz[vind]
                all_velocities[
                    self.nv_ptr[vind]:self.nv_ptr[vind+1], :
                ] = np.asarray([vx, vy, vz][:self.dim]).T
        else:
            vx = self.vx[scheme_id]
            vy = self.vy[scheme_id]
            vz = self.vz[scheme_id]
            all_velocities = np.asarray([vx, vy, vz][:self.dim]).T
        return all_velocities

    def get_symmetric(self, axis=None):
        """
        get the symmetric velocities.
        """
        ksym = np.empty(self.nv_ptr[-1], dtype=np.int32)

        k = 0
        for v in self.v:
            for vk in v:
                num = vk.get_symmetric(axis).num
                n = np.searchsorted(self.nv_ptr, k, side='right') - 1
                index = self.num2index[
                    self.nv_ptr[n]:self.nv_ptr[n+1]
                ].index(num) + self.nv_ptr[n]
                ksym[k] = index
                k += 1

        return ksym

    def __str__(self):
        from .utils import header_string
        from .jinja_env import env
        template = env.get_template('stencil.tpl')
        return template.render(
            header=header_string('Stencil information'),
            stencil=self
        )

    def __repr__(self):
        return self.__str__()

    def is_symmetric(self):
        """
        check if all the velocities have their symmetric.
        """
        for n in range(self.nstencils):
            is_sym = True
            scheme_velocities = np.array(self.num[n])
            for k in range(self.nv[n]):
                v = self.v[n][k]
                contains_sym = np.all(np.where(
                    scheme_velocities == v.get_symmetric().num, False, True
                ))
                if contains_sym:
                    err_msg = "The velocity {0} ".format(v)
                    err_msg += "has no symmetric velocity "
                    err_msg += "in the stencil {0:d}".format(n)
                    log.warning(err_msg)
                    is_sym = False
            if is_sym:
                log.info("The stencil %d is symmetric", n)
            else:
                log.warning("The stencil %d is not symmetric", n)
        return is_sym

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    # pylint: disable=too-complex
    def visualize(self,
                  viewer_mod=viewer.matplotlib_viewer,
                  k=None,
                  unique_velocities=False,
                  view_label=True):
        """
        plot the velocities

        Parameters
        ----------

        viewer : package used to plot the figure (could be matplotlib, ...)
            see viewer for more information
        k : list of stencil index to plot
            if None plot all stencils
        unique_velocities : if True plot the unique velocities
        view_label: if True show the velocity numbering

        Returns
        -------

        fig
            the figure (fig if matplotlib is used)

        """
        def populate(vx, vy, vz):
            dummy = np.asarray([vx, vy, vz][:self.dim])
            if self.dim == 1:
                pos = np.zeros((dummy.size, 2))
                pos[:, 0] = dummy
            else:
                pos = dummy.T
            return pos

        pos = []
        title = []
        if unique_velocities:
            pos.append(populate(self.uvx, self.uvy, self.uvz))
            title.append("Stencil of the unique velocities")
        else:
            if k is None:
                schemes2plot = list(range(self.nstencils))
            elif isinstance(k, int):
                schemes2plot = [k]
            else:
                schemes2plot = k
            for i in schemes2plot:
                pos.append(populate(self.vx[i], self.vy[i], self.vz[i]))
                title.append("Stencil {0:d}".format(i))

        views = viewer_mod.Fig(
            len(pos), 1, dim=self.dim, figsize=(5, 5*len(pos))
        )
        views.fix_space(wspace=0.25, hspace=0.25)

        for i, posi in enumerate(pos):
            view = views[i]

            pminmax = np.zeros(6 if self.dim == 3 else 4)
            pminmax[::2] = np.min(posi, axis=0) - 1
            pminmax[1::2] = np.max(posi, axis=0) + 1
            view.axis(*pminmax, dim=self.dim, aspect='equal')

            view.title = title[i]
            view.xaxis(np.arange(pminmax[0], pminmax[1] + 1))
            if self.dim == 1:
                view.yaxis_set_visible(False)
            view.plot(pminmax[:2], [0, 0], color='orange', alpha=0.25)

            if self.dim >= 2:
                view.yaxis(np.arange(pminmax[2], pminmax[3] + 1))
                view.plot([0, 0], pminmax[2:4], color='orange', alpha=0.25)
            if self.dim == 3:
                view.zaxis(np.arange(pminmax[4], pminmax[5] + 1))
                view.plot(
                    [0, 0], [0, 0],
                    pminmax[4:], color='orange', alpha=0.25
                )

            if view_label:
                view.text(list(map(str, self.num[i])), posi,
                          fontsize=12, color='navy', fontweight='bold')
            view.grid(visible=True, which='major', alpha=0.25)

        return views
