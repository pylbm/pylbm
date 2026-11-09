# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Module for LBM boundary conditions
"""

import collections
import logging
import warnings
import numpy as np
from sympy import symbols, IndexedBase, Idx, Eq

from .storage import Array

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


# pylint: disable=too-few-public-methods
class BoundaryVelocity:
    """
    Indices and distances for the label and the velocity ksym
    """

    def __init__(self, domain, label, ksym):
        # We are looking for the points on the outside that have a speed
        # that goes in (index ksym) on a border labeled by label.
        # We go through all the lattice velocities and determine the inner points
        # that have the symmetric lattice velocity (index k) that comes out
        # then we write in a list with the order of the lattice velocities
        # involved in the schemes:
        # - indices of the corresponding external points
        # - associated distances
        self.label = label
        self.v = domain.stencil.unique_velocities[ksym]
        v = self.v.get_symmetric()
        num = domain.stencil.unum2index[v.num]

        ind = np.where(domain.flag[num] == self.label)
        self.indices = np.array(ind)
        if self.indices.size != 0:
            self.indices += np.asarray(v.v)[:, np.newaxis]
        self.distance = np.array(domain.distance[(num,) + ind])
        self.normal = np.array(domain.normal[(num,) + ind])  #


class Boundary:
    """
    Construct the boundary problem by defining the list of indices on the border and the methods used on each label.

    Parameters
    ----------
    domain : pylbm.Domain
        the simulation domain
    dico : dictionary
        describes the boundaries
            - key is a label
            - value are again a dictionnary with
                + "method" key that gives the boundary method class used (Bounce_back, Anti_bounce_back, ...)
                + "value_bc" key that gives the value on the boundary

    Attributes
    ----------
    bv_per_label : dictionnary
        for each label key, a list of spatial indices and distance define for each velocity the points
        on the domain that are on the boundary.

    methods : list
        list of boundary methods used in the LBM scheme
        The list contains Boundary_method instance.

    """

    # pylint: disable=too-many-locals
    def __init__(self, domain, generator, dico):
        self.domain = domain

        # build the list of indices for each unique velocity and for each label
        self.bv_per_label = {}
        for label in self.domain.list_of_labels():
            if label in [-1, -2]:  # periodic or interface conditions
                continue
            dummy_bv = []
            for k in range(self.domain.stencil.unvtot):
                dummy_bv.append(BoundaryVelocity(self.domain, label, k))
            self.bv_per_label[label] = dummy_bv

        # build the list of boundary informations for each stencil and each label
        dico_bound = dico.get("boundary_conditions", {})
        stencil = self.domain.stencil

        istore = (
            collections.OrderedDict()
        )  # important to set the boundary conditions always in the same way !!!
        ilabel = {}
        distance = {}
        value_bc = {}
        time_bc = {}
        normal = {}

        # pylint: disable=too-many-nested-blocks
        for label in self.domain.list_of_labels():
            if label in [-1, -2]:  # periodic or interface conditions
                continue

            value_bc[label] = dico_bound[label].get("value", None)
            time_bc[label] = dico_bound[label].get("time_bc", False)
            methods = dico_bound[label]["method"]
            # for each method get the list of points, the labels and the distances
            # where the distribution function must be updated on the boundary
            for k, v in methods.items():
                for inumk, numk in enumerate(stencil.num[k]):
                    if (
                        self.bv_per_label[label][stencil.unum2index[numk]].indices.size
                        != 0
                    ):
                        indices = self.bv_per_label[label][
                            stencil.unum2index[numk]
                        ].indices
                        distance_tmp = self.bv_per_label[label][
                            stencil.unum2index[numk]
                        ].distance
                        normal_tmp = self.bv_per_label[label][
                            stencil.unum2index[numk]
                        ].normal
                        velocity = (inumk + stencil.nv_ptr[k]) * np.ones(
                            indices.shape[1], dtype=np.int32
                        )[np.newaxis, :]
                        ilabel_tmp = label * np.ones(indices.shape[1], dtype=np.int32)
                        istore_tmp = np.concatenate([velocity, indices])
                        if istore.get(v, None) is None:
                            istore[v] = istore_tmp.copy()
                            ilabel[v] = ilabel_tmp.copy()
                            distance[v] = distance_tmp.copy()
                            normal[v] = normal_tmp.copy()
                        else:
                            istore[v] = np.concatenate([istore[v], istore_tmp], axis=1)
                            ilabel[v] = np.concatenate([ilabel[v], ilabel_tmp])
                            distance[v] = np.concatenate([distance[v], distance_tmp])
                            normal[v] = np.concatenate([normal[v], normal_tmp])  #

        # for each method create the instance associated
        self.methods = []
        for k in list(istore.keys()):
            self.methods.append(
                k(
                    istore[k],
                    ilabel[k],
                    distance[k],
                    normal[k],
                    stencil,
                    value_bc,
                    time_bc,
                    domain.distance.shape,
                    generator,
                )
            )


# pylint: disable=protected-access
class BoundaryMethod:
    """
    Set boundary method.

    Parameters
    ----------
    FIXME : add parameters documentation

    Attributes
    ----------
    feq : ndarray
        the equilibrium values of the distribution function on the border
    rhs : ndarray
        the additional terms to fix the boundary values
    distance : ndarray
        distance to the border (needed for Bouzidi type conditions)
    istore : ndarray
        indices of points where we store the boundary condition
    ilabel : ndarray
        label of the boundary
    iload : list
        indices of points needed to compute the boundary condition
    value_bc : dictionnary
       the prescribed values on the border

    """

    def __init__(
        self,
        istore,
        ilabel,
        distance,
        normal,
        stencil,
        value_bc,
        time_bc,
        nspace,
        generator,
    ):
        self.istore = istore
        self.feq = np.zeros((stencil.nv_ptr[-1], istore.shape[1]))
        self.rhs = np.zeros(istore.shape[1])
        self.ilabel = ilabel
        self.distance = distance
        self.normal = normal
        self.stencil = stencil
        self.time_bc = {}
        self.value_bc = {}
        for k in np.unique(self.ilabel):
            self.value_bc[k] = value_bc[k]
            self.time_bc[k] = time_bc[k]
        self.iload = []
        self.nspace = nspace
        self.generator = generator
        # copies on the device of the arrays of _host_args (Loopy backend)
        self._device_args = None

        # used if time boundary
        self.func = []
        self.args = []
        self.f = []
        self.m = []
        self.indices = []

    @classmethod
    def reads_previous_populations(cls):
        """
        Return True if the condition reads the populations of the previous
        time step: the simulation then passes them to update as fprevious.
        """
        return False

    def init_previous_populations(self, container):
        """
        Prepare the populations of the previous time step before the first
        one. Nothing to do for a condition that does not read them.
        """

    def fix_iload(self):
        """
        Transpose iload and istore.

        Must be fix in a future version.
        """
        # Fixme : store in a good way and in the right type istore and iload
        for i in range(len(self.iload)):
            self.iload[i] = np.ascontiguousarray(self.iload[i].T, dtype=np.int32)
        self.istore = np.ascontiguousarray(self.istore.T, dtype=np.int32)

    # pylint: disable=too-many-locals
    def prepare_rhs(self, simulation):
        """
        Compute the distribution function at the equilibrium with the value on the border.

        Parameters
        ----------
        simulation : Simulation
            simulation class

        """

        nv = simulation.container.nv
        sorder = simulation.container.sorder
        nspace = [1] * (len(sorder) - 1)
        v = self.stencil.get_all_velocities()

        gpu_support = simulation.container.gpu_support

        for key, value in self.value_bc.items():
            if value is not None:
                indices = np.where(self.ilabel == key)
                # TODO: check the index in sorder to be the most contiguous
                nspace[0] = indices[0].size
                k = self.istore[0, indices]

                s = 1 - self.distance[indices]
                coords = tuple()
                for i in range(simulation.domain.dim):
                    x = simulation.domain.coords_halo[i][self.istore[i + 1, indices]]
                    x += s * v[k, i] * simulation.domain.dx
                    x = x.ravel()
                    for j in range(
                        1, simulation.domain.dim
                    ):  # pylint: disable=unused-variable
                        x = x[:, np.newaxis]
                    coords += (x,)

                m = Array(
                    simulation.container.nm,
                    nspace,
                    0,
                    sorder,
                    gpu_support=gpu_support,
                )
                m.set_conserved_moments(simulation.scheme.consm)

                f = Array(nv, nspace, 0, sorder, gpu_support=gpu_support)
                f.set_conserved_moments(simulation.scheme.consm)

                args = coords
                func = value
                if isinstance(value, tuple):
                    func = value[0]
                    args += value[1]

                if self.time_bc[key]:
                    func(f, m, 0, *args)
                else:
                    func(f, m, *args)

                simulation.equilibrium(m)
                simulation.m2f(m, f)

                if self.generator.backend.upper() == "LOOPY":
                    f.array_cpu[...] = f.array.get()

                self.feq[:, indices[0]] = f.swaparray.reshape((nv, indices[0].size))

                if self.time_bc[key]:
                    self.func.append(func)
                    self.args.append(args)
                    self.f.append(f)
                    self.m.append(m)
                    self.indices.append(indices[0])

    def update_feq(self, simulation):
        t = simulation.t
        nv = simulation.container.nv

        for i in range(len(self.func)):
            self.func[i](self.f[i], self.m[i], t, *self.args[i])
            simulation.equilibrium(self.m[i])
            simulation.m2f(self.m[i], self.f[i])

            if self.generator.backend.upper() == "LOOPY":
                self.f[i].array_cpu[...] = self.f[i].array.get()

            self.feq[:, self.indices[i]] = self.f[i].swaparray.reshape(
                (nv, self.indices[i].size)
            )

    def _get_istore_iload_symb(self, dim):
        ncond = symbols("ncond", integer=True)

        istore = symbols("istore", integer=True)
        istore = IndexedBase(istore, [ncond, dim + 1])

        iload = []
        for i in range(len(self.iload)):
            iloads = symbols("iload%d" % i, integer=True)
            iload.append(IndexedBase(iloads, [ncond, dim + 1]))
        return istore, iload, ncond

    @staticmethod
    def _get_rhs_dist_symb(ncond):
        rhs = IndexedBase("rhs", [ncond])
        dist = IndexedBase("dist", [ncond])
        return rhs, dist

    def update(self, ff, **kwargs):
        """
        Update distribution functions with this boundary condition.

        Parameters
        ----------

        ff : array
            The distribution functions
        """
        from .symbolic import call_genfunction

        args = self._get_args(ff)
        args.update(kwargs)
        call_genfunction(self.function, args)  # pylint: disable=no-member

    def _host_args(self):
        """
        Return the arrays of the boundary that the generated function may
        need, keyed by the names used in the generated code. They stay on
        the host: set_rhs and update_feq compute with them at each step.
        """
        args = {f"iload{i}": iload for i, iload in enumerate(self.iload)}
        args["istore"] = self.istore
        args["rhs"] = self.rhs
        if hasattr(self, "s"):
            args["dist"] = self.s
        return args

    def _get_args(self, ff):
        """
        Return the arguments that the generated boundary function may need,
        keyed by the names used in the generated code.
        """
        args = {"f": ff.array}
        for name, size in zip(["nx", "ny", "nz"], ff.nspace):
            args[name] = size

        if self._device_args is None:
            args.update(self._host_args())
        else:
            # rhs is computed on the host at each step
            self._device_args["rhs"].set(self.rhs)
            args.update(self._device_args)
        args["ncond"] = self.istore.shape[0]
        return args

    def move2gpu(self):
        """
        Copy the arrays needed to compute the boundary to the GPU memory.
        """
        if self.generator.backend.upper() == "LOOPY":
            try:
                import pyopencl as cl
                import pyopencl.array  # pylint: disable=unused-variable
                from .context import queue
            except ImportError:
                raise ImportError("Please install pylbm[gpu]")

            self._device_args = {
                name: cl.array.to_device(queue, array)
                for name, array in self._host_args().items()
            }


class BounceBack(BoundaryMethod):
    """
    Boundary condition of type bounce-back

    Notes
    ------

    .. plot:: codes/bounce_back.py

    """

    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k][np.newaxis, :]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:] + v[k].T
        self.iload.append(np.concatenate([ksym, indices]))

    def set_rhs(self):
        """
        Compute and set the additional terms to fix the boundary values.
        """
        k = self.istore[:, 0]
        ksym = self.stencil.get_symmetric()[k]
        self.rhs[:] = self.feq[k, np.arange(k.size)] - self.feq[ksym, np.arange(k.size)]

    # pylint: disable=too-many-locals
    def generate(self, sorder):
        """
        Generate the numerical code.

        Parameters
        ----------
        sorder : list
            the order of nv, nx, ny and nz
        """
        from .generator import For
        from .symbolic import nx, ny, nz, indexed, ix

        ns = int(self.stencil.nv_ptr[-1])
        dim = self.stencil.dim

        istore, iload, ncond = self._get_istore_iload_symb(dim)
        rhs, _ = self._get_rhs_dist_symb(ncond)

        idx = Idx(ix, (0, ncond))
        fstore = indexed(
            "f",
            [ns, nx, ny, nz],
            index=[istore[idx, k] for k in range(dim + 1)],
            priority=sorder,
        )
        fload = indexed(
            "f",
            [ns, nx, ny, nz],
            index=[iload[0][idx, k] for k in range(dim + 1)],
            priority=sorder,
        )

        self.generator.add_routine(
            ("bounce_back", For(idx, Eq(fstore, fload + rhs[idx])))
        )

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.bounce_back


class BFLBounceBack(BoundaryMethod):
    """
    Boundary condition of type Bouzidi-Firdaouss-Lallemand bounce-back
    [BFL01]

    .. versionchanged:: 0.12.0
       Renamed from ``BouzidiBounceBack``, which is deprecated.

    Notes
    ------

    .. plot:: codes/Bouzidi.py

    """

    def __init__(
        self,
        istore,
        ilabel,
        distance,
        normal,
        stencil,
        value_bc,
        time_bc,
        nspace,
        generator,
    ):
        super().__init__(
            istore,
            ilabel,
            distance,
            normal,
            stencil,
            value_bc,
            time_bc,
            nspace,
            generator,
        )
        self.s = np.empty(self.istore.shape[1])
        # factor of the wall term rhs: 1 when the wall is closer than half
        # a link to the fluid node, 1/(2 distance) otherwise
        self.rhs_scale = np.empty(self.istore.shape[1])

    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k]
        v = self.stencil.get_all_velocities()

        iload1 = np.zeros(self.istore.shape, dtype=np.int32)
        iload2 = np.zeros(self.istore.shape, dtype=np.int32)

        mask = self.distance < 0.5
        iload1[0, mask] = ksym[mask]
        iload2[0, mask] = ksym[mask]
        iload1[1:, mask] = self.istore[1:, mask] + v[k[mask]].T
        iload2[1:, mask] = self.istore[1:, mask] + 2 * v[k[mask]].T
        self.s[mask] = 2.0 * self.distance[mask]
        self.rhs_scale[mask] = 1.0

        mask = np.logical_not(mask)
        iload1[0, mask] = ksym[mask]
        iload2[0, mask] = k[mask]
        iload1[1:, mask] = self.istore[1:, mask] + v[k[mask]].T
        iload2[1:, mask] = self.istore[1:, mask] + v[k[mask]].T
        self.s[mask] = 0.5 / self.distance[mask]
        self.rhs_scale[mask] = 0.5 / self.distance[mask]

        self.iload.append(iload1)
        self.iload.append(iload2)

    def _get_args(self, ff):
        args = super()._get_args(ff)
        # FIXME: needed to have the same results between numpy and cython
        # That means that there are dependencies between the rhs and the lhs
        # during the loop over the boundary elements
        # check why (to test it use air_conditioning example)
        args["fcopy"] = ff.array.copy()
        return args

    def set_rhs(self):
        """
        Compute and set the additional terms to fix the boundary values.
        """
        k = self.istore[:, 0]
        ksym = self.stencil.get_symmetric()[k]
        self.rhs[:] = self.rhs_scale * (
            self.feq[k, np.arange(k.size)] - self.feq[ksym, np.arange(k.size)]
        )

    # pylint: disable=too-many-locals
    def generate(self, sorder):
        """
        Generate the numerical code.

        Parameters
        ----------
        sorder : list
            the order of nv, nx, ny and nz
        """
        from .generator import For
        from .symbolic import nx, ny, nz, indexed, ix

        ns = int(self.stencil.nv_ptr[-1])
        dim = self.stencil.dim

        istore, iload, ncond = self._get_istore_iload_symb(dim)
        rhs, dist = self._get_rhs_dist_symb(ncond)

        idx = Idx(ix, (0, ncond))
        fstore = indexed(
            "f",
            [ns, nx, ny, nz],
            index=[istore[idx, k] for k in range(dim + 1)],
            priority=sorder,
        )
        fload0 = indexed(
            "fcopy",
            [ns, nx, ny, nz],
            index=[iload[0][idx, k] for k in range(dim + 1)],
            priority=sorder,
        )
        fload1 = indexed(
            "fcopy",
            [ns, nx, ny, nz],
            index=[iload[1][idx, k] for k in range(dim + 1)],
            priority=sorder,
        )

        self.generator.add_routine(
            (
                "Bouzidi_bounce_back",
                For(
                    idx,
                    Eq(
                        fstore, dist[idx] * fload0 + (1 - dist[idx]) * fload1 + rhs[idx]
                    ),
                ),
            )
        )

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.Bouzidi_bounce_back


class AntiBounceBack(BounceBack):
    """
    Boundary condition of type anti bounce-back

    Notes
    ------

    .. plot:: codes/anti_bounce_back.py

    """

    def set_rhs(self):
        """
        Compute and set the additional terms to fix the boundary values.
        """
        k = self.istore[:, 0]
        ksym = self.stencil.get_symmetric()[k]
        self.rhs[:] = self.feq[k, np.arange(k.size)] + self.feq[ksym, np.arange(k.size)]

    # pylint: disable=too-many-locals
    def generate(self, sorder):
        """
        Generate the numerical code.

        Parameters
        ----------
        sorder : list
            the order of nv, nx, ny and nz
        """
        from .generator import For
        from .symbolic import nx, ny, nz, indexed, ix

        ns = int(self.stencil.nv_ptr[-1])
        dim = self.stencil.dim

        istore, iload, ncond = self._get_istore_iload_symb(dim)
        rhs, _ = self._get_rhs_dist_symb(ncond)

        idx = Idx(ix, (0, ncond))
        fstore = indexed(
            "f",
            [ns, nx, ny, nz],
            index=[istore[idx, k] for k in range(dim + 1)],
            priority=sorder,
        )
        fload = indexed(
            "f",
            [ns, nx, ny, nz],
            index=[iload[0][idx, k] for k in range(dim + 1)],
            priority=sorder,
        )

        self.generator.add_routine(
            ("anti_bounce_back", For(idx, Eq(fstore, -fload + rhs[idx])))
        )

    @property
    def function(self):
        return self.generator.module.anti_bounce_back


class BFLAntiBounceBack(BFLBounceBack):
    """
    Boundary condition of type Bouzidi-Firdaouss-Lallemand anti bounce-back

    .. versionchanged:: 0.12.0
       Renamed from ``BouzidiAntiBounceBack``, which is deprecated.

    Notes
    ------

    .. plot:: codes/Bouzidi.py

    """

    def set_rhs(self):
        """
        Compute and set the additional terms to fix the boundary values.
        """
        k = self.istore[:, 0]
        ksym = self.stencil.get_symmetric()[k]
        self.rhs[:] = self.rhs_scale * (
            self.feq[k, np.arange(k.size)] + self.feq[ksym, np.arange(k.size)]
        )

    def set_iload(self):
        super().set_iload()
        # The scheme reads f_q at the fluid node and one node further when
        # the wall is closer than half a link, and f_q and f_-q at the fluid
        # node otherwise. Only the sign of the second term differs from the
        # BFL bounce-back.
        self.sign = np.where(self.distance < 0.5, -1.0, 1.0)

    def _host_args(self):
        args = super()._host_args()
        args["sign"] = self.sign
        return args

    # pylint: disable=too-many-locals
    def generate(self, sorder):
        """
        Generate the numerical code.

        Parameters
        ----------
        sorder : list
            the order of nv, nx, ny and nz
        """
        from .generator import For
        from .symbolic import nx, ny, nz, indexed, ix

        ns = int(self.stencil.nv_ptr[-1])
        dim = self.stencil.dim

        istore, iload, ncond = self._get_istore_iload_symb(dim)
        rhs, dist = self._get_rhs_dist_symb(ncond)
        sign = IndexedBase("sign", [ncond])

        idx = Idx(ix, (0, ncond))
        fstore = indexed(
            "f",
            [ns, nx, ny, nz],
            index=[istore[idx, k] for k in range(dim + 1)],
            priority=sorder,
        )
        # read the copy made by BFLBounceBack._get_args: near a corner,
        # a link reads a population that another link of this condition
        # writes, and the result must not depend on the loop order
        fload0 = indexed(
            "fcopy",
            [ns, nx, ny, nz],
            index=[iload[0][idx, k] for k in range(dim + 1)],
            priority=sorder,
        )
        fload1 = indexed(
            "fcopy",
            [ns, nx, ny, nz],
            index=[iload[1][idx, k] for k in range(dim + 1)],
            priority=sorder,
        )

        self.generator.add_routine(
            (
                "Bouzidi_anti_bounce_back",
                For(
                    idx,
                    Eq(
                        fstore,
                        -dist[idx] * fload0
                        + sign[idx] * (1 - dist[idx]) * fload1
                        + rhs[idx],
                    ),
                ),
            )
        )

    @property
    def function(self):
        return self.generator.module.Bouzidi_anti_bounce_back


class LinearBounceBack(BoundaryMethod):
    r"""
    Linear interpolation of the bounce-back for a Dirichlet condition on
    the antisymmetric equilibrium (velocity of a flow).

    For a link cut by the wall, let :math:`x_b` be the fluid node,
    :math:`q` the velocity that leaves :math:`x_b` towards the wall,
    :math:`\bar q` its opposite, and :math:`\delta \in [0, 1]` the distance
    from :math:`x_b` to the wall in link units. The population that enters
    the fluid at the next time step is

    .. math::

        f_{\bar q}(x_b, t+1) = \hat\alpha \hat f_q(x_b, t)
        + \hat\beta \hat f_{\bar q}(x_b, t) + \beta f_q(x_b, t+1)
        - \alpha e^-_q(x_w, t) + \hat K (g^-_q + S^-_q)(x_b, t),

    where :math:`\hat f` are the post-collision populations,
    :math:`e^-_q = (f^{eq}_q - f^{eq}_{\bar q})/2` is the antisymmetric part
    of the equilibrium computed with the moments given by the ``value``
    function at the wall point :math:`x_w = x_b + \delta c_q`, and :math:`g^-_q + S^-_q` is the
    antisymmetric part of the change of the populations during the time step
    (collision and source). The coefficients only depend on
    :math:`\alpha(\delta)`:

    .. math::

        \hat\alpha = -1 + \alpha (1/2 + \delta), \quad
        \beta = 1 - \alpha \delta, \quad
        \hat\beta = 1 - \alpha / 2,

    and :math:`\hat K = 1 - \hat\alpha` when the post-collision correction
    is applied, :math:`\hat K = 0` otherwise. With the correction, or when
    :math:`\hat\alpha = 1`, the scheme is parametrized: with a TRT
    collision, the steady solution depends on the relaxation rates only
    through the product :math:`\Lambda^+\Lambda^-`, as with the bounce-back.

    A subclass defines the scheme with the method ``alpha`` and the class
    attribute ``post_collision_correction``. :py:class:`BounceBack` is
    :math:`\alpha = 2, \delta = 1/2`, and :py:class:`BouzidiBounceBack` is
    :math:`\alpha = 2` for :math:`\delta \le 1/2` and :math:`\alpha = 1/\delta`
    otherwise.

    The post-collision correction reads the populations of the previous
    time step: with the NumPy generator, the simulation then stores them in
    a second array.
    """

    # +1: antisymmetric part (flow), -1: symmetric part (advection-diffusion)
    parity = 1
    post_collision_correction = False

    def alpha(self, delta):
        """
        Return the coefficient alpha of the scheme for the distances delta
        (array of the distances from the fluid nodes to the wall).
        """
        raise NotImplementedError

    @classmethod
    def reads_previous_populations(cls):
        return cls.post_collision_correction

    def __init__(
        self,
        istore,
        ilabel,
        distance,
        normal,
        stencil,
        value_bc,
        time_bc,
        nspace,
        generator,
    ):
        super().__init__(
            istore,
            ilabel,
            distance,
            normal,
            stencil,
            value_bc,
            time_bc,
            nspace,
            generator,
        )
        sigma = self.parity
        delta = self.distance
        alpha = np.broadcast_to(np.asarray(self.alpha(delta), dtype=float), delta.shape)
        self.coef_alpha = alpha.copy()
        self.coef_alpha_hat = -sigma + alpha * (0.5 + delta)
        self.coef_beta = sigma - alpha * delta
        self.coef_beta_hat = 1.0 - 0.5 * sigma * alpha
        if self.post_collision_correction:
            self.coef_k_hat = sigma - self.coef_alpha_hat
        else:
            self.coef_k_hat = np.zeros_like(delta)

    def set_iload(self):
        """
        Compute the indices that are needed: the outgoing velocity at the
        fluid node and at the next node towards the fluid, and the incoming
        velocity at the fluid node.
        """
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k]
        v = self.stencil.get_all_velocities()
        fluid = self.istore[1:] + v[k].T
        # outgoing population at the fluid node
        self.iload.append(np.concatenate([ksym[np.newaxis, :], fluid]).astype(np.int32))
        # outgoing population one node further in the fluid
        self.iload.append(
            np.concatenate([ksym[np.newaxis, :], fluid + v[k].T]).astype(np.int32)
        )
        # incoming population at the fluid node
        self.iload.append(np.concatenate([k[np.newaxis, :], fluid]).astype(np.int32))

    def set_rhs(self):
        """
        Compute and set the wall term -alpha e_q(x_w).
        """
        k = self.istore[:, 0]
        ksym = self.stencil.get_symmetric()[k]
        n = np.arange(k.size)
        self.rhs[:] = (
            -0.5 * self.coef_alpha * (self.feq[ksym, n] - self.parity * self.feq[k, n])
        )

    def _host_args(self):
        args = super()._host_args()
        args["alpha_hat"] = self.coef_alpha_hat
        args["beta"] = self.coef_beta
        args["beta_hat"] = self.coef_beta_hat
        if self.post_collision_correction:
            args["k_hat"] = self.coef_k_hat
        return args

    def _get_args(self, ff):
        args = super()._get_args(ff)
        # the condition reads populations that another link of the same
        # condition may write (corners)
        args["fcopy"] = ff.array.copy()
        return args

    def update(self, ff, fprevious=None, **kwargs):
        """
        Update distribution functions with this boundary condition.

        Parameters
        ----------

        ff : Array
            The distribution functions
        fprevious : Array
            The distribution functions of the previous time step, needed by
            the post-collision correction
        """
        from .symbolic import call_genfunction

        args = self._get_args(ff)
        if self.post_collision_correction:
            if fprevious is None:
                raise ValueError(
                    f"{type(self).__name__} needs the populations of the previous time step"
                )
            args["fprev"] = fprevious.array
        args.update(kwargs)
        call_genfunction(self.function, args)

    def init_previous_populations(self, container):
        """
        Set the previous populations read by the correction so that the
        change during the first time step is zero.

        At the first time step, no collision has happened yet: the
        correction must vanish. It reads f_q(x - c_q) in the previous
        populations and f_q(x) in the current ones, for the incoming and the
        outgoing velocity of each link. Only these values are set; the other
        ones are written by the first time step before being read.
        """
        if not self.post_collision_correction:
            return
        F, Fprev = container.F, container.Fnew
        if F.gpu_support:
            F.array_cpu[...] = F.array.get()
            Fprev.array_cpu[...] = Fprev.array.get()
        # after fix_iload, the indices are stored by row
        for current, previous in [
            (self.iload[0], self.iload[1]),
            (self.iload[2], self.istore),
        ]:
            Fprev.swaparray[tuple(previous.T)] = F.swaparray[tuple(current.T)]
        if F.gpu_support:
            Fprev.array.set(Fprev.array_cpu)

    @property
    def routine_name(self):
        """Name of the generated function."""
        return "linear_bc_" + type(self).__name__.lower()

    # pylint: disable=too-many-locals
    def generate(self, sorder):
        """
        Generate the numerical code.

        Parameters
        ----------
        sorder : list
            the order of nv, nx, ny and nz
        """
        from .generator import For
        from .symbolic import nx, ny, nz, indexed, ix

        ns = int(self.stencil.nv_ptr[-1])
        dim = self.stencil.dim

        istore, iload, ncond = self._get_istore_iload_symb(dim)
        rhs, _ = self._get_rhs_dist_symb(ncond)
        alpha_hat = IndexedBase("alpha_hat", [ncond])
        beta = IndexedBase("beta", [ncond])
        beta_hat = IndexedBase("beta_hat", [ncond])

        idx = Idx(ix, (0, ncond))

        def population(name, index):
            return indexed(
                name,
                [ns, nx, ny, nz],
                index=[index[idx, k] for k in range(dim + 1)],
                priority=sorder,
            )

        fstore = population("f", istore)
        # post-collision populations: outgoing and incoming at the fluid
        # node, outgoing one node further
        f_out = population("fcopy", iload[0])
        f_next = population("fcopy", iload[1])
        f_in = population("fcopy", iload[2])

        expr = (
            alpha_hat[idx] * f_out
            + beta_hat[idx] * f_in
            + beta[idx] * f_next
            + rhs[idx]
        )

        if self.post_collision_correction:
            k_hat = IndexedBase("k_hat", [ncond])
            # change of the populations during the previous time step:
            # post-collision at the fluid node minus the population that
            # the collision read there
            g_out = f_out - population("fprev", iload[1])
            g_in = f_in - population("fprev", istore)
            expr += k_hat[idx] * 0.5 * (g_out - self.parity * g_in)

        self.generator.add_routine((self.routine_name, For(idx, Eq(fstore, expr))))

    @property
    def function(self):
        """Return the generated function"""
        return getattr(self.generator.module, self.routine_name)


class LinearAntiBounceBack(LinearBounceBack):
    r"""
    Linear interpolation of the anti-bounce-back for a Dirichlet condition
    on the symmetric equilibrium (advection-diffusion).

    The scheme is the one of :py:class:`LinearBounceBack` with the symmetric
    parts :math:`e^+_q = (f^{eq}_q + f^{eq}_{\bar q})/2` and
    :math:`g^+_q + S^+_q` instead of the antisymmetric ones, and

    .. math::

        \hat\alpha = 1 + \alpha (1/2 + \delta), \quad
        \beta = -1 - \alpha \delta, \quad
        \hat\beta = 1 + \alpha / 2,

    with :math:`\hat K = -1 - \hat\alpha` when the post-collision correction
    is applied. :py:class:`AntiBounceBack` is
    :math:`\alpha = -2, \delta = 1/2`, and :py:class:`BouzidiAntiBounceBack`
    is :math:`\alpha = -2` for :math:`\delta \le 1/2` and
    :math:`\alpha = -1/\delta` otherwise.
    """

    parity = -1


def _alpha_bfl(delta):
    return np.where(delta <= 0.5, 2.0, 1.0 / np.maximum(delta, 0.5))


class CLIBounceBack(LinearBounceBack):
    r"""
    Central linear interpolation: :math:`\alpha = 4/(1 + 2\delta)`.

    One rule for all :math:`\delta \in [0, 1]`, reduced to the bounce-back for
    :math:`\delta = 1/2`. :math:`\hat\alpha = 1`, so the scheme is
    parametrized without correction.
    """

    def alpha(self, delta):
        return 4.0 / (1.0 + 2.0 * delta)


class YLIBounceBack(LinearBounceBack):
    r"""
    Linear interpolation of Yu, Mei, Luo and Shyy:
    :math:`\alpha = 2/(1 + \delta)`.

    One rule for all :math:`\delta \in [0, 1]`. Without correction, the
    scheme is not parametrized: see :py:class:`ParYLIBounceBack`.
    """

    def alpha(self, delta):
        return 2.0 / (1.0 + delta)


class ParYLIBounceBack(YLIBounceBack):
    """
    :py:class:`YLIBounceBack` with the post-collision correction.
    """

    post_collision_correction = True


class ParBFLBounceBack(LinearBounceBack):
    r"""
    :py:class:`BouzidiBounceBack` with the post-collision correction:
    :math:`\alpha = 2` for :math:`\delta \le 1/2`, :math:`1/\delta` otherwise.
    """

    post_collision_correction = True

    def alpha(self, delta):
        return _alpha_bfl(delta)


class LMKCAntiBounceBack(LinearAntiBounceBack):
    r"""
    Central linear interpolation of the anti-bounce-back:
    :math:`\alpha = -4/(1 + 2\delta)`.

    One rule for all :math:`\delta \in [0, 1]`, reduced to the
    anti-bounce-back for :math:`\delta = 1/2`. :math:`\hat\alpha = -1`, so the
    scheme is parametrized without correction.
    """

    def alpha(self, delta):
        return -4.0 / (1.0 + 2.0 * delta)


class ParLMKAAntiBounceBack(LinearAntiBounceBack):
    r"""
    :py:class:`BouzidiAntiBounceBack` with the post-collision correction:
    :math:`\alpha = -2` for :math:`\delta \le 1/2`,
    :math:`-1/\delta` otherwise.
    """

    post_collision_correction = True

    def alpha(self, delta):
        return -_alpha_bfl(delta)


class Neumann(BoundaryMethod):
    """
    Boundary condition of type Neumann

    """

    name = "neumann"

    def set_rhs(self):
        """
        Compute and set the additional terms to fix the boundary values.
        """
        pass

    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:] + v[k].T
        self.iload.append(np.concatenate([k[np.newaxis, :], indices]))

    # pylint: disable=too-many-locals
    def generate(self, sorder):
        """
        Generate the numerical code.

        Parameters
        ----------
        sorder : list
            the order of nv, nx, ny and nz
        """
        from .generator import For
        from .symbolic import nx, ny, nz, indexed, ix

        ns = int(self.stencil.nv_ptr[-1])
        dim = self.stencil.dim

        istore, iload, ncond = self._get_istore_iload_symb(dim)

        idx = Idx(ix, (0, ncond))
        fstore = indexed(
            "f",
            [ns, nx, ny, nz],
            index=[istore[idx, k] for k in range(dim + 1)],
            priority=sorder,
        )
        fload = indexed(
            "f",
            [ns, nx, ny, nz],
            index=[iload[0][idx, k] for k in range(dim + 1)],
            priority=sorder,
        )

        self.generator.add_routine((self.name, For(idx, Eq(fstore, fload))))

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.neumann


class NeumannX(Neumann):
    """
    Boundary condition of type Neumann along the x direction

    """

    name = "neumannx"

    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:].copy()
        indices[0] += v[k].T[0]
        self.iload.append(np.concatenate([k[np.newaxis, :], indices]))

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.neumannx


class NeumannY(Neumann):
    """
    Boundary condition of type Neumann along the y direction

    """

    name = "neumanny"

    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:].copy()
        indices[1] += v[k].T[1]
        self.iload.append(np.concatenate([k[np.newaxis, :], indices]))

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.neumanny


class NeumannZ(Neumann):
    """
    Boundary condition of type Neumann along the z direction

    """

    name = "neumannz"

    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:].copy()
        indices[2] += v[k].T[2]
        self.iload.append(np.concatenate([k[np.newaxis, :], indices]))

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.neumannz


# former names of the boundary conditions, kept for compatibility
_DEPRECATED_NAMES = {
    "BouzidiBounceBack": "BFLBounceBack",
    "BouzidiAntiBounceBack": "BFLAntiBounceBack",
}


def __getattr__(name):
    if name in _DEPRECATED_NAMES:
        new_name = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"{name} is deprecated since pylbm 0.12.0, use {new_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
