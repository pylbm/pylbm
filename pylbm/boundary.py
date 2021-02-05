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
import types
import numpy as np
from sympy import symbols, IndexedBase, Idx, Eq

from .storage import Array

log = logging.getLogger(__name__) #pylint: disable=invalid-name

#pylint: disable=too-few-public-methods
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
    #pylint: disable=too-many-locals
    def __init__(self, domain, generator, dico):
        self.domain = domain

        # build the list of indices for each unique velocity and for each label
        self.bv_per_label = {}
        for label in self.domain.list_of_labels():
            if label in [-1, -2]: # periodic or interface conditions
                continue
            dummy_bv = []
            for k in range(self.domain.stencil.unvtot):
                dummy_bv.append(BoundaryVelocity(self.domain, label, k))
            self.bv_per_label[label] = dummy_bv

        # build the list of boundary informations for each stencil and each label
        dico_bound = dico.get('boundary_conditions', {})
        stencil = self.domain.stencil

        istore = collections.OrderedDict() # important to set the boundary conditions always in the same way !!!
        ilabel = {}
        distance = {}
        value_bc = {}
        time_bc = {}

        #pylint: disable=too-many-nested-blocks
        for label in self.domain.list_of_labels():
            if label in [-1, -2]: # periodic or interface conditions
                continue

            value_bc[label] = dico_bound[label].get('value', None)
            time_bc[label] = dico_bound[label].get('time_bc', False)
            methods = dico_bound[label]['method']
            # for each method get the list of points, the labels and the distances
            # where the distribution function must be updated on the boundary
            for k, v in methods.items():
                for inumk, numk in enumerate(stencil.num[k]):
                    if self.bv_per_label[label][stencil.unum2index[numk]].indices.size != 0:
                        indices = self.bv_per_label[label][stencil.unum2index[numk]].indices
                        distance_tmp = self.bv_per_label[label][stencil.unum2index[numk]].distance
                        velocity = (inumk + stencil.nv_ptr[k])*np.ones(indices.shape[1], dtype=np.int32)[np.newaxis, :]
                        ilabel_tmp = label*np.ones(indices.shape[1], dtype=np.int32)
                        istore_tmp = np.concatenate([velocity, indices])
                        if istore.get(v, None) is None:
                            istore[v] = istore_tmp.copy()
                            ilabel[v] = ilabel_tmp.copy()
                            distance[v] = distance_tmp.copy()
                        else:
                            istore[v] = np.concatenate([istore[v], istore_tmp], axis=1)
                            ilabel[v] = np.concatenate([ilabel[v], ilabel_tmp])
                            distance[v] = np.concatenate([distance[v], distance_tmp])

        # for each method create the instance associated
        self.methods = []
        for k in list(istore.keys()):
            self.methods.append(k(istore[k], ilabel[k], distance[k], stencil,
                                  value_bc, time_bc, domain.distance.shape, generator))


#pylint: disable=protected-access
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
    def __init__(self, istore, ilabel, distance, stencil, value_bc, time_bc, nspace, generator):
        self.istore = istore
        self.feq = np.zeros((stencil.nv_ptr[-1], istore.shape[1]))
        self.rhs = np.zeros(istore.shape[1])
        self.ilabel = ilabel
        self.distance = distance
        self.stencil = stencil
        self.time_bc = {}
        self.value_bc = {}
        for k in np.unique(self.ilabel):
            self.value_bc[k] = value_bc[k]
            self.time_bc[k] = time_bc[k]
        self.iload = []
        self.nspace = nspace
        self.generator = generator

        # used if time boundary
        self.func = []
        self.args = []
        self.f = []
        self.m = []
        self.indices = []

    def fix_iload(self):
        """
        Transpose iload and istore.

        Must be fix in a future version.
        """
        # Fixme : store in a good way and in the right type istore and iload
        for i in range(len(self.iload)):
            self.iload[i] = np.ascontiguousarray(self.iload[i].T, dtype=np.int32)
        self.istore = np.ascontiguousarray(self.istore.T, dtype=np.int32)

    #pylint: disable=too-many-locals
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
        nspace = [1]*(len(sorder)-1)
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
                    x += s*v[k, i]*simulation.domain.dx
                    x = x.ravel()
                    for j in range(1, simulation.domain.dim): #pylint: disable=unused-variable
                        x = x[:, np.newaxis]
                    coords += (x,)

                m = Array(nv, nspace, 0, sorder, gpu_support=gpu_support)
                m.set_conserved_moments(simulation.scheme.consm)

                f = Array(nv, nspace, 0, sorder, gpu_support=gpu_support)
                f.set_conserved_moments(simulation.scheme.consm)

                args = coords
                if isinstance(value, types.FunctionType):
                    func = value
                elif isinstance(value, tuple):
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

            self.feq[:, self.indices[i]] = self.f[i].swaparray.reshape((nv, self.indices[i].size))

    def _get_istore_iload_symb(self, dim):
        ncond = symbols('ncond', integer=True)

        istore = symbols('istore', integer=True)
        istore = IndexedBase(istore, [ncond, dim+1])

        iload = []
        for i in range(len(self.iload)):
            iloads = symbols('iload%d'%i, integer=True)
            iload.append(IndexedBase(iloads, [ncond, dim+1]))
        return istore, iload, ncond

    @staticmethod
    def _get_rhs_dist_symb(ncond):
        rhs = IndexedBase('rhs', [ncond])
        dist = IndexedBase('dist', [ncond])
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
        call_genfunction(self.function, args) #pylint: disable=no-member

    #pylint: disable=possibly-unused-variable
    def _get_args(self, ff):
        dim = len(ff.nspace)
        nx = ff.nspace[0]
        if dim > 1:
            ny = ff.nspace[1]
        if dim > 2:
            nz = ff.nspace[2]

        f = ff.array

        for i in range(len(self.iload)):
            exec('iload{i} = self.iload[{i}]'.format(i=i)) #pylint: disable=exec-used

        istore = self.istore
        rhs = self.rhs
        if hasattr(self, 's'):
            dist = self.s
        ncond = istore.shape[0]
        return locals()

    def move2gpu(self):
        """
        Move arrays needed to compute the boundary on the GPU memory.
        """
        if self.generator.backend.upper() == "LOOPY":
            try:
                import pyopencl as cl
                import pyopencl.array #pylint: disable=unused-variable
                from .context import queue
            except ImportError:
                raise ImportError("Please install loo.py")

            self.rhs = cl.array.to_device(queue, self.rhs)
            if hasattr(self, 's'):
                self.s = cl.array.to_device(queue, self.s) #pylint: disable=attribute-defined-outside-init
            self.istore = cl.array.to_device(queue, self.istore)
            for i in range(len(self.iload)):
                self.iload[i] = cl.array.to_device(queue, self.iload[i])

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

    #pylint: disable=too-many-locals
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
        fstore = indexed('f', [ns, nx, ny, nz], index=[istore[idx, k] for k in range(dim+1)], priority=sorder)
        fload = indexed('f', [ns, nx, ny, nz], index=[iload[0][idx, k] for k in range(dim+1)], priority=sorder)

        self.generator.add_routine(('bounce_back', For(idx, Eq(fstore, fload + rhs[idx]))))

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.bounce_back

class BouzidiBounceBack(BoundaryMethod):
    """
    Boundary condition of type Bouzidi bounce-back [BFL01]

    Notes
    ------

    .. plot:: codes/Bouzidi.py

    """
    def __init__(self, istore, ilabel, distance, stencil, value_bc, time_bc, nspace, generator):
        super(BouzidiBounceBack, self).__init__(istore, ilabel, distance, stencil, value_bc, time_bc, nspace, generator)
        self.s = np.empty(self.istore.shape[1])

    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k]
        v = self.stencil.get_all_velocities()

        iload1 = np.zeros(self.istore.shape, dtype=np.int)
        iload2 = np.zeros(self.istore.shape, dtype=np.int)

        mask = self.distance < .5
        iload1[0, mask] = ksym[mask]
        iload2[0, mask] = ksym[mask]
        iload1[1:, mask] = self.istore[1:, mask] + v[k[mask]].T
        iload2[1:, mask] = self.istore[1:, mask] + 2*v[k[mask]].T
        self.s[mask] = 2.*self.distance[mask]

        mask = np.logical_not(mask)
        iload1[0, mask] = ksym[mask]
        iload2[0, mask] = k[mask]
        iload1[1:, mask] = self.istore[1:, mask] + v[k[mask]].T
        iload2[1:, mask] = self.istore[1:, mask] + v[k[mask]].T
        self.s[mask] = .5/self.distance[mask]

        self.iload.append(iload1)
        self.iload.append(iload2)

    def _get_args(self, ff):
        dim = len(ff.nspace)
        nx = ff.nspace[0]
        if dim > 1:
            ny = ff.nspace[1]
        if dim > 2:
            nz = ff.nspace[2]

        f = ff.array
        # FIXME: needed to have the same results between numpy and cython
        # That means that there are dependencies between the rhs and the lhs
        # during the loop over the boundary elements
        # check why (to test it use air_conditioning example)
        fcopy = ff.array.copy()

        for i in range(len(self.iload)):
            exec('iload{i} = self.iload[{i}]'.format(i=i)) #pylint: disable=exec-used

        istore = self.istore
        rhs = self.rhs
        if hasattr(self, 's'):
            dist = self.s
        ncond = istore.shape[0]
        return locals()

    def set_rhs(self):
        """
        Compute and set the additional terms to fix the boundary values.
        """
        k = self.istore[:, 0]
        ksym = self.stencil.get_symmetric()[k]
        self.rhs[:] = self.feq[k, np.arange(k.size)] - self.feq[ksym, np.arange(k.size)]

    #pylint: disable=too-many-locals
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
        fstore = indexed('f', [ns, nx, ny, nz], index=[istore[idx, k] for k in range(dim+1)], priority=sorder)
        fload0 = indexed('fcopy', [ns, nx, ny, nz], index=[iload[0][idx, k] for k in range(dim+1)], priority=sorder)
        fload1 = indexed('fcopy', [ns, nx, ny, nz], index=[iload[1][idx, k] for k in range(dim+1)], priority=sorder)

        self.generator.add_routine(('Bouzidi_bounce_back', For(idx, Eq(fstore, dist[idx]*fload0 + (1-dist[idx])*fload1 + rhs[idx]))))

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

    #pylint: disable=too-many-locals
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
        fstore = indexed('f', [ns, nx, ny, nz], index=[istore[idx, k] for k in range(dim+1)], priority=sorder)
        fload = indexed('f', [ns, nx, ny, nz], index=[iload[0][idx, k] for k in range(dim+1)], priority=sorder)

        self.generator.add_routine(('anti_bounce_back', For(idx, Eq(fstore, -fload + rhs[idx]))))

    @property
    def function(self):
        return self.generator.module.anti_bounce_back

class BouzidiAntiBounceBack(BouzidiBounceBack):
    """
    Boundary condition of type Bouzidi anti bounce-back

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
        self.rhs[:] = self.feq[k, np.arange(k.size)] + self.feq[ksym, np.arange(k.size)]

    #pylint: disable=too-many-locals
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
        fstore = indexed('f', [ns, nx, ny, nz], index=[istore[idx, k] for k in range(dim+1)], priority=sorder)
        fload0 = indexed('f', [ns, nx, ny, nz], index=[iload[0][idx, k] for k in range(dim+1)], priority=sorder)
        fload1 = indexed('f', [ns, nx, ny, nz], index=[iload[1][idx, k] for k in range(dim+1)], priority=sorder)

        self.generator.add_routine(('Bouzidi_anti_bounce_back', For(idx, Eq(fstore, -dist[idx]*fload0 + (1-dist[idx])*fload1 + rhs[idx]))))

    @property
    def function(self):
        return self.generator.module.Bouzidi_anti_bounce_back

class Neumann(BoundaryMethod):
    """
    Boundary condition of type Neumann

    """
    name = 'neumann'
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

    #pylint: disable=too-many-locals
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
        fstore = indexed('f', [ns, nx, ny, nz], index=[istore[idx, k] for k in range(dim+1)], priority=sorder)
        fload = indexed('f', [ns, nx, ny, nz], index=[iload[0][idx, k] for k in range(dim+1)], priority=sorder)

        self.generator.add_routine((self.name, For(idx, Eq(fstore, fload))))

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.neumann

class NeumannX(Neumann):
    """
    Boundary condition of type Neumann along the x direction

    """
    name = 'neumannx'
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
    name = 'neumanny'
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
    name = 'neumannz'
    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:].copy()
        indices[1] += v[k].T[2]
        self.iload.append(np.concatenate([k[np.newaxis, :], indices]))

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.neumannz
