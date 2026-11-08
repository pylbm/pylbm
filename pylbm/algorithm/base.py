# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Base Algorithm for lattice Boltzmann methods
============================================

This module describes various methods involved during
the computation of a solution using lattice Boltzmann
methods.

The default kernels defined here are:

    - transport kernel
    - f2m kernel
    - m2f_kernel
    - equilibrium kernel
    - relaxation kernel
    - source terms kernel
    - one_time_step kernel which makes the stream + source terms + relaxation

You can modify each functions to define your own behavior. This is what is done
when we want to specialize our one_time_step kernel for Pull algorithm,
Push algorithm, ... All these kernels are defined using SymPy and thus must be
expressed as symbolic expressions. Then, we will generate the numerical code.

These kernels are defined in a class (the base class is BaseAlgorithm) where we
have all the needed information of our scheme.

Let's take an example: the f2m kernel allows to compute the moments from the
distribution functions thanks to the matrix M define in our scheme.

The formula is straightforward

    m = M f

where m is the moments, f the distributed functions and M the matrix build with
the polynoms defining the moments of our scheme.

The SymPy expression can be written as

    Eq(m, M*f)

where m and f are symbol matrices and M a SymPy matrix.

When we define a kernel, we make that in two steps. First we define locally
the expression (what we have to do for each point). Then we define the kernel
using this local kernel for the whole domain (we write the loop).

Therefore, for our example we have

    def f2m_local(self, f, m):
        return Eq(m, self.M*f)

    def f2m(self):
        space_index = self._get_space_idx_full()
        f = self._get_indexed_1('f', space_index)
        m = self._get_indexed_1('m', space_index)
        return {'code': For(space_index, self.f2m_local(f, m))}

"""

import numpy as np
import sympy as sp
from sympy import Eq

from ..generator import For, If
from ..symbolic import (
    ix,
    iy,
    iz,
    ix_,
    iy_,
    iz_,
    nx,
    ny,
    nz,
    nv,
    indexed,
    space_idx,
    alltogether,
    recursive_sub,
    SymbolicVector,
)
from ..symbolic import rel_ux, rel_uy, rel_uz
from .transform import parse_expr
from .ode import euler
from ..monitoring import monitor


class BaseAlgorithm:
    def __init__(self, scheme, sorder, generator, settings=None):
        xx, yy, zz = sp.symbols("xx, yy, zz")
        self.symb_coord_local = [xx, yy, zz]
        self.symb_coord = scheme.symb_coord
        self.dim = scheme.dim
        self.ns = scheme.stencil.nv_ptr[-1]
        # number of moments stored for each point: ns, plus the conserved
        # moments defined with conserved_polynomials
        self.nm = scheme.nmoments
        self.M = scheme.M.subs(scheme.param.items())
        self.invM = scheme.invM.subs(scheme.param.items())
        self.all_velocities = scheme.stencil.get_all_velocities()
        self.mv = sp.MatrixSymbol("m", self.nm, 1)

        if scheme.rel_vel is not None:
            self.rel_vel_symb = [rel_ux, rel_uy, rel_uz][: self.dim]
            self.rel_vel = sp.Matrix(scheme.rel_vel)

            self.Tu = scheme.Tu.subs(scheme.param.items())
            self.Tmu = scheme.Tmu.subs(scheme.param.items())

            self.Mu = self.Tu * self.M
            self.invMu = self.invM * self.Tmu
            alltogether(self.Mu, nsimplify=True)
            alltogether(self.invMu, nsimplify=True)
        else:
            self.rel_vel_symb = None

        self.consm = {}
        for k, v in scheme.consm.items():
            self.consm[str(k)] = v

        self.sorder = sorder
        self.generator = generator

        subs_coords = list(zip(self.symb_coord, self.symb_coord_local))
        subs_moments = list(
            zip(
                scheme.consm.keys(), [self.mv[int(i), 0] for i in scheme.consm.values()]
            )
        )
        to_subs = subs_coords + list(scheme.param.items())
        to_subs_full = to_subs + subs_moments

        self.eq = recursive_sub(scheme.EQ, to_subs_full)
        self.s = recursive_sub(scheme.s, to_subs_full)
        alltogether(self.eq, nsimplify=True)
        alltogether(self.s)

        if self.rel_vel_symb:
            self.rel_vel = recursive_sub(self.rel_vel, to_subs_full)
            alltogether(self.rel_vel)

        self.source_eq = []
        for source in scheme._source_terms:
            if source:
                for k, v in source.items():
                    lhs = recursive_sub(k, to_subs_full)
                    if isinstance(v, (float, int)):
                        rhs = v
                    else:
                        rhs = recursive_sub(v, to_subs)
                    self.source_eq.append((lhs, rhs))

        self.vmax = [0] * 3
        self.vmax[: scheme.dim] = scheme.stencil.vmax
        self.local_vars = self.symb_coord_local[: self.dim]
        self.settings = settings if settings else {}
        # generated routines that need the local space variables
        relaxation = [self.eq, self.s]
        if self.rel_vel_symb:
            relaxation.append(self.rel_vel)
        sources = [rhs for _, rhs in self.source_eq]
        self.routines_with_coords = {
            name
            for name, exprs in [
                ("equilibrium", [self.eq]),
                ("relaxation", relaxation),
                ("source_term", sources),
                ("one_time_step", relaxation + sources),
            ]
            if self._depends_on_coords(exprs)
        }
        self.uses_coords = bool(self.routines_with_coords)

    def _depends_on_coords(self, exprs):
        """
        Return True if one of the expressions depends on the space variables.
        """
        local_coords = set(self.symb_coord_local[: self.dim])
        for expr in exprs:
            if isinstance(expr, (list, tuple, sp.MatrixBase, SymbolicVector)):
                if self._depends_on_coords(list(expr)):
                    return True
            elif isinstance(expr, sp.Basic) and expr.free_symbols & local_coords:
                return True
        return False

    def _with_coords(self, routine, space_index, code):
        """
        Prepend the computation of the local space variables to the code
        of a loop body when the routine depends on them.
        """
        if routine not in self.routines_with_coords:
            return code
        if not isinstance(code, list):
            code = [code]
        return [*self.coords(space_index), *code]

    def _get_space_idx_full(self):
        """
        Return a list of SymPy Idx ordered with sorder
        and with the dimensions.

            ix -> [0, nx[
            iy -> [0, ny[
            iz -> [0, nz[

        The length of the list is the dimension of the problem.
        """
        return space_idx([(0, nx), (0, ny), (0, nz)], priority=self.sorder[1:])

    def _get_space_idx_inner(self):
        """
        Return a list of SymPy Idx ordered with sorder
        and with the dimensions.

            ix -> [vmax_x, nx-vmax_x[
            iy -> [vmax_y, ny-vmax_y[
            iz -> [vmax_z, nz-vmax_z[

        where vmax_i is the maximum of the velocities modulus in direction i.
        The length of the list is the dimension of the problem.
        """
        return space_idx(
            [
                (self.vmax[0], nx - self.vmax[0]),
                (self.vmax[1], ny - self.vmax[1]),
                (self.vmax[2], nz - self.vmax[2]),
            ],
            priority=self.sorder[1:],
        )

    @staticmethod
    def _logical_order(space_index):
        """
        Return the space indices in the order x, y, z.

        The list given by _get_space_idx_full or _get_space_idx_inner is
        ordered by sorder to set the nesting of the loops. The function
        indexed applies sorder itself, so it must receive the indices in
        their logical order.
        """
        rank = {ix_: 0, iy_: 1, iz_: 2}
        return sorted(space_index, key=lambda idx: rank[idx.label])

    def _get_indexed_on_range(self, name, space_index):
        """
        Return a SymPy matrix of indexed objects
        (one component for each velocity index).

        Parameters
        ----------

        name : string
            name of the SymPy symbol for the indexed object

        space_index : list
            list of SymPy Idx corresponding to space variables

        Return
        ------

        SymPy Matrix
            indexed objects for each velocity

        """
        size = self._size(name)
        return indexed(
            name,
            [size, nx, ny, nz],
            [nv] + self._logical_order(space_index),
            velocities_index=range(size),
            priority=self.sorder,
        )

    def _size(self, name):
        """
        Return the number of components of the array called name:
        the number of moments for m, the number of velocities otherwise.
        """
        return self.nm if name == "m" else self.ns

    def _get_indexed_on_velocities(self, name, space_index, velocities):
        """
        Return a SymPy matrix of indexed objects (one component for each velocity).

        Parameters
        ----------

        name : string
            name of the SymPy symbol for the indexed object

        space_index : list
            list of SymPy Idx corresponding to space variables

        velocities : list
            list of velocity components

        Return
        ------

        SymPy Matrix
            indexed objects for each velocity

        """
        return indexed(
            name,
            [self._size(name), nx, ny, nz],
            [nv] + self._logical_order(space_index),
            velocities=velocities,
            priority=self.sorder,
        )

    def relative_velocity(self, m):
        rel_vel = sp.Matrix(self.rel_vel).subs(list(zip(self.mv, m)))
        return [Eq(self.rel_vel_symb[i], rel_vel[i]) for i in range(self.dim)]

    def restore_conserved_moments(self, m, f):
        nconsm = len(self.consm)

        if isinstance(m[nconsm:], list):
            m_consm = sp.Matrix(m[:nconsm])
        else:
            m_consm = m[:nconsm]

        return Eq(m_consm, sp.Matrix((self.Mu * f)[:nconsm]))

    def coords(self, space_index):
        """
        Return the symbolic expressions which set the local space variables
        (xx, yy, zz) from the 1D arrays of coordinates (with halo) at the
        current loop indices.

        Parameters
        ----------

        space_index : list
            the SymPy Idx of the space loops
        """
        idx_by_label = {idx.label: idx for idx in space_index}
        loop_idx = [idx_by_label[label] for label in [ix_, iy_, iz_][: self.dim]]
        sizes = [nx, ny, nz][: self.dim]

        code = []
        for d in range(self.dim):
            if self._dense_coords:
                # vectorized backend: the loops become slices on all the space
                # directions, so the coordinates are arrays of the domain shape
                x = indexed(
                    self.symb_coord[d].name,
                    sizes,
                    index=loop_idx,
                    priority=self.sorder[1:],
                )
            else:
                x = sp.IndexedBase(self.symb_coord[d].name, [sizes[d]])[loop_idx[d]]
            code.append(Eq(self.symb_coord_local[d], x))
        return code

    @property
    def _dense_coords(self):
        return self.generator.backend.upper() == "NUMPY"

    def _coords_args(self, coords_halo):
        """
        Return the coordinate arrays given to the generated code, keyed by
        the names of the space variables.
        """
        if not self._dense_coords:
            return {self.symb_coord[d].name: coords_halo[d] for d in range(self.dim)}

        key = tuple(c.size for c in coords_halo)
        if getattr(self, "_dense_coords_cache", (None,))[0] != key:
            grids = np.meshgrid(*coords_halo[: self.dim], indexing="ij")
            axes = np.argsort(self.sorder[1:])
            self._dense_coords_cache = (
                key,
                {
                    self.symb_coord[d].name: np.ascontiguousarray(np.transpose(g, axes))
                    for d, g in enumerate(grids)
                },
            )
        return self._dense_coords_cache[1]

    def transport_local(self, f, fnew):
        """
        Return the symbolic expression of the lbm transport.

        Parameters
        ----------

        f : SymPy Matrix
            indexed objects of rhs for the distributed functions

        fnew : SymPy Matrix
            indexed objects of lhs for the distributed functions

        """
        return Eq(fnew, f)

    def transport(self):
        """
        Return the code expression of the lbm transport on the whole inner domain.
        """
        space_index = self._get_space_idx_inner()
        f = self._get_indexed_on_velocities("f", space_index, -self.all_velocities)
        fnew = self._get_indexed_on_range("fnew", space_index)
        return {"code": For(space_index, self.transport_local(f, fnew))}

    def f2m_local(self, f, m, with_rel_velocity=False):
        """
        Return symbolic expression which computes the moments from the
        distributed functions.

        Parameters
        ----------

        f : SymPy Matrix
            indexed objects for the distributed functions

        m : SymPy Matrix
            indexed objects for the moments

        with_rel_velocity : boolean
            check if the scheme uses relative velocity.
            If yes, the conserved moments must be computed first.
            (default is False)

        """
        if with_rel_velocity:
            nconsm = len(self.consm)

            if isinstance(m[:nconsm], list):
                m_consm = sp.Matrix(m[:nconsm])
                m_notconsm = sp.Matrix(m[nconsm:])
            else:
                m_consm = m[:nconsm]
                m_notconsm = m[nconsm:]

            return [
                Eq(m_consm, sp.Matrix((self.M * f)[:nconsm])),
                *self.relative_velocity(m),
                Eq(m_notconsm, sp.Matrix((self.Mu * f)[nconsm:])),
            ]
        else:
            return Eq(m, self.M * f)

    def f2m(self):
        """
        Return the code expression which computes the moments from the
        distributed functions on the whole domain.
        """
        space_index = self._get_space_idx_full()
        f = self._get_indexed_on_range("f", space_index)
        m = self._get_indexed_on_range("m", space_index)
        return {"code": For(space_index, self.f2m_local(f, m))}

    def m2f_local(self, m, f, with_rel_velocity=False):
        """
        Return symbolic expression which computes the distributed functions
        from the moments.

        Parameters
        ----------

        m : SymPy Matrix
            indexed objects for the moments

        f : SymPy Matrix
            indexed objects for the distributed functions

        with_rel_velocity : boolean
            check if the scheme uses relative velocity.
            If yes, the conserved moments must be computed first.
            (default is False)

        """
        if with_rel_velocity:
            return Eq(f, self.invMu * m)
        else:
            return Eq(f, self.invM * m)

    def m2f(self):
        """
        Return the code expression which computes the distributed functions
        from the moments on the whole domain.
        """
        space_index = self._get_space_idx_full()
        f = self._get_indexed_on_range("f", space_index)
        m = self._get_indexed_on_range("m", space_index)
        return {"code": For(space_index, self.m2f_local(m, f))}

    def equilibrium_local(self, m):
        """
        Return symbolic expression which computes the equilibrium.

        Parameters
        ----------

        m : SymPy Matrix
            indexed objects for the moments

        """
        eq = self.eq.subs(list(zip(self.mv, m)))
        return Eq(m, eq)

    def equilibrium(self):
        """
        Return the code expression which computes the equilibrium
        on the whole domain.
        """
        space_index = self._get_space_idx_full()
        m = self._get_indexed_on_range("m", space_index)
        code = self._with_coords("equilibrium", space_index, self.equilibrium_local(m))
        output = {"code": For(space_index, code)}
        if "equilibrium" in self.routines_with_coords:
            output["local_vars"] = self.local_vars
        return output

    def relaxation_local(self, m, with_rel_velocity=False):
        """
        Return symbolic expression which computes the relaxation operator.

        Parameters
        ----------

        m : SymPy Matrix
            indexed objects for the moments

        with_rel_velocity : boolean
            check if the scheme uses relative velocity.
            (default is False)

        """
        if with_rel_velocity:
            eq = (self.Tu * self.eq).subs(list(zip(self.mv, m)))
        else:
            eq = self.eq.subs(list(zip(self.mv, m)))
        # Ensure eq is a SymbolicVector for element-wise multiplication with self.s
        if not isinstance(eq, SymbolicVector):
            eq = SymbolicVector(eq)
        relax = (1 - self.s) * m + self.s * eq
        alltogether(relax)
        return Eq(m, relax)

    def relaxation(self):
        """
        Return the code expression which computes the relaxation
        on the whole domain.
        """
        space_index = self._get_space_idx_full()
        m = self._get_indexed_on_range("m", space_index)
        code = self._with_coords("relaxation", space_index, self.relaxation_local(m))
        output = {"code": For(space_index, code)}
        if "relaxation" in self.routines_with_coords:
            output["local_vars"] = self.local_vars
        return output

    def source_term_local(self, m):
        """
        Return symbolic expression which computes the source term
        using explicit Euler (should be more flexible in a near future).

        Parameters
        ----------

        m : SymPy Matrix
            indexed objects for the moments

        """
        rhs_eq = [mm for mm in m]
        m_local = self.settings.get("m_local", False)
        split = self.settings.get("split", False)

        indices = []
        if m_local and not split:
            local_dict = {
                "m": m,
                "consm": self.consm,
                "sorder": None,
                "default_index": [0],
            }
        else:
            indices_str = ["ix_", "iy_", "iz_"]
            lbm_ind = [ix, iy, iz]
            ind_to_subs = []
            for i, sorder in enumerate(self.sorder[1:]):
                indices.append(m[0].indices[sorder])
                ind_to_subs.extend(
                    [
                        (indices_str[i], indices[i]),
                        (lbm_ind[i], indices[i]),
                    ]
                )
            local_dict = {
                "m": m[0].base,
                "consm": self.consm,
                "sorder": self.sorder,
                "default_index": indices,
            }
            for i, ind in enumerate(indices):
                local_dict[indices_str[i]] = ind

        for lhs, rhs in self.source_eq:
            lhs_m = lhs.subs(list(zip(self.mv, m)))
            if isinstance(rhs, (float, int)):
                rhs_m = rhs
            else:
                rhs_m = parse_expr(rhs, local_dict)
            dummy = euler(lhs_m, rhs_m)
            if indices:
                rhs_eq[lhs.i] = dummy.rhs.subs(ind_to_subs)
            else:
                rhs_eq[lhs.i] = dummy.rhs
        return [Eq(m, sp.Matrix(rhs_eq))]

    def source_term(self):
        """
        Return the code expression which computes the source terms
        on the whole inner domain.
        """
        space_index = self._get_space_idx_inner()
        m = self._get_indexed_on_range("m", space_index)
        code = self._with_coords("source_term", space_index, self.source_term_local(m))
        output = {"code": For(space_index, code)}
        if "source_term" in self.routines_with_coords:
            output["local_vars"] = self.local_vars
        return output

    def one_time_step_local(self, f, fnew, m):
        """
        Return symbolic expression which makes one time step of
        LBM algorithm:

            - transport
            - compute the moments from the distributed functions
            - source terms with dt/2 (with the moments)
            - relaxation (with the moments)
            - source terms with dt/2 (with the moments)
            - compute the new distributed functions from the moments

        Parameters
        ----------

        f : SymPy Matrix
            indexed objects for the old distributed functions

        fnew : SymPy Matrix
            indexed objects for the new distributed functions

        m : SymPy Matrix
            indexed objects for the moments

        """
        with_rel_velocity = True if self.rel_vel_symb else False

        code = [self.transport_local(f, fnew)]
        f2m = self.f2m_local(fnew, m, with_rel_velocity)
        if isinstance(f2m, list):
            code.extend(f2m)
        else:
            code.append(f2m)

        if self.source_eq:
            code.extend(self.source_term_local(m))

        code.append(self.relaxation_local(m, with_rel_velocity))

        if self.source_eq:
            code.extend(self.source_term_local(m))

        code.append(self.m2f_local(m, fnew, with_rel_velocity))
        return code

    def one_time_step(self):
        """
        Return the code expression which  makes one time step of
        LBM algorithm on the whole inner domain.
        """
        m_local = self.settings.get("m_local", False)
        check_isfluid = self.settings.get("check_isfluid", False)
        split = self.settings.get("split", False)

        space_index = self._get_space_idx_inner()
        if m_local:
            if split:
                m = self._get_indexed_on_range("m", space_index)
                local_vars = [m[0]]
            else:
                m = sp.MatrixSymbol("m", self.nm, 1)
                local_vars = [m]
        else:
            m = self._get_indexed_on_range("m", space_index)
            local_vars = []

        if self.rel_vel_symb:
            local_vars.extend(self.rel_vel_symb)

        f = self._get_indexed_on_velocities("f", space_index, -self.all_velocities)
        fnew = self._get_indexed_on_range("fnew", space_index)

        internal = self.one_time_step_local(f, fnew, m)

        if check_isfluid:
            valin = sp.Symbol("valin", real=True)
            in_or_out = indexed(
                "in_or_out",
                [nx, ny, nz],
                self._logical_order(space_index),
                priority=self.sorder[1:],
            )
            loop = lambda x: For(space_index, If((Eq(in_or_out, valin), x)))
        else:
            loop = lambda x: For(space_index, x)

        if split:
            code = [
                loop(self._with_coords("one_time_step", space_index, [i]))
                for i in internal
            ]
        else:
            code = loop(self._with_coords("one_time_step", space_index, [*internal]))

        return {
            "code": code,
            "local_vars": local_vars + self.local_vars,
            "settings": {"prefetch": [f[0]]},
        }

    @monitor
    def generate(self):
        """
        Define the routines which must be generate by code generator of SymPy
        for a given generator.
        """
        to_generate = [
            self.transport,
            self.f2m,
            self.m2f,
            self.relaxation,
            self.equilibrium,
            self.one_time_step,
        ]

        if self.source_eq:
            to_generate.append(self.source_term)

        for gen in to_generate:
            name = gen.__name__
            output = gen()
            code = output["code"]
            local_vars = output.get("local_vars", [])
            settings = output.get("settings", {})
            self.generator.add_routine(
                (name, code), local_vars=local_vars, settings=settings
            )

    def _get_args(self, simulation, m_user=None, f_user=None, **kwargs):
        """
        Define all the arguments needed to call the generated code.
        """
        mm = m_user if m_user else simulation.container.m

        args = {
            "m": mm.array,
            "f": f_user.array if f_user else simulation.container.F.array,
            "fnew": simulation.container.Fnew.array,
            "t": simulation.t,
            "dt": simulation.dt,
            "in_or_out": simulation.domain.in_or_out,
            "valin": simulation.domain.valin,
        }
        for name, size in zip(["nx", "ny", "nz"], mm.nspace):
            args[name] = size

        # The generated loops run over the arrays with their halo, so the
        # coordinates must include the halo too. They only make sense for
        # arrays defined on the whole domain.
        coords_halo = simulation.domain.coords_halo
        if list(mm.nspace) == [c.size for c in coords_halo]:
            if self.uses_coords:
                args.update(self._coords_args(coords_halo))
            else:
                for d in range(len(mm.nspace)):
                    args[self.symb_coord[d].name] = coords_halo[d]

        extra = {str(k): v for k, v in simulation.extra_parameters.items()}
        if extra.get("lambda", None):
            extra["lambda_"] = extra["lambda"]
        args.update(extra)
        return args

    def call_function(
        self, function_name, simulation, m_user=None, f_user=None, **kwargs
    ):
        """
        Call the generated function.
        """
        from ..symbolic import call_genfunction

        func = getattr(self.generator.module, function_name)

        if (
            m_user
            and function_name in self.routines_with_coords
            and list(m_user.nspace) != list(simulation.domain.shape_halo)
        ):
            raise NotImplementedError(
                f"{function_name} depends on the space variables in this scheme: "
                "it can only be called on arrays defined on the whole domain."
            )

        args = self._get_args(simulation, m_user, f_user)
        args.update(kwargs)
        call_genfunction(func, args)
