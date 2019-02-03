# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import sympy as sp
from sympy import Eq

from ..generator import For, If
from ..symbolic import ix, iy, iz, nx, ny, nz, nv, indexed, space_loop, alltogether
from ..symbolic import rel_ux, rel_uy, rel_uz
from .transform import parse_expr
from .ode import euler
from ..monitoring import monitor


class BaseAlgorithm:
    def __init__(self, scheme, sorder, generator, settings=None):
        xx, yy, zz = sp.symbols('xx, yy, zz')
        self.symb_coord_local = [xx, yy, zz]
        self.symb_coord = scheme.symb_coord
        self.dim = scheme.dim
        self.ns = scheme.stencil.nv_ptr[-1]
        self.M = scheme.M.subs(scheme.param.items())
        self.invM = scheme.invM.subs(scheme.param.items())
        self.all_velocities = scheme.stencil.get_all_velocities()
        self.mv = sp.MatrixSymbol('m', self.ns, 1)

        if scheme.rel_vel is not None:
            self.rel_vel_symb = [rel_ux, rel_uy, rel_uz][:self.dim]
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
        subs_moments = list(zip(scheme.consm.keys(), [self.mv[int(i), 0] for i in scheme.consm.values()]))
        to_subs = subs_coords + list(scheme.param.items())
        to_subs_full = to_subs + subs_moments

        self.eq = scheme.EQ.subs(to_subs_full)
        self.s = scheme.s.subs(to_subs_full)
        alltogether(self.eq, nsimplify=True)
        alltogether(self.s)

        if self.rel_vel_symb:
            self.rel_vel = self.rel_vel.subs(to_subs_full)
            alltogether(self.rel_vel)

        self.source_eq = []
        for source in scheme._source_terms:
            if source:
                for k, v in source.items():
                    lhs = k.subs(to_subs_full)
                    if isinstance(v, (float, int)):
                        rhs = v
                    else:
                        rhs = v.subs(to_subs)
                    self.source_eq.append((lhs, rhs))

        self.vmax = [0]*3
        self.vmax[:scheme.dim] = scheme.stencil.vmax
        self.local_vars = self.symb_coord_local[:self.dim]
        self.settings = settings if settings else {}

    def _get_loop_0(self):
        return space_loop([(0, nx), (0, ny), (0, nz)], permutation=self.sorder)

    def _get_loop(self):
        return space_loop([(self.vmax[0], nx-self.vmax[0]),
                           (self.vmax[1], ny-self.vmax[1]),
                           (self.vmax[2], nz-self.vmax[2])], permutation=self.sorder)

    def _get_indexed_1(self, name, iloop):
        return indexed(name, [self.ns, nx, ny, nz],
                       index=[nv] + iloop,
                       ranges=range(self.ns), permutation=self.sorder)

    def _get_indexed_2(self, name, iloop, indices):
        return indexed(name, [self.ns, nx, ny, nz],
                       index=[nv] + iloop,
                       list_ind=indices, permutation=self.sorder)

    def coords(self):
        coord = []
        for x in self.symb_coord[:self.dim]:
            coord.append(indexed(x.name, [self.ns, nx, ny, nz], permutation=self.sorder, remove_ind=[0]))
        return [Eq(xx, x) for xx, x in zip(self.symb_coord_local[:self.dim], coord)]

    def transport_local(self, f, fnew):
        return Eq(fnew, f)

    def f2m_local(self, f, m, with_rel_velocity=False):
        if with_rel_velocity:
            nconsm = len(self.consm)
            return [Eq(m[:nconsm], sp.Matrix((self.M*f)[:nconsm])),
                    *self.relative_velocity(m),
                    Eq(m[nconsm:], sp.Matrix((self.Mu*f)[nconsm:]))]
        else:
            return Eq(m, self.M*f)

    def m2f_local(self, m, f, with_rel_velocity=False):
        if with_rel_velocity:
            return Eq(f, self.invMu*m)
        else:
            return Eq(f, self.invM*m)

    def equilibrium_local(self, m):
        eq = self.eq.subs(list(zip(self.mv, m)))
        return Eq(m, eq)

    def relaxation_local(self, m, with_rel_velocity=False):
        if with_rel_velocity:
            eq = (self.Tu*self.eq).subs(list(zip(self.mv, m)))
        else:
            eq = self.eq.subs(list(zip(self.mv, m)))
        relax = (sp.ones(*self.s.shape) - self.s).multiply_elementwise(sp.Matrix(m)) + self.s.multiply_elementwise(eq)
        alltogether(relax)
        return Eq(m, relax)

    def source_term_local(self, m):
        rhs_eq = [mm for mm in m]
        m_local = self.settings.get('m_local', False)
        split = self.settings.get('split', False)

        indices = []
        if m_local and not split:
            local_dict = {'m': m,
                          'consm': self.consm,
                          'sorder': None,
                          'default_index': [0],
                          }
        else:
            indices_str = ['ix_', 'iy_', 'iz_']
            lbm_ind = [ix, iy, iz]
            ind_to_subs = []
            for i, sorder in enumerate(self.sorder[1:]):
                indices.append(m[0].indices[sorder])
                ind_to_subs.extend([(indices_str[i], indices[i]),
                                    (lbm_ind[i], indices[i]),
                                    ])
            local_dict = {'m': m[0].base,
                          'consm': self.consm,
                          'sorder': self.sorder,
                          'default_index': indices,
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

    def relative_velocity(self, m):
        rel_vel = sp.Matrix(self.rel_vel).subs(list(zip(self.mv, m)))
        return [Eq(self.rel_vel_symb[i], rel_vel[i]) for i in range(self.dim)]

    def restore_conserved_moments(self, m, f):
        nconsm = len(self.consm)
        return Eq(m[:nconsm], sp.Matrix((self.Mu*f)[:nconsm]))

    def one_time_step_local(self, f, fnew, m):
        code = [self.transport_local(f, fnew),
                self.f2m_local(fnew, m)]

        if self.source_eq:
            code.extend(self.source_term_local(m))

        code.append(self.relaxation_local(m))

        if self.source_eq:
            code.extend(self.source_term_local(m))

        code.append(self.m2f_local(m, fnew))
        return code

    def transport(self):
        iloop = self._get_loop()
        f = self._get_indexed_2('f', iloop, -self.all_velocities)
        fnew = self._get_indexed_1('fnew', iloop)
        return {'code': For(iloop, self.transport_local(f, fnew))}

    def m2f(self):
        iloop = self._get_loop_0()
        f = self._get_indexed_1('f', iloop)
        m = self._get_indexed_1('m', iloop)
        return {'code': For(iloop, self.m2f_local(m, f))}

    def f2m(self):
        iloop = self._get_loop_0()
        f = self._get_indexed_1('f', iloop)
        m = self._get_indexed_1('m', iloop)
        return {'code': For(iloop, self.f2m_local(f, m))}

    def equilibrium(self):
        iloop = self._get_loop_0()
        m = self._get_indexed_1('m', iloop)
        return {'code': For(iloop, self.equilibrium_local(m))}

    def relaxation(self):
        iloop = self._get_loop_0()
        m = self._get_indexed_1('m', iloop)
        return {'code': For(iloop, self.relaxation_local(m))}

    def source_term(self):
        iloop = self._get_loop()
        m = self._get_indexed_1('m', iloop)
        return {'code': For(iloop, self.source_term_local(m))}

    def one_time_step(self):
        m_local = self.settings.get('m_local', False)
        check_isfluid = self.settings.get('check_isfluid', False)
        split = self.settings.get('split', False)

        iloop = self._get_loop()
        if m_local:
            if split:
                m = self._get_indexed_1('m', iloop)
                local_vars = [m[0]]
            else:
                m = sp.MatrixSymbol('m', self.ns, 1)
                local_vars = [m]
        else:
            m = self._get_indexed_1('m', iloop)
            local_vars = []

        if self.rel_vel_symb:
            local_vars.extend(self.rel_vel_symb)

        f = self._get_indexed_2('f', iloop, -self.all_velocities)
        fnew = self._get_indexed_1('fnew', iloop)

        internal = self.one_time_step_local(f, fnew, m)

        if check_isfluid:
            valin = sp.Symbol('valin', real=True)
            in_or_out = indexed('in_or_out', [self.ns, nx, ny, nz],
                                permutation=self.sorder, remove_ind=[0])
            loop = lambda x: For(iloop, If((Eq(in_or_out, valin), x)))
        else:
            loop = lambda x: For(iloop, x)
        
        if split:
            # code = [loop([*self.coords(), i]) for i in internal]
            code = [loop([i]) for i in internal]
        else:
            # code = loop([*self.coords(), *internal])
            code = loop([*internal])

        return {'code': code, 'local_vars': local_vars+self.local_vars}

    @monitor
    def generate(self):
        to_generate = [self.transport,
                       self.f2m,
                       self.m2f,
                       self.relaxation,
                       self.equilibrium,
                       self.one_time_step
                       ]

        if self.source_eq:
            to_generate.append(self.source_term)

        for gen in to_generate:
            name = gen.__name__
            output = gen()
            code = output['code']
            local_vars = output.get('local_vars', [])
            self.generator.add_routine((name, code), local_vars=local_vars)

    def _get_args(self, simulation, m_user=None, f_user=None):
        if m_user:
            mm = m_user
        else:
            mm = simulation.container.m
        m = mm.array

        dim = len(mm.nspace)
        nx = mm.nspace[0]
        X = simulation.domain.coords[0]
        if dim > 1:
            ny = mm.nspace[1]
            Y = simulation.domain.coords[1]
        if dim > 2:
            nz = mm.nspace[2]
            Z = simulation.domain.coords[1]

        if f_user:
            f = f_user.array
        else:
            f = simulation.container.F.array
        fnew = simulation.container.Fnew.array

        t = simulation.t
        dt = simulation.dt
        in_or_out = simulation.domain.in_or_out
        valin = simulation.domain.valin

        return locals()

    def call_function(self, function_name, simulation,
                      m_user=None, f_user=None):
        from ..symbolic import call_genfunction
        func = getattr(self.generator.module, function_name)

        args = self._get_args(simulation, m_user, f_user)
        call_genfunction(func, args)
