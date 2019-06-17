# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Symbolic computation of equivalent equations
"""

#pylint: disable=invalid-name
import sympy as sp
from ..symbolic import alltogether

class EquivalentEquation:
    def __init__(self, scheme):
        # TODO: add source terms

        t, x, y, z = sp.symbols("t x y z", type='real')
        consm = list(scheme.consm.keys())
        nconsm = len(scheme.consm)

        self.consm = sp.Matrix(consm)
        self.dim = scheme.dim

        space = [x, y, z]

        LA = scheme.symb_la
        if not LA:
            LA = scheme.la

        func = []
        for i in range(nconsm):
            func.append(sp.Function('f{}'.format(i))(t, x, y, z)) #pylint: disable=not-callable
        func = sp.Matrix(func)

        sublist = [(i, j) for i, j in zip(consm, func)]
        sublist_inv = [(j, i) for i, j in zip(consm, func)]

        eq_func = sp.Matrix(scheme.EQ[nconsm:]).subs(sublist)
        s = sp.Matrix(scheme.s[nconsm:])

        all_vel = scheme.stencil.get_all_velocities()

        Lambda = []
        for i in range(all_vel.shape[1]):
            vd = LA*sp.diag(*all_vel[:, i])
            Lambda.append(scheme.M*vd*scheme.invM)

        phi1 = sp.zeros(s.shape[0], 1) #pylint: disable=unsubscriptable-object
        sigma = sp.diag(*s).inv() - sp.eye(len(s))/2
        gamma_1 = sp.zeros(nconsm, 1)

        self.coeff_order1 = []
        for dim, lambda_ in enumerate(Lambda):
            A, B = sp.Matrix([lambda_[:nconsm, :nconsm]]), sp.Matrix([lambda_[:nconsm, nconsm:]])
            C, D = sp.Matrix([lambda_[nconsm:, :nconsm]]), sp.Matrix([lambda_[nconsm:, nconsm:]])

            self.coeff_order1.append(A*func + B*eq_func)
            alltogether(self.coeff_order1[-1], nsimplify=True)
            for i in range(nconsm):
                gamma_1[i] += sp.Derivative(self.coeff_order1[-1][i], space[dim])

            dummy = -C*func - D*eq_func
            alltogether(dummy, nsimplify=True)
            for i in range(dummy.shape[0]):
                phi1[i] += sp.Derivative(dummy[i], space[dim])

        self.coeff_order2 = [[sp.zeros(nconsm) for _ in range(scheme.dim)] for _ in range(scheme.dim)]

        for dim, lambda_ in enumerate(Lambda):
            A, B = sp.Matrix([lambda_[:nconsm, :nconsm]]), sp.Matrix([lambda_[:nconsm, nconsm:]])

            meq = sp.Matrix(scheme.EQ[nconsm:])
            jac = meq.jacobian(consm)
            jac = jac.subs(sublist)
            delta1 = jac*gamma_1

            phi1_ = phi1 + delta1
            sphi1 = B*sigma*phi1_
            sphi1 = sphi1.doit()
            alltogether(sphi1, nsimplify=True)

            for i in range(scheme.dim):
                for jc in range(nconsm):
                    for ic in range(nconsm):
                        self.coeff_order2[dim][i][ic, jc] += sphi1[ic].expand().coeff(sp.Derivative(func[jc], space[i]))

        for ic, c in enumerate(self.coeff_order1):
            self.coeff_order1[ic] = c.subs(sublist_inv).doit()

        for ic, c in enumerate(self.coeff_order2):
            for jc, cc in enumerate(c):
                self.coeff_order2[ic][jc] = cc.subs(sublist_inv).doit()

    def __str__(self):
        from ..utils import header_string
        from ..jinja_env import env
        template = env.get_template('equivalent_equation.tpl')
        t, x, y, z, U, Fx, Fy, Fz, Delta = sp.symbols('t, x, y, z, U, Fx, Fy, Fz, Delta_t')
        Bxx, Bxy, Bxz = sp.symbols('Bxx, Bxy, Bxz')
        Byx, Byy, Byz = sp.symbols('Byx, Byy, Byz')
        Bzx, Bzy, Bzz = sp.symbols('Bzx, Bzy, Bzz')

        phys_equation = sp.Derivative(U, t) + sp.Derivative(Fx, x)
        if self.dim > 1:
            phys_equation += sp.Derivative(Fy, y)
        if self.dim == 3:
            phys_equation += sp.Derivative(Fz, z)

        order2 = []
        space = [x, y, z]
        B = [[Bxx, Bxy, Bxz],
             [Byx, Byy, Byz],
             [Bzx, Bzy, Bzz],
            ]

        phys_equation_rhs = 0
        for i in range(self.dim):
            for j in range(self.dim):
                order2.append(sp.pretty(sp.Eq(B[i][j], -Delta*self.coeff_order2[i][j], evaluate=False)))
                phys_equation_rhs += sp.Derivative(B[i][j]*sp.Derivative(U, space[j]), space[i])
        return template.render(header=header_string('Equivalent Equations'),
                               dim=self.dim,
                               phys_equation=sp.pretty(sp.Eq(phys_equation, phys_equation_rhs)),
                               conserved_moments=sp.pretty(sp.Eq(U, self.consm, evaluate=False)),
                               order1=[sp.pretty(sp.Eq(F, coeff, evaluate=False)) for F, coeff in zip([Fx, Fy, Fz][:self.dim], self.coeff_order1)],
                               order2=order2
                              )

    def __repr__(self):
        return self.__str__()
