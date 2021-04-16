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
            # FIXME: hack for a bug in sympy
            l = [int(v) for v in all_vel[:, i]]
            vd = LA*sp.diag(*l)
            # vd = LA*sp.diag(*all_vel[:, i])
            Lambda.append(scheme.M*vd*scheme.invM)

        phi1 = sp.zeros(s.shape[0], 1) #pylint: disable=unsubscriptable-object
        inv_s = [1/v for v in s]
        sigma = sp.diag(*inv_s) - sp.eye(len(s))/2
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

    def vue(self):
        import jinja2

        try:
            import ipyvuetify as v
            import ipywidgets as widgets
        except ImportError:
            raise ImportError("Please install ipyvuetify")

        t, x, y, z, U, Fx, Fy, Fz, Delta = sp.symbols('t, x, y, z, U, F_x, F_y, F_z, Delta_t')
        Bxx, Bxy, Bxz = sp.symbols('B_{xx}, B_{xy}, B_{xz}')
        Byx, Byy, Byz = sp.symbols('B_{yx}, B_{yy}, B_{yz}')
        Bzx, Bzy, Bzz = sp.symbols('B_{zx}, B_{zy}, B_{zz}')

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
                phys_equation_rhs += sp.Derivative(B[i][j]*sp.Derivative(U, space[j]), space[i])

        order1_dict = {}
        F = [Fx, Fy, Fz]
        for d in range(self.dim):
            order1_dict[sp.latex(F[d])] = [sp.latex(c) for c in self.coeff_order1[d]]

        order0_template = jinja2.Template("""
        {%- macro coeff(order) %}
            {%- for o in order %}
                $$ {{ o }} $$
            {% endfor %}
        {%- endmacro %}
        {{ coeff(consm) }}
        """)

        order1_template = jinja2.Template("""
        {%- macro coeff_dict(consm, order) %}
            \\begin{align*}
            {%- for key, value in order.items() %}
                {%- for i in range(consm|length) %}
                    {{ key }}^{ {{ consm[i] }} } &= {{ value[i] }} \\\\ \\\\
                {% endfor %}
            {% endfor %}
            \\end{align*}
        {%- endmacro %}
        {{ coeff_dict(consm, order1_dict) }}
        """)

        order2_template = jinja2.Template("""
        {%- macro coeff_dict_2(consm, order) %}
            \\begin{align*}
            {%- for key, value in order.items() %}
                {%- for i in range(consm|length) %}
                    {%- for j in range(consm|length) %}
                        {{ key }}^{ {{ consm[i] }}, {{ consm[j] }} } &= {{ value[i*(consm|length) + j] }} \\\\ \\\\
                    {% endfor %}
                {% endfor %}
            {% endfor %}
            \\end{align*}
        {%- endmacro %}
        {{ coeff_dict_2(consm, order2_dict) }}
        """)

        order2_dict = {}
        for i in range(self.dim):
            for j in range(self.dim):
                order2_dict[sp.latex(B[i][j])] = [sp.latex(-Delta*c) for c in self.coeff_order2[i][j]]

        consm = [sp.latex(c) for c in self.consm]
        return v.Container(children=[
            v.Row(children=['The equivalent equation is given by']),
            v.Row(children=[
                widgets.HTMLMath(sp.latex(sp.Eq(phys_equation, phys_equation_rhs), mode='equation*'))
                ],
                justify='center',
            ),
            v.ExpansionPanels(children=[
                v.ExpansionPanel(children=[
                    v.ExpansionPanelHeader(children=['Conserved moments'], class_="title"),
                    v.ExpansionPanelContent(children=[
                        v.Row(children=[
                            widgets.HTMLMath(order0_template.render(consm=consm))
                            ],
                            justify='center'
                        )
                    ])
                ], class_="ma-2"),
                v.ExpansionPanel(children=[
                    v.ExpansionPanelHeader(children=['Order 1'], class_="title"),
                    v.ExpansionPanelContent(children=[
                        v.Row(children=[
                            widgets.HTMLMath(order1_template.render(consm=consm, order1_dict=order1_dict))
                            ],
                            justify='center'
                        )
                    ])
                ], class_="ma-2"),
                v.ExpansionPanel(children=[
                    v.ExpansionPanelHeader(children=['Order 2'], class_="title"),
                    v.ExpansionPanelContent(children=[
                        v.Row(children=[
                            widgets.HTMLMath(order2_template.render(consm=consm, order2_dict=order2_dict))
                            ],
                            justify='center'
                        )
                    ])
                ], class_="ma-2"),
            ])
        ])

    def _repr_mimebundle_(self, **kwargs):
        data = {
            'text/plain': repr(self),
        }
        data['application/vnd.jupyter.widget-view+json'] = {
            'version_major': 2,
            'version_minor': 0,
            'model_id': self.vue()._model_id
        }
        return data
