The Scheme
##############################

With pylbm, elementary schemes can be gathered and coupled through the
equilibrium in order to simplify the implementation of the vectorial schemes.
Of course, the user can implement a single elementary scheme and then recover the
classical framework of the d'Humières schemes.

For pylbm, the :py:class:`scheme<pylbm.scheme.Scheme>` is performed
through a dictionary. The generalized d'Humières framework for vectorial schemes
is used [dH92]_, [G14]_. In the first section, we describe how build an elementary scheme. Then
the vectorial schemes are introduced as coupled elementary schemes.

The elementary schemes
******************************

Let us first consider a regular lattice :math:`L` in dimension :math:`d`
with a typical mesh size :math:`dx`, and the time step :math:`dt`.
The scheme velocity :math:`\lambda` is then defined by
:math:`\lambda = dx/dt`.
We introduce a set of :math:`q` velocities adapted to this lattice
:math:`\{v_0, \ldots, v_{q-1}\}`, that is
to say that, if :math:`x` is a point of the lattice :math:`L`, the point
:math:`x+v_jdt` is on the lattice for every :math:`j\in\{0,\ldots,q{-}1\}`.

The aim of the :math:`DdQq` scheme is to compute a distribution function
vector :math:`{\boldsymbol f} = (f_0,\ldots,f_{q-1})` on the lattice
:math:`L` at discret values of time.
The scheme splits into two phases:
the relaxation and the transport. That is, the passage from the time :math:`t`
to the time :math:`t+dt` consists in the succession of these two phases.

- the relaxation phase

  This phase, also called collision, is local in space:
  on every site :math:`x` of the lattice,
  the values of the vector :math:`{\boldsymbol f}` are modified,
  the result after the collision being denoted by
  :math:`{\boldsymbol f}^\star`.
  The operator of collision is a linear operator of relaxation
  toward an equilibrium value denoted
  :math:`{\boldsymbol f}^{\textrm{eq}}`.

  pylbm uses the framework of d'Humières: the linear operator of the collision
  is diagonal in a special basis called moments denoted by
  :math:`{\boldsymbol m} = (m_0,\ldots,m_{q-1})`.
  The change-of-basis matrix :math:`M` is such that
  :math:`{\boldsymbol m} = M{\boldsymbol f}`
  and
  :math:`{\boldsymbol f} = M^{-1}{\boldsymbol m}`.
  In the basis of the moments, the collision operator then just reads

  .. math::
     :nowrap:

     \begin{equation*}
     m_k^\star = m_k - s_k (m_k - m_k^{\textrm{eq}}),
     \qquad
     0\leqslant k\leqslant q{-}1,
     \end{equation*}

  where :math:`s_k` is the relaxation parameter associated to the kth moment.
  The kth moment is said conserved during the collision
  if the associated relaxation parameter :math:`s_k=0`.

  By analogy with the kinetic theory,
  the change-of-basis matrix :math:`M` is defined by a set of polynomials
  with :math:`d` variables :math:`(P_0,\ldots,P_{q-1})` by

  .. math::
     :nowrap:

     \begin{equation*}
     M_{ij} = P_i(v_j).
     \end{equation*}

- the transport phase

  This phase just consists in a shift of the indices and reads

  .. math::
     :nowrap:

     \begin{equation*}
     f_j(x, t+dt) = f_j^\star(x-v_jdt, t),
     \qquad
     0\leqslant j\leqslant q{-}1,
     \end{equation*}

Notations
==============================

The :py:class:`scheme<pylbm.scheme.Scheme>` is defined and build
through a dictionary in pylbm. Let us first list the several key words
of this dictionary:

- ``dim``: the spatial dimension. This argument is optional if the geometry is
  known, that is if the dimension can be computed through the list of the variables;
- ``scheme_velocity``: the velocity of the scheme denoted by :math:`\lambda` in the
  previous section and defined as the spatial step over the time step
  (:math:`\lambda=dx/dt`) ;
- ``schemes``: the list of the schemes. In pylbm, several coupled schemes can be used,
  the coupling being done through the equilibrium values of the moments.
  Some examples with only one scheme and with more than one schemes are given in the next sections.
  Each element of the list should be a dictionay with the following key words:

  - ``velocities``: the list of the velocity indices,
  - ``conserved_moments``: the list of the conserved moments (list of symbolic variables),
  - ``polynomials``: the list of the polynomials that define the moments, the polynomials are built with the symbolic variables X, Y, and Z,
  - ``equilibrium``: the list of the equilibrium value of the moments,
  - ``relaxation_parameters``: the list of the relaxation parameters, (by convention, the relaxation parameter of a conserved moment is taken to 0).

Examples in 1D
==============================

:download:`script<codes/scheme_D1Q2_advection.py>`

:math:`D_1Q_2` for the advection
--------------------------------

A velocity :math:`c\in{\mathbb R}` being given, the advection equation reads

.. math::
  :nowrap:

  \begin{equation*}
  \partial_t u(t,x) + c \partial_x u(t,x) = 0, \qquad t>0, x\in{\mathbb R}.
  \end{equation*}

Taken for instance :math:`c=0.5`, the following scheme can be used:

.. literalinclude:: codes/scheme_D1Q2_advection.py
    :lines: 11-

The dictionary ``d`` is used to set the dimension to 1,
the scheme velocity to 1. The used scheme has two velocities:
the first one :math:`v_0=1` and the second one :math:`v_1=-1`.
The polynomials that define the moments are
:math:`P_0 = 1` and :math:`P_1 = X` so that
the matrix of the moments is

.. math::
  :nowrap:

  \begin{equation*} M =
  \begin{pmatrix}
  1&1\\ 1&-1
  \end{pmatrix}
  \end{equation*}

with the convention :math:`M_{ij} = P_i(v_j)`.
Then, there are two distribution functions :math:`f_0` and
:math:`f_1` that move at the velocities :math:`v_0` and :math:`v_1`,
and two moments :math:`m_0 = f_0+f_1` and :math:`m_1 = f_0-f_1`.
The first moment :math:`m_0` is conserved during the relaxation phase
(as the associated relaxation parameter is set to 0),
while the second moment :math:`m_1` relaxes to its equilibrium value
:math:`0.5m_0` with a relaxation parameter :math:`1.9` by the relation

.. math::
  :nowrap:

  \begin{equation*}
  m_1^\star = m_1 - 1.9 (m_1 - 0.5m_0).
  \end{equation*}

:download:`script<codes/scheme_D1Q2_Burgers.py>`

:math:`D_1Q_2` for Burger's
------------------------------

The Burger's equation reads

.. math::
  :nowrap:

  \begin{equation*}
  \partial_t u(t,x) + \tfrac{1}{2}\partial_x u^2(t,x) = 0, \qquad t>0, x\in{\mathbb R}.
  \end{equation*}

The following scheme can be used:

.. literalinclude:: codes/scheme_D1Q2_Burgers.py
    :lines: 11-

The same dictionary has been used for this application with only one
modification: the equilibrium value of the second moment
:math:`m_1^{\textrm{eq}}` is taken to :math:`\tfrac{1}{2}m_0^2`.

:download:`script<codes/scheme_D1Q3_wave.py>`

:math:`D_1Q_3` for the wave equation
------------------------------------

The wave equation is rewritten into the system of two partial differential equations

.. math::
  :nowrap:

  \begin{equation*}
  \left\{
  \begin{aligned}
  &\partial_t u(t, x) + \partial_x v(t, x) = 0, & t>0, x\in{\mathbb R},\\
  &\partial_t v(t, x) + c^2\partial_x u(t, x) = 0, & t>0, x\in{\mathbb R}.
  \end{aligned}
  \right.
  \end{equation*}

The following scheme can be used:

.. literalinclude:: codes/scheme_D1Q3_wave.py
    :lines: 11-


Examples in 2D
==============================

:download:`script<codes/scheme_D2Q4_advection.py>`

:math:`D_2Q_4` for the advection
--------------------------------

A velocity :math:`(c_x, c_y)\in{\mathbb R}^2` being given,
the advection equation reads

.. math::
  :nowrap:

  \begin{equation*}
  \partial_t u(t, x, y) +
  c_x \partial_x u(t, x, y) +
  c_y \partial_y u(t, x, y) = 0,
  \qquad t>0, x, y \in{\mathbb R}.
  \end{equation*}

Taken for instance :math:`c_x=0.1, c_y=0.2`, the following scheme can be used:

.. literalinclude:: codes/scheme_D2Q4_advection.py
    :lines: 11-

The dictionary ``d`` is used to set the dimension to 2,
the scheme velocity to 1. The used scheme has four velocities:
:math:`v_0=(1,0)`, :math:`v_1=(0,1)`, :math:`v_2=(-1,0)`, and
:math:`v_3=(0,-1)`.
The polynomials that define the moments are
:math:`P_0 = 1`, :math:`P_1 = X`, :math:`P_2 = Y`, and
:math:`P_3 = X^2-Y^2` so that
the matrix of the moments is

.. math::
  :nowrap:

  \begin{equation*} M =
  \begin{pmatrix}
  1&1&1&1\\ 1&0&-1&0\\ 0&1&0&-1\\ 1&-1&1&-1
  \end{pmatrix}
  \end{equation*}

with the convention :math:`M_{ij} = P_i(v_j)`.
Then, there are four distribution functions :math:`f_j, 0\leq j\leq 3`
that move at the velocities :math:`v_j`,
and four moments :math:`m_k = \sum_{j=0}^3 M_{kj}f_j`.
The first moment :math:`m_0` is conserved during the relaxation phase
(as the associated relaxation parameter is set to 0),
while the other moments :math:`m_k, 1\leq k\leq 3` relaxe to their equilibrium values
by the relations

.. math::
  :nowrap:

  \begin{equation*}
  \left\{
  \begin{aligned}
  m_1^\star &= m_1 - 1.9 (m_1 - 0.1m_0),\\
  m_2^\star &= m_2 - 1.9 (m_2 - 0.2m_0),\\
  m_3^\star &= (1-1.4)m_3.
  \end{aligned}
  \right.
  \end{equation*}

:download:`script<codes/scheme_D2Q9_Navier-Stokes.py>`

:math:`D_2Q_9` for Navier-Stokes
--------------------------------

The system of the compressible Navier-Stokes equations
reads

.. math::
  :nowrap:

  \begin{equation*}
  \left\{
  \begin{aligned}
  &\partial_t \rho + \nabla{\cdot}(\rho {\boldsymbol u}) = 0,\\
  &\partial_t (\rho {\boldsymbol u}) + \nabla{\cdot}(\rho {\boldsymbol u}{\otimes}{\boldsymbol u})
  + \nabla p = \kappa \nabla (\nabla{\cdot}{\boldsymbol u}) + \eta \nabla{\cdot}
  \bigl(\nabla{\boldsymbol u} + (\nabla{\boldsymbol u})^T - \nabla{\cdot}{\boldsymbol u}{\mathbb I}\bigr),
  \end{aligned}
  \right.
  \end{equation*}

where we removed the dependency of all unknown on the variables :math:`(t, x, y)`.
The vector :math:`{\boldsymbol x}` stands for :math:`(x, y)` and,
if :math:`\psi` is a scalar function of :math:`{\boldsymbol x}` and
:math:`{\boldsymbol\phi}=(\phi_x,\phi_y)`
is a vectorial function of :math:`{\boldsymbol x}`,
the usual partial differential operators read

.. math::
  :nowrap:

  \begin{equation*}
  \begin{aligned}
  &\nabla\psi = (\partial_x\psi, \partial_y\psi),\\
  &\nabla{\cdot}{\boldsymbol\phi} = \partial_x\phi_x + \partial_y\phi_y,\\
  &\nabla{\cdot}({\boldsymbol\phi}{\otimes}{\boldsymbol\phi}) = (\nabla{\cdot}(\phi_x{\boldsymbol\phi}), \nabla{\cdot}(\phi_y{\boldsymbol\phi})).
  \end{aligned}
  \end{equation*}

The coefficients :math:`\kappa` and :math:`\eta` are the bulk and the shear viscosities.

The following dictionary can be used to simulate the system of Navier-Stokes equations
in the limit of small velocities:

.. literalinclude:: codes/scheme_D2Q9_Navier-Stokes.py
    :lines: 11-

The scheme generated by the dictionary is the 9 velocities scheme with orthogonal
moments introduced in [QdHL92]_

Examples in 3D
==============================

:download:`script<codes/scheme_D3Q6_advection.py>`

:math:`D_3Q_6` for the advection
--------------------------------

A velocity :math:`(c_x, c_y, c_z)\in{\mathbb R}^2` being given,
the advection equation reads

.. math::
  :nowrap:

  \begin{equation*}
  \partial_t u(t, x, y, z) +
  c_x \partial_x u(t, x, y, z) +
  c_y \partial_y u(t, x, y, z) +
  c_z \partial_z u(t, x, y, z) = 0,
  \quad t>0, x, y, z \in{\mathbb R}.
  \end{equation*}

Taken for instance :math:`c_x=0.1, c_y=-0.1, c_z=0.2`, the following scheme can be used:

.. literalinclude:: codes/scheme_D3Q6_advection.py
    :lines: 11-

The dictionary ``d`` is used to set the dimension to 3,
the scheme velocity to 1. The used scheme has six velocities:
:math:`v_0=(0,0,1)`,
:math:`v_1=(0,0,-1)`,
:math:`v_2=(0,1,0)`,
:math:`v_3=(0,-1,0)`,
:math:`v_4=(1,0,0)`, and
:math:`v_5=(-1,0,0)`.
The polynomials that define the moments are
:math:`P_0 = 1`, :math:`P_1 = X`, :math:`P_2 = Y`, :math:`P_3 = Z`,
:math:`P_4 = X^2-Y^2`, and :math:`P_5 = X^2-Z^2` so that
the matrix of the moments is

.. math::
  :nowrap:

  \begin{equation*} M =
  \begin{pmatrix}
  1&1&1&1&1&1\\
  0&0&0&0&1&-1\\
  0&0&1&-1&0&0\\
  1&-1&0&0&0&0\\
  0&0&-1&-1&1&1\\
  -1&-1&0&0&1&1
  \end{pmatrix}
  \end{equation*}

with the convention :math:`M_{ij} = P_i(v_j)`.
Then, there are six distribution functions :math:`f_j, 0\leq j\leq 5`
that move at the velocities :math:`v_j`,
and six moments :math:`m_k = \sum_{j=0}^5 M_{kj}f_j`.
The first moment :math:`m_0` is conserved during the relaxation phase
(as the associated relaxation parameter is set to 0),
while the other moments :math:`m_k, 1\leq k\leq 5` relaxe to their equilibrium values
by the relations

.. math::
  :nowrap:

  \begin{equation*}
  \left\{
  \begin{aligned}
  m_1^\star &= m_1 - 1.5 (m_1 - 0.1m_0),\\
  m_2^\star &= m_2 - 1.5 (m_2 + 0.1m_0),\\
  m_3^\star &= m_3 - 1.5 (m_3 - 0.2m_0),\\
  m_4^\star &= (1-1.5)m_4,\\
  m_5^\star &= (1-1.5)m_5.
  \end{aligned}
  \right.
  \end{equation*}


The vectorial schemes
******************************

With pylbm, vectorial schemes can be built easily by using a list of elementary schemes.
Each elementary scheme is given by a dictionary as in the previous section.
The conserved moments of all the elementary schemes can be used in the equilibrium values of the non conserved moments,
in order to couple the schemes. For more details on the vectorial schemes,
the reader can refer to [G14]_.

Examples in 1D
==============================

:download:`script<codes/scheme_D1Q22_shallow_water.py>`

:math:`D_1Q_{2,2}` for the shallow water equation
--------------------------------------------------

A constant :math:`g\in{\mathbb R}` being given, the shallow water system reads

.. math::
  :nowrap:

  \begin{align*}
  &\partial_t h(t,x) + \partial_x q(t,x) = 0, &\qquad t>0, x\in{\mathbb R},\\
  &\partial_t q(t,x) + \partial_x \bigl(q^2(t,x)/h(t,x) + gh^2(t,x)/2\bigr) = 0, &\qquad t>0, x\in{\mathbb R}.
  \end{align*}

Taken for instance :math:`g=1`, the following scheme can be used:

.. literalinclude:: codes/scheme_D1Q22_shallow_water.py
    :lines: 11-

Two elementary schemes have been built, these two schemes are identical
except for the equilibrium values of the non conserved moment and of the relaxation parameter:
The first one is used to simulate the equation on :math:`h` and the second one
to simulate the equation on :math:`q`.
For each scheme, the equilibrium value of the non conserved moment is equal
to the flux of the corresponding equation: the equilibrium value of the kth scheme
can so depend on all the conserved moments (and not only on those of the kth scheme).

Examples in 2D
==============================

:download:`script<codes/scheme_D2Q444_shallow_water.py>`

:math:`D_2Q_{4,4,4}` for the shallow water equation
---------------------------------------------------

A constant :math:`g\in{\mathbb R}` being given, the shallow water system reads

.. math::
  :nowrap:

  \begin{align*}
  &\partial_t h(t,x,y) + \partial_x q_x(t,x,y) + \partial_y q_y(t,x,y) = 0, &\qquad t>0, x,y\in{\mathbb R},\\
  &\partial_t q_x(t,x,y) + \partial_x \bigl(q_x^2(t,x,y)/h(t,x,y) + gh^2(t,x,y)/2\bigr) \\
  & \phantom{\partial_t q_x(t,x,y)} + \partial_y \bigl( q_x(t,x,y)q_y(t,x,y)/h(t,x,y)\bigr) = 0, &\qquad t>0, x,y\in{\mathbb R},\\
  &\partial_t q_y(t,x,y) + \partial_x \bigl( q_x(t,x,y)q_y(t,x,y)/h(t,x,y)\bigr)\\
  & \phantom{\partial_t q_y(t,x,y)} + \partial_y \bigl(q_y^2(t,x,y)/h(t,x,y) + gh^2(t,x,y)/2\bigr) = 0, &\qquad t>0, x,y\in{\mathbb R}.
  \end{align*}

Taken for instance :math:`g=1`, the following scheme can be used:

.. literalinclude:: codes/scheme_D2Q444_shallow_water.py
    :lines: 11-

Three elementary schemes have been built, these three schemes are identical
except for the equilibrium values of the non conserved moment and of the relaxation parameter:
The first one is used to simulate the equation on :math:`h` and the others
to simulate the equation on :math:`q_x` and :math:`q_y`.
For each scheme, the equilibrium value of the non conserved moment is equal
to the flux of the corresponding equation: the equilibrium value of the kth scheme
can so depend on all the conserved moments (and not only on those of the kth scheme).
