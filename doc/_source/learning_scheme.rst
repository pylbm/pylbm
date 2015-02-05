The Scheme
##############################

With pyLBM, elementary schemes can be gathered and coupled through the
equilibrium in order to simplify the implementation of the vectorial schemes.
Of course, the user can implement a single elementary scheme and then recover the
classical framework of the d'Humières schemes.

For pyLBM, the :py:class:`scheme<pyLBM.scheme.Scheme>` is performed
through a dictionary. The generalized d'Humières framework for vectorial schemes
is used. In the first section, we describe how build an elementary scheme. Then
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

  pyLBM uses the framework of d'Humières: the linear operator of the collision
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



Examples in 1D
==============================

The transport equation
------------------------------

:download:`script<codes/scheme_D1Q2_advection.py>`

A velocity :math:`c>0` being given, the system reads

.. math::
  :nowrap:

  \begin{equation*}\left\{
  \begin{aligned}
  &\partial_t u(t,x) + c \partial_x u(t,x) = 0, &&t>0, x\in{\mathbb R}, \\
  &u(0, x) = u_0(x),&& x\in{\mathbb R}.
  \end{aligned}\right.
  \end{equation*}

Taken for instance :math:`c=0.5`, the following scheme can be used:

.. literalinclude:: codes/scheme_D1Q2_advection.py
    :lines: 7-
    :linenos:

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


The Burger's equation
------------------------------

:download:`script<codes/scheme_D1Q2_Burgers.py>`

The system reads

.. math::
  :nowrap:

  \begin{equation*}\left\{
  \begin{aligned}
  &\partial_t u(t,x) + \tfrac{1}{2}\partial_x u^2(t,x) = 0, &&t>0, x\in{\mathbb R}, \\
  &u(0, x) = u_0(x),&& x\in{\mathbb R}.
  \end{aligned}\right.
  \end{equation*}

The following scheme can be used:

.. literalinclude:: codes/scheme_D1Q2_Burgers.py
    :lines: 7-
    :linenos:

The same dictionary has been used for this application with only one
modification: the equilibrium value of the second moment
:math:`m_1^{\textrm{eq}}` is taken to :math:`\tfrac{1}{2}m_0^2`.

The wave equation
------------------------------

:download:`script<codes/scheme_D1Q3_wave.py>`



The vectorial schemes
******************************
