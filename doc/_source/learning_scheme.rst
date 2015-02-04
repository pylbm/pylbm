The Scheme
******************************

With pyLBM, elementary schemes can be gathered and coupled through the
equilibrium in order to simplify the implementation of the vectorial schemes.
Of course, the user can implement a single elementary scheme and then recover the
classical framework of the d'Humières schemes.

For pyLBM, the scheme (:py:class:`pyLBM.scheme.Scheme`) is performed
through a dictionary. The generalized d'Humières framework for vectorial schemes
is used. In the first section, we describe how build an elementary scheme. Then
the vectorial schemes are introduced as coupled elementary schemes.

The elementary scheme
==============================

First, we need to define a stencil of velocities, build from a list of velocities
(:py:class:`pyLBM.stencil.Stencil`)



Examples in 1D
------------------------------

:download:`script<codes/scheme_D1Q2.py>`

.. literalinclude:: codes/scheme_D1Q2.py
    :lines: 7-
    :linenos:
