The Domain of the simulation
##############################

With pylbm, the numerical simulations can be performed in a domain
with a complex geometry.
The creation of the geometry from a dictionary is explained `here <learning_geometry.html>`_.
All the informations needed to build the domain are defined through a dictionary
and put in a object of the class :py:class:`Domain <pylbm.domain.Domain>`.

The domain is built from three types of informations:

* a geometry (class :py:class:`Geometry <pylbm.geometry.Geometry>`),
* a stencil (class :py:class:`Stencil <pylbm.stencil.Stencil>`),
* a space step (a float for the grid step of the simulation).

The domain is a uniform cartesian discretization of the geometry with a grid step
:math:`dx`. The whole box is discretized even if some elements are added to reduce
the domain of the computation.
The stencil is necessary in order to know the maximal velocity in each direction
so that the corresponding number of phantom cells are added at the borders of
the domain (for the treatment of the boundary conditions).
The user can get the coordinates of the points in the domain by the fields
``x``, ``y``, and ``z``.
By convention, if the spatial dimension is one, ``y=z=None``; and if it is two, ``z=None``.

Several examples of domains can be found in
demo/examples/domain/

Examples in 1D
******************************

:download:`script<codes/domain_D1Q3_segment.py>`

The segment :math:`[0, 1]` with a :math:`D_1Q_3`
================================================

.. literalinclude:: codes/domain_D1Q3_segment.py
    :lines: 12-

.. plot:: codes/domain_D1Q3_segment.py

The segment :math:`[0,1]` is created by the dictionary with the key ``box``.
The stencil is composed by the velocity :math:`v_0=0`, :math:`v_1=1`, and
:math:`v_2=-1`. One phantom cell is then added at the left and at the right of
the domain.
The space step :math:`dx` is taken to :math:`0.1` to allow the visualization.
The result is then visualized with the distance of the boundary points
by using the method
:py:meth:`visualize<pylbm.domain.Domain.visualize>`.

:download:`script<codes/domain_D1Q5_segment.py>`

The segment :math:`[0, 1]` with a :math:`D_1Q_5`
================================================

.. literalinclude:: codes/domain_D1Q5_segment.py
    :lines: 12-

.. plot:: codes/domain_D1Q5_segment.py

The segment :math:`[0,1]` is created by the dictionary with the key ``box``.
The stencil is composed by the velocity :math:`v_0=0`, :math:`v_1=1`,
:math:`v_2=-1`, :math:`v_3=2`, :math:`v_4=-2`.
Two phantom cells are then added at the left and at the right of
the domain.
The space step :math:`dx` is taken to :math:`0.1` to allow the visualization.
The result is then visualized with the distance of the boundary points
by using the method
:py:meth:`visualize<pylbm.domain.Domain.visualize>`.


Examples in 2D
******************************

:download:`script<codes/domain_D2Q9_square.py>`

The square :math:`[0,1]^2` with a :math:`D_2Q_9`
================================================

.. literalinclude:: codes/domain_D2Q9_square.py
    :lines: 12-

.. plot::  codes/domain_D2Q9_square.py

The square :math:`[0,1]^2` is created by the dictionary with the key ``box``.
The stencil is composed by the nine velocities

.. math::
    :nowrap:

    \begin{equation}
    \begin{gathered}
    v_0=(0,0),\\
    v_1=(1,0), v_2=(0,1), v_3=(-1,0), v_4=(0,-1),\\
    v_5=(1,1), v_6=(-1,1), v_7=(-1,-1), v_8=(1,-1).
    \end{gathered}
    \end{equation}

One phantom cell is then added all around the square.
The space step :math:`dx` is taken to :math:`0.1` to allow the visualization.
The result is then visualized by using the method
:py:meth:`visualize <pylbm.domain.Domain.visualize>`.
This method can be used without parameter: the domain is visualize in white
for the fluid part (where the computation is done) and in black for the solid part
(the phantom cells or the obstacles).
An optional parameter view_distance can be used to visualize more precisely the
points (a black circle inside the domain and a square outside). Color lines are added
to visualize the position of the border: for each point that can reach the border
for a given velocity in one time step, the distance to the border is computed.

:download:`script 1<codes/domain_D2Q13_square_hole.py>`

A square with a hole with a :math:`D_2Q_{13}`
=============================================

The unit square :math:`[0,1]^2` can be holed with a circle.
In this example,
a solid disc lies in the fluid domain defined by
a :py:class:`circle <pylbm.elements.Circle>`
with a center of (0.5, 0.5) and a radius of 0.125

.. literalinclude:: codes/domain_D2Q13_square_hole.py
    :lines: 12-

.. plot:: codes/domain_D2Q13_square_hole.py


:download:`script <codes/domain_D2Q9_step.py>`

A step with a :math:`D_2Q_9`
==============================

A step can be build by removing a rectangle in the left corner.
For a :math:`D_2Q_9`, it gives the following domain.

.. literalinclude:: codes/domain_D2Q9_step.py
    :lines: 12-

.. plot:: codes/domain_D2Q9_step.py

Note that the distance with the bound is visible only for the specified labels.

Examples in 3D
******************************

:download:`script<codes/domain_D3Q19_cube.py>`

The cube :math:`[0,1]^3` with a :math:`D_3Q_{19}`
=================================================

.. literalinclude:: codes/domain_D3Q19_cube.py
    :lines: 12-

.. plot::  codes/domain_D3Q19_cube.py

The cube :math:`[0,1]^3` is created by the dictionary with the key ``box``
and the first 19th  velocities.
The result is then visualized by using the method
:py:meth:`visualize <pylbm.domain.Domain.visualize>`.

The cube with a hole with a :math:`D_3Q_{19}`
=================================================

.. literalinclude:: codes/domain_D3Q19_cube_hole.py
    :lines: 12-

.. plot::  codes/domain_D3Q19_cube_hole.py
