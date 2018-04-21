The Geometry of the simulation
##############################

With pylbm, the numerical simulations can be performed in a domain
with a complex geometry. This geometry is construct without considering a
particular mesh but only with geometrical objects.
All the geometrical informations are defined through a dictionary and
put into an object of the class :py:class:`Geometry <pylbm.geometry.Geometry>`.

First, the domain is put into a box: a segment in 1D, a rectangle in 2D, and
a rectangular parallelepipoid in 3D.

Then, the domain is modified by adding or deleting some elementary shapes.
In 2D, the elementary shapes are

* a :py:class:`Circle <pylbm.elements.Circle>`
* an :py:class:`Ellipse <pylbm.elements.Ellipse>`
* a :py:class:`Parallelogram <pylbm.elements.Parallelogram>`
* a :py:class:`Triangle <pylbm.elements.Triangle>`

From version 0.2, the geometrical elements are implemented in 3D.
The elementary shapes are

* a :py:class:`Sphere <pylbm.elements.Sphere>`
* an :py:class:`Ellipsoid <pylbm.elements.Ellipsoid>`
* a :py:class:`Parallelepiped <pylbm.elements.Parallelepiped>`
* a Cylinder with a 2D-base

  - :py:class:`Cylinder (Circle) <pylbm.elements.Cylinder_Circle>`
  - :py:class:`Cylinder (Ellipse) <pylbm.elements.Cylinder_Ellipse>`
  - :py:class:`Cylinder (Triangle) <pylbm.elements.Cylinder_Triangle>`

Several examples of geometries can be found in
demo/examples/geometry/

Examples in 1D
******************************

:download:`script<codes/geometry_1D_segment.py>`

The segment :math:`[0, 1]`
==============================

.. literalinclude:: codes/geometry_1D_segment.py
    :lines: 11-

.. plot:: codes/geometry_1D_segment.py

The segment :math:`[0,1]` is created by the dictionary with the key ``box``.
We then add the labels 0 and 1 on the edges with the key ``label``.
The result is then visualized with the labels by using the method
:py:meth:`visualize<pylbm.geometry.Geometry.visualize>`.
If no labels are given in the dictionary, the default value is -1.


Examples in 2D
******************************

:download:`script<codes/geometry_2D_square_label.py>`

The square :math:`[0,1]^2`
==============================

.. literalinclude:: codes/geometry_2D_square.py
    :lines: 11-

.. plot:: codes/geometry_2D_square.py

The square :math:`[0,1]^2` is created by the dictionary with the key ``box``.
The result is then visualized by using the method
:py:meth:`visualize <pylbm.geometry.Geometry.visualize>`.

We then add the labels on each edge of the square
through a list of integers with the conventions:

.. hlist::
  :columns: 2

  * first for the left (:math:`x=x_{\operatorname{min}}`)
  * third for the bottom (:math:`y=y_{\operatorname{min}}`)
  * second for the right (:math:`x=x_{\operatorname{max}}`)
  * fourth for the top (:math:`y=y_{\operatorname{max}}`)

.. literalinclude:: codes/geometry_2D_square_label.py
    :lines: 11-

.. plot:: codes/geometry_2D_square_label.py

If all the labels have the same value, a shorter solution is to
give only the integer value of the label instead of the list.
If no labels are given in the dictionary, the default value is -1.

:download:`script 3<codes/geometry_2D_square_parallelogram.py>`
:download:`script 2<codes/geometry_2D_square_triangle.py>`
:download:`script 1<codes/geometry_2D_square_hole.py>`

A square with a hole
==============================

The unit square :math:`[0,1]^2` can be holed with a circle (script 1)
or with a triangular or with a parallelogram (script 3)

In the first example,
a solid disc lies in the fluid domain defined by
a :py:class:`circle <pylbm.elements.Circle>`
with a center of (0.5, 0.5) and a radius of 0.125

.. literalinclude:: codes/geometry_2D_square_hole.py
    :lines: 11-

.. plot:: codes/geometry_2D_square_hole.py

The dictionary of the geometry then contains an additional key ``elements``
that is a list of elements.
In this example, the circle is labelized by 1 while the edges of the square by 0.

The element can be also a :py:class:`triangle <pylbm.elements.Triangle>`

.. literalinclude:: codes/geometry_2D_square_triangle.py
    :lines: 11-

.. plot:: codes/geometry_2D_square_triangle.py

or a :py:class:`parallelogram <pylbm.elements.Parallelogram>`

.. literalinclude:: codes/geometry_2D_square_parallelogram.py
    :lines: 11-

.. plot:: codes/geometry_2D_square_parallelogram.py

:download:`script <codes/geometry_2D_cavity.py>`

A complex cavity
==============================

A complex geometry can be build by using a list of elements. In this example,
the box is fixed to the unit square :math:`[0,1]^2`. A square hole is added with the
argument ``isfluid=False``. A strip and a circle are then added with the argument
``isfluid=True``. Finally, a square hole is put. The value of ``elements``
contains the list of all the previous elements. Note that the order of
the elements in the list is relevant.

.. literalinclude:: codes/geometry_2D_cavity.py
    :lines: 11-19

Once the geometry is built, it can be modified by adding or deleting
other elements. For instance, the four corners of the cavity can be rounded
in this way.

.. literalinclude:: codes/geometry_2D_cavity.py
    :lines: 21-

.. plot:: codes/geometry_2D_cavity.py


Examples in 3D
******************************

:download:`script<codes/geometry_3D_cube.py>`

The cube :math:`[0,1]^3`
==============================

.. literalinclude:: codes/geometry_3D_cube.py
    :lines: 12-

.. plot:: codes/geometry_3D_cube.py

The cube :math:`[0,1]^3` is created by the dictionary with the key ``box``.
The result is then visualized by using the method
:py:meth:`visualize <pylbm.geometry.Geometry.visualize>`.

We then add the labels on each edge of the square
through a list of integers with the conventions:

.. hlist::
  :columns: 2

  * first for the left (:math:`x=x_{\operatorname{min}}`)
  * third for the bottom (:math:`y=y_{\operatorname{min}}`)
  * fifth for the front (:math:`z=z_{\operatorname{min}}`)
  * second for the right (:math:`x=x_{\operatorname{max}}`)
  * fourth for the top (:math:`y=y_{\operatorname{max}}`)
  * sixth for the back (:math:`z=z_{\operatorname{max}}`)

If all the labels have the same value, a shorter solution is to
give only the integer value of the label instead of the list.
If no labels are given in the dictionary, the default value is -1.

The cube :math:`[0,1]^3` with a hole
====================================

.. literalinclude:: codes/geometry_3D_cube_hole.py
    :lines: 11-

.. plot:: codes/geometry_3D_cube_hole.py

The cube :math:`[0,1]^3` and the spherical hole are created
by the dictionary with the keys ``box`` and ``elements``.
The result is then visualized by using the method
:py:meth:`visualize <pylbm.geometry.Geometry.visualize>`.
