The Geometry of the simulation
##############################

With pyLBM, the numerical simulations can be performed in a domain
with a complex geometry. This geometry is construct without considering a
particular mesh but only with geometrical objects.
All the geometrical informations are defined through a dictionary and
put in an object of the class :py:class:`Geometry <pyLBM.geometry.Geometry>`.

Fist, the domain is put into a box: a segment in 1D, a rectangle in 2D, and
a rectangular parallelepipoid in 3D.

Then, the domain is modified by adding or deleting some elementary shapes.
In 2D, the elementary shapes are

* a :py:class:`Circle <pyLBM.elements.Circle>`
* a :py:class:`Parallelogram <pyLBM.elements.Parallelogram>`
* a :py:class:`Triangle <pyLBM.elements.Triangle>`

In 3D, the elementary shapes are not yet implemented.


Several examples of geometries can be found in
demo/examples/geometry/

Examples in 1D
******************************

The segment [0, 1]
==============================

:download:`script<codes/geometry_1D_segment.py>`

.. literalinclude:: codes/geometry_1D_segment.py
    :lines: 11-
    :linenos:

The segment [0,1] is created by the dictionary with the key ``box``.
We then add the labels 0 and 1 on the ends with the key ``label``.
The result is then visualized with the labels by using the method
:py:meth:`visualize<pyLBM.geometry.Geometry.visualize>`.
If no labels are given in the dictionary, the default value is 0.


Examples in 2D
******************************

A simple squared domain
==============================

:download:`script<codes/geometry_2D_square_label.py>`

.. literalinclude:: codes/geometry_2D_square.py
    :lines: 11-
    :linenos:

The square [0,1]x[0,1] is created by the dictionary with the key ``box``.
The result is then visualized by using the method
:py:meth:`visualize <pyLBM.geometry.Geometry.visualize>`.

We then add the labels on each edge of the square
through a list of integers with the conventions:

.. hlist::
  :columns: 2

  * first for the bottom
  * second for the right
  * third for the top
  * fourth for the left

.. literalinclude:: codes/geometry_2D_square_label.py
    :lines: 11-
    :linenos:

If all the labels have the same value, a shorter solution is to
give only the integer value of the label instead of the list.
If no labels are given in the dictionary, the default value is 0.

A square with a hole
==============================

:download:`script 1<codes/geometry_2D_square_hole.py>`
:download:`script 2<codes/geometry_2D_square_triangle.py>`
:download:`script 3<codes/geometry_2D_square_parallelogram.py>`

The unit square [0,1]x[0,1] can be holed with a circle (script 1)
or with a triangular or with a parallelogram (script 3)

In the first example,
a solid disc lies in the fluid domain defined by
a :py:class:`circle <pyLBM.elements.Circle>`
with a center of (0.5, 0.5) and a radius of 0.125

.. literalinclude:: codes/geometry_2D_square_hole.py
    :lines: 11-
    :linenos:

The dictionary of the geometry then contains an additional key ``elements``
that is a list of elements.
In this example, the circle is labelized by 1 while the edges of the square by 0.

The element can be also a :py:class:`triangle <pyLBM.elements.Triangle>`

.. literalinclude:: codes/geometry_2D_square_triangle.py
    :lines: 11-
    :linenos:

or a :py:class:`parallelogram <pyLBM.elements.Parallelogram>`

.. literalinclude:: codes/geometry_2D_square_parallelogram.py
    :lines: 11-
    :linenos:

A complex cavity
==============================

:download:`script <codes/geometry_2D_cavity.py>`

A complex geometry can be build by using a list of elements. In this example,
the box is fixed to the unit square [0,1]x[0,1]. A square hole is added with the
argument ``isfluid=False``. A strip and a circle are then added with the argument
``isfluid=True``. Finally, a square hole is put. The value of ``elements``
contains the list of all the previous elements. Note that the order of
the elements in the list is relevant.

.. literalinclude:: codes/geometry_2D_cavity.py
    :lines: 11-19
    :linenos:

.. image:: /images/geometry_2D_cavity_1.png

Once the geometry is built, it can be modified by adding or deleting
other elements. For instance, the four corners of the cavity can be rounded
in this way.

.. literalinclude:: codes/geometry_2D_cavity.py
    :lines: 21-
    :linenos:

.. image:: /images/geometry_2D_cavity_2.png


Examples in 3D
******************************

TODO