The Geometry of the simulation
##############################

With pyLBM, the numerical simulations can be performed in a domain
with a complex geometry. This geometry is construct without considering a
particular mesh but only with geometrical objects.
All the geometrical informations are defined through a dictionary and
put in an object of the class :py:class:`pyLBM.Geometry`.

Fist, the domain is put into a box: a segment in 1D, a rectangle in 2D, and
a rectangular parallelepipoid in 3D.

Then, the domain is modified by adding or deleting some elementary shapes.
In 2D, the elementary shapes are

* a circle (:py:class:`pyLBM.Circle`)
* a parallelogram (:py:class:`pyLBM.Parallelogram`)
* a triangle (:py:class:`pyLBM.Triangle`)

In 3D, the elementary shapes are not yet implemented.


Examples
******************************

Several examples of geometries can be found in
demo/examples/geometry/

Examples in 1D
==============================

The segment [0, 1]
------------------------------
:download:`script<codes/geometry_1D_segment.py>`

.. literalinclude:: codes/geometry_1D_segment.py
    :lines: 11-
    :linenos:

The segment [0,1] is created by the dictionary with the key ``box``.
We then add the labels 0 and 1 on the ends with the key ``label``.
The result is then visualized with the labels by using the method
:py:meth:`visualize` of the class :py:class:`pyLBM.Geometry`.
If no labels are given in the dictionary, the default value is 0.


Examples in 2D
==============================

A simple squared domain
------------------------------
:download:`script<codes/geometry_2D_square_label.py>`

.. literalinclude:: codes/geometry_2D_square.py
    :lines: 11-
    :linenos:

The square [0,1]x[0,1] is created by the dictionary with the key ``box``.
The result is then visualized by using the method
:py:meth:`visualize` of the class :py:class:`pyLBM.Geometry`.

We then add the labels on each edge of the square
through a list of integers with the conventions:
    - first for the bottom
    - second for the right
    - third for the top
    - fourth for the left

.. literalinclude:: codes/geometry_2D_square_label.py
    :lines: 11-
    :linenos:

If all the labels have the same value, a shorter solution is to
give only the integer value of the label instead of the list.
If no labels are given in the dictionary, the default value is 0.

A square with a hole
------------------------------

The second example is the same square with a hole.
A solid disc lies in the fluid domain defined by
a circle with a center of (.5, .5) and a radius of .125
(:py:class:`pyLBM.Circle`)::

    dgeom = {
        'box':{'x': [0, 1], 'y': [0, 1], 'label':0},
        'elements':[pyLBM.Circle((0.5,0.5), 0.125, label = 1)],
    }
    geom = pyLBM.Geometry(dgeom)
    geom.visualize(viewlabel=True)

In this example, the circle is labelized by 1 while the edges of the square by 0.



Examples in 3D
==============================

TODO
