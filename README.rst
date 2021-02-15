pylbm
=====

|Binder| |GithubAction| |Doc badge| |Gitter Badge|

.. |Binder| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/pylbm/pylbm/master
.. |GithubAction| image:: https://github.com/pylbm/pylbm/workflows/ci/badge.svg
.. |Gitter Badge| image:: https://badges.gitter.im/pylbm/pylbm.svg
   :alt: Join the chat at https://gitter.im/pylbm/pylbm
   :target: https://gitter.im/pylbm/pylbm?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |Doc badge| image:: https://readthedocs.org/projects/pylbm/badge/?version=latest
   :target: http://pylbm.readthedocs.io/en/latest/

pylbm is an all-in-one package for numerical simulations using Lattice Boltzmann solvers.

This package gives all the tools to describe your lattice Boltzmann scheme in 1D, 2D and 3D problems.

We choose the D'Humières formalism to describe the problem. You can have complex geometry with a set of simple shape like circle, sphere, ...

pylbm performs the numerical scheme using Cython, NumPy or Loo.py from the scheme and the domain given by the user. Pythran and Numba wiil be available soon. pylbm has MPI support with mpi4py.

Installation
============

You can install pylbm in several ways

**With mamba or conda**

.. code:: bash

   mamba install pylbm -c conda-forge

.. code:: bash

   conda install pylbm -c conda-forge

With Pypi
---------

.. code::

   pip install pylbm

or

.. code::

   pip install pylbm --user

From source
-----------

You can also clone the project and install the latest version

.. code::

   git clone https://github.com/pylbm/pylbm

To install pylbm from source, we encourage you to create a fresh environment using conda.

.. code::

    conda create -n pylbm_env python

As mentioned at the end of the creation of this environment, you can activate it
using the comamnd line

.. code::

    conda activate pylbm_env

Now, you just have to go into the pylbm directory that you cloned and install
the dependencies

.. code::

    conda install --file requirements-dev.txt -c conda-forge

and then, install pylbm

.. code::

   python setup.py install

For more information about what you can achieve with pylbm, take a look at the documentation

`<http://pylbm.readthedocs.io>`_

