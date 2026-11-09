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

With PyPI
---------

.. code:: bash

   uv pip install pylbm

or, without uv

.. code:: bash

   pip install pylbm

The PyPI package of mpi4py, a dependency of pylbm, needs an MPI library. Install
one with your system package manager (for example Open MPI or MPICH), or from
PyPI with ``pip install mpich``. The conda package installs it for you.

From source
-----------

You can also clone the project and work on the latest version with
`uv <https://docs.astral.sh/uv/>`_

.. code:: bash

   git clone https://github.com/pylbm/pylbm
   cd pylbm
   uv sync

``uv sync`` creates the environment ``.venv`` with pylbm and the tools to test
it. Then run the tests, or any script, through uv

.. code:: bash

   cd tests
   uv run pytest --mpl --h5diff -m "not gpu and not slow"

Add ``--extra gpu`` to ``uv sync`` for the Loopy (OpenCL) backend, and
``--group docs`` for the documentation.

For more information about what you can achieve with pylbm, take a look at the documentation

`<http://pylbm.readthedocs.io>`_
