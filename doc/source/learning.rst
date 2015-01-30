Learning by examples
##############################


pyLBM can be a simple way to make numerical simulations
by using the Lattice Boltzmann method.
Once the module is installed by the command::

    python setup.py install

you just have to understand how build a dictionary that will be
understood by pyLBM to perform the simulation.
The dictionary should contain all the needed informations as
  - the geometry (see :doc:`here<learning_geometry>` for documentation)
  - the scheme (see :doc:`here<learning_scheme>` for documentation)
  - the boundary conditions (see :doc:`here<learning_bounds>` for documentation)
  - another informations like the space step, the scheme velocity, the generator
    of the functions... 
