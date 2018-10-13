---
title: '`pylbm`: A flexible Python package for lattice Boltzmann method'
tags:
  - Python
  - lattice Boltzmann
  - fl
  - galactic dynamics
  - milky way
authors:
  - name: Loïc Gouarin
    orcid: 0000-0003-4761-9989
    affiliation: 1
  - name: Benjamin Graille
    orcid: 0000-0000-0000-0000
    affiliation: 2
affiliations:
 - name: CMAP/CNRS, école polytechnique
   index: 1
 - name: LMO, Université Paris-Saclay
   index: 2
date: 13 October 2018
bibliography: paper.bib
---

# Summary

`pylbm` is an open source software written in Python [@pylbm]. It proposes a leading edge implementation of the Lattice Boltzmann Method for 1D/2D/3D problems and has been created to address various communities:

- **Mathematicians**: It provides a very pleasant environment in order to test new schemes, understand their mathematical properties, such as stability and consistency and open towards new frontiers in the field such as mesh refinement, equivalent equations...
- **Physicists**: `pylbm` offers an ideal framework to conduct numerical experiments for various levels of modeling and schemes, without having to dive into the mathematical machinery of LBM schemes. The high level of the proposed algorithm can also allow to go beyond the first test and easily conduct large scales simulations thanks to its parallel capability.
- **Computer scientists**: In `pylbm`, the Lattice Boltzmann Method is not hard-coded. Advanced tools of code generation, based on a large set of newly developed computer algebra libraries, allow a high level entry by the user of scheme definition and boundary conditions. The `pylbm` software then gererates the resulting numerical code. It is therefore possible to modify the code building kernels to test performance and code optimization on different architectures (AA patern and pull algorithm); the code can also be generated in different languages (C, C++, openCL, …).

The principle feature of `pylbm` is its high flexibility in term of LBM schemes and numerical code implementations. Moreover, it has excellent parallel capabilities and uses MPI for distributed computing and openCL for GPUs.

The generalized d’Humières framework is used to describe the schemes [@dhumiere_generalized_1992]. It's then easy to define your lattice Boltzmann scheme by providing the velocities, equilibrium values and the relaxation parameters, ... Moreover, you can have multiple $D_dQ_q$ schemes for your simulation where $d$ is the dimension and $q$ the number of velocities. That's generally the case when you want to simulate for example thermodynamic fluid flows as in the Rayleigh-Benard test case. But you can also experiment with new types of lattice Boltzmann schemes like vectorized schemes [@graille_approximation_2014] or with relative velocities [@dubois_lattice_2015].

`pylbm` will offer in the future releases more tools to help the user to design their lattice Boltzmann schemes and make large simulations with complex geometries.

- **More examples**: we want to give access to various lattice Boltzmann schemes you can find in the litterature. We will add multi-component flows, multi-phase flows, ... in order to have a full gallery of what we can do with LBM. We hope this way the users can improve this list with their own schemes.
- **Equivalent equations**: the hard part with the LBM is that you never write the physical equations you want to solve but the lattice Boltzmann scheme associated. We will offer the possibility to retrieve the physical equations from the given scheme by doing a Chapman-Enskog expansion for nonlinear equations up to the second order.
- **Complex geometries**: the geometry in `pylbm` can be described by a union of simple geometry elements like circle, triangle, sphere,... It's not realistic for industrial challenges and we will offer the possibility to use various CAD formats like STL.

# References
