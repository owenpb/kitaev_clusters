
kitaev_clusters
===============

Python package for Exact Diagonalization of the Kitaev Model
-------------------------------------------------------------

.. image:: https://readthedocs.org/projects/kitaev-clusters/badge/?version=latest
    :alt: Documentation 
    :target: https://kitaev-clusters.readthedocs.io/en/latest/?badge=latest


.. image:: https://github.com/owenpb/kitaev_clusters/actions/workflows/pytest.yml/badge.svg
    :alt: Tests
    :target: https://github.com/owenpb/kitaev_clusters/actions

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :alt: Style
   :target: https://github.com/psf/black


Brief Overview
-----------------

**kitaev_clusters** is a Python package designed for efficient Exact Diagonalization of the spin-1 Kitaev honeycomb model on finite-size clusters.
This package contains a variety of functions which exploit the symmetries of the honeycomb lattice to effectively reduce the dimension of the Hilbert space.
Since the components of the Hamiltonian are typically sparse matrices, we provide functions for their efficient construction in a List of Lists (LIL) format, with conversion to a Compressed Sparse Row (CSR) format (if desired) for reduced memory needs.
The package also contains a number of functions to calculate the ground state, energy, and bipartite entanglement entropy.
Adding single-ion anisotropy to the model is also straightforward, with functions available for measuring of the local moment in the [1, 1, 1] direction.

This package was used to calculate the properties of the spin-1 Kitaev model with single ion anisotropy, on lattices with up to 18 sites, in our recent publication:
`O. Bradley and R. R. P. Singh, Phys. Rev. B 105, L060405 (2022) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.105.L060405>`_