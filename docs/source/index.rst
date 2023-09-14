.. ExamplePackage documentation master file, created by
   sphinx-quickstart on Sat Jun 18 02:48:40 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

kitaev_clusters Documentation
==========================================

**kitaev_clusters** is a Python package designed for efficient Exact Diagonalization of the spin-1 Kitaev honeycomb model on finite-size clusters.
This package contains a variety of functions which exploit the symmetries of the honeycomb lattice to effectively reduce the dimension of the Hilbert space.
Since the components of the Hamiltonian are typically sparse matrices, we provide functions for their efficient construction in a List of Lists (LIL) format, with conversion to a Compressed Sparse Row (CSR) format (if desired) for reduced memory needs.
The package also contains a number of functions to calculate the ground state, energy, and bipartite entanglement entropy.
Adding single-ion anisotropy to the model is also straightforward, with functions available for measuring of the local moment in the [1, 1, 1] direction.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api_doc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
