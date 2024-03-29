kitaev_clusters Documentation
=============================

Welcome to the documentation for the **kitaev_clusters** library. Here you can find:

* An overview of the library with reference paper
* Package structure, requirements, and installation guide
* Tutorial for spin-1 Kitaev model with single-ion anisotropy on a 12-site lattice
* API documentation of the **kitaev_clusters** library

Overview
--------------

**kitaev_clusters** is a Python package designed for efficient Exact Diagonalization of the spin-1 Kitaev honeycomb model on finite-size clusters.
This package contains a variety of functions which exploit the symmetries of the honeycomb lattice to effectively reduce the dimension of the Hilbert space.
Since the components of the Hamiltonian are typically sparse matrices, we provide functions for their efficient construction in a List of Lists (LIL) format, with conversion to a Compressed Sparse Row (CSR) format (if desired) for reduced memory needs.
The package also contains a number of functions to calculate the ground state, energy, and bipartite entanglement entropy.
Adding single-ion anisotropy to the model is also straightforward, with functions available for measuring of the local moment in the :math:`[1, 1, 1]` direction.

This package was used to calculate the properties of the spin-1 Kitaev model with single ion anisotropy, on lattices with up to 18 sites, in our recent publication
`O. Bradley and R. R. P. Singh, Phys. Rev. B 105, L060405 (2022) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.105.L060405>`_ `(arXiv link) <https://arxiv.org/abs/2108.05040>`_, which provides further discussion of the model.

Package Structure
-----------------
* **kitaev_clusters** contains the main code files. The main modules are:

 1. symmetry_functions: a collection of functions which use lattice symmetries to reduce the Hilbert space dimension.
 2. hamiltonian_functions: a collection of functions for efficient construction of the sparse matrix Hamiltonian.
 3. ground_state_functions: a collection functions for obtaining ground states and measuring physical quantities.
* **scripts** contains example scripts showing how the library can be used.
* **docs** contains documentation files.
* **tests** contains units tests of functions using the pytest framework.

Requirements
------------
The code has been checked to run with the following versions:

* Python 3.10
* NumPy 1.22.3
* SciPy 1.7.3

Required libraries can be installed with the following command (after downloading the `requirements.txt <https://github.com/owenpb/kitaev_clusters/blob/main/requirements.txt>`_ file from this repository):

``pip install -r requirements.txt``

Installation
------------
The package can be installed from source via git with the following command:

``git clone https://github.com/owenpb/kitaev_clusters``

Units tests can be run using pytest after installation. Install pytest if needed using ``pip install pytest`` and execute the command ``pytest`` in the project directory to run all unit tests.



Tutorial
========
Here we work through an example of using the library to calculate the ground state properties of the spin-1 Kitaev Hamiltonian with single-ion anisotropy on a 12-site lattice:

.. toctree::
   :maxdepth: 2

   tutorial




Module Documentation
====================
Here you can find API documentation for each function in the main modules:

.. toctree::
   :maxdepth: 2

   api_doc


Index and Search
==================

* :ref:`genindex`
* :ref:`search`
