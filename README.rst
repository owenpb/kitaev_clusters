
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
`O. Bradley and R. R. P. Singh, Phys. Rev. B 105, L060405 (2022) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.105.L060405>`_ `(arXiv) <https://arxiv.org/abs/2108.05040>`_

Package Structure
-----------------
* **kitaev_clusters** contains the main code files.
* **scripts** contains example scripts showing how the library can be used.
* **docs** contains documentation files.
* **tests** contains units tests of functions using the pytest framework.

Requirements
-----------------
The code has been checked to run with the following versions:

* Python 3.10
* NumPy 1.22.3
* SciPy 1.7.3

Required libraries can be installed with the following command (after downloading the `requirements.txt <https://github.com/owenpb/kitaev_clusters/blob/main/requirements.txt>`_ file from this repository):

``pip install -r requirements.txt``

Installation
-----------------
The package can be installed from source via git with the following command:

``git clone https://github.com/owenpb/kitaev_clusters``

Units tests can be run using pytest after installation. Install pytest if needed using ``pip install pytest`` and execute the command ``pytest`` in the project directory to run all unit tests.

Documentation
-----------------
To get started with using the package, see the latest version of the documentation here:

`kitaev-clusters.readthedocs.io/en/latest/ <https://kitaev-clusters.readthedocs.io/en/latest/>`_

The documentation includes a tutorial showing how to use the library to find the ground state properties of a spin-1 cluster with 12 sites. There is also full API documentation available for each module.
