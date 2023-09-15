Tutorial: 12-site Kitaev cluster
================================

Here we will work through an example of using the library to calculate the ground state properties of the spin-1 Kitaev Hamiltonian with single-ion anisotropy on a lattice with :math:`L=12`. sites
The full Hamiltonian for this model is given by:

.. math::
    \hat{H} = K_x \sum_{\langle i, j \rangle} S_i^x S_j^x \, + \, K_y \sum_{(i, k)} S_i^y S_k^y \, + \,  K_z \sum_{[i, l]} S_i^z S_l^z + D \sum_i (S_i^x + S_i^y + S_i^z)^2

Here :math:`K_x`, :math:`K_y`, and :math:`K_z` are the Kitaev couplings along the :math:`x`, :math:`y`, and :math:`z` bond directions of the honeycomb lattice. The Hamiltonian
includes a term describing single-ion anisotropy in the :math:`[1, 1, 1]` direction, with magnitude :math:`D`, which preserves the symmetry between :math:`x`, :math:`y`, and :math:`z` bonds.
For the spin-1 case, the Hilbert space has dimension :math:`3^L`, where :math:`L` is the total number of sites. However, using lattice symmetries we can greatly reduce the dimensions
of the Hamiltonian (by a factor of :math:`L`), and through efficient construction and storage of the sparse matrices :math:`H_{x}`, :math:`H_{y}`, :math:`H_{z}`, and  :math:`H_{D}`,
it becomes feasible to study clusters with :math:`L \geq 18` sites.

1. Importing modules
--------------------

Let's start by importing the modules we will need. After specifying the size of our lattice, we will be using the symmetry functions
``get_neighbors``, ``lattice_translations`` and ``get_representative_states``. These find all nearest-neighbor pairs, the configurations
of the lattice which are equivalent via symmetry, and the set of representative states we need to keep (which greatly reduces the size
of the Hilbert space and therefore the dimensions of our Hamiltonian matrix).

Then we will use a set of functions to efficiently construct and store the components of the Hamiltonian: ``construct_Hxx``, ``construct_Hyy``, ``construct_Hzz``, ``construct_HD``,
and finally ``construct_hamiltonian`` to reproduce the full Hamiltonian. Then we will obtain the ground state and calculate its energy per site :math:`E_0`,
bipartite entanglement entropy, and the expectation value of the squared spin operator :math:`(S^x + S^y + S^z)^2`. For this we will need to import the functions
``ground_state``, ``s_squared``, and ``entanglement_entropy``:

.. code-block::
    import numpy as np
    import scipy
    import gc

    from kitaev_clusters.symmetry_functions import get_neighbors, lattice_translations, get_representative_states
    from kitaev_clusters.hamiltonian_functions import construct_Hxx, construct_Hyy, construct_Hzz, construct_HD, construct_hamiltonian
    from kitaev_clusters.ground_state_functions import ground_state, s_squared, entanglement_entropy


2. Defining lattice parameters
------------------------------

Each unit cell of the honeycomb lattice consists of two sites. For a lattice with :math:`L=12` sites, we can choose a geometry with :math:`L_x=3` unit cells
in the x-direction, and :math:`L_y=2` unit cells in the y-direction. Then we will call the function ``get_neighbors`` to get a list of site pairs along each
of the three bond directions of the honeycomb lattice. After this, we call the function ``translation_order_list`` to get a list of all permutations of :math:`L`
sites which are equivalent via honeycomb lattice symmetries:

.. code-block::
    L = 12 # Total number of lattice sites (L = Lx * Ly * 2)
    Lx = 3 # Unit cells in horizontal direction
    Ly = 2 # Unit cells in vertical direction

    x_neighbors = get_neighbors(Lx, Ly, bond='x')
    y_neighbors = get_neighbors(Lx, Ly, bond='y')
    z_neighbors = get_neighbors(Lx, Ly, bond='z')


    translation_order_list = lattice_translations(Lx, Ly)


3. Finding representative states
--------------------------------

Here we will find the representative states we need to keep using the function ``get_representative_states``, and construct and store the following arrays.
We find ``kept_ints``, an array of the base-10 integers corresponding to the representative states. We also get ``state_map``, an array containing the representative
state (in base-10) for each of the :math:`3^L` possible states, i.e. a mapping from a state to its representative state. Each representative state has a number of states
equivalent to it by symmetry, however only a certain number of these will be unique. We store ``n_unique_list``, which is an array containing the number of unique
mirror states for each representative state. Finally, the function also returns ``ndim``, which is the total number of representative states we keep, i.e. the dimension
of the reduced Hilbert space.

.. code-block::
    kept_ints, state_map, n_unique_list, ndim = get_representative_states(L, translation_order_list,
                                                                      ints_filename='kept_ints_L12',
                                                                      states_filename='state_map_L12',
                                                                      nunique_filename='n_unique_list_L12')


4. Loading representative states
--------------------------------

This step is only necessary if you want to load from file the arrays of representative states, state map, and number of unique mirror states. However, for a given
lattice size, these arrays will be fixed and therefore only need to be calculated once. It is therefore recommended to load these arrays from file. Note that the
function ``get_representative_states`` will save these arrays in .npz format:

.. code-block::
    kept_ints = np.load('kept_ints_L12.npy')
    state_map = np.load('state_map_L12.npy')
    n_unique_list = np.load('n_unique_list_L12.npy')


5. Constructing components of Hamiltonian
-----------------------------------------

Here we pass our previously loaded arrays to the functions ``construct_Hxx``, ``construct_Hyy``, ``construct_Hzz``, and ``construct_HD`` to efficiently construct
the components of the Hamiltonian as sparse matrices in List of List (LIL) format. These matrices are then converted to Compressed Sparse Row (CSR) format
for reduced memory requirements (although this can be disabled by setting ``as_csr=False``), and saved in .npz format. We can specify the filename for each
matrix to be saved.

.. code-block::
    Hxx = construct_Hxx(L, x_neighbors, kept_ints, state_map, n_unique_list, filename='Hxx_L12', as_csr=True)
    Hyy = construct_Hyy(L, y_neighbors, kept_ints, state_map, n_unique_list, filename='Hyy_L12', as_csr=True)
    Hzz = construct_Hzz(L, z_neighbors, kept_ints, filename='Hzz_L12', as_csr=True)
    HD = construct_HD(L, kept_ints, state_map, n_unique_list, filename='HD_L12', as_csr=True)


6. Loading components of Hamiltonian
------------------------------------

For a given lattice size, the matrices :math:`H_x`, :math:`H_y`, :math:`H_z`, and :math:`H_D` are fixed. It is therefore recommended to only construct them
once then load then file when needed. Since these are sparse matrices saved in .npz format, we can load them as follows:

.. code-block::
    Hxx = scipy.sparse.load_npz('Hxx_L12.npz')
    Hyy = scipy.sparse.load_npz('Hyy_L12.npz')
    Hzz = scipy.sparse.load_npz('Hzz_L12.npz')
    HD = scipy.sparse.load_npz('HD_L12.npz')

However, if it is not necessary to load each component of the Hamiltonian separately if you just want to use them to construct the full Hamiltonian. That
is performed using the function ``construct_hamiltonian`` as shown in the next step.

7. Specifying couplings, anisotropy, and constructing full Hamiltonian
----------------------------------------------------------------------

Now we can specify values for the Kitaev couplings :math:`K_x`, :math:`K_y`, and :math:`K_z`, as well as the anisotropy parameter :math:`D`:

.. code-block::
    Kx = 1  # Kitaev coupling along x-bonds
    Ky = 1  # Kitaev coupling along y-bonds
    Kz = 1  # Kitaev coupling along z-bonds
    D = 0.0   # [1, 1, 1] single-ion anisotropy

The function ``construct_hamiltonian`` can now be called, passing these parameters as arguments, to return the full Hamiltonian matrix :math:`H`.
This function loads each of the components of the Hamiltonian from file, with the filenames of these .npz files also passed as arguments:

H = construct_hamiltonian(Kx, Ky, Kz, D, 'Hxx_L12.npz', 'Hyy_L12.npz', 'Hzz_L12.npz', 'HD_L12.npz')

However, if you are not loading the components of the Hamiltonian from file, for example if you already have constructed the matrices :math:`H_x`,
:math:`H_y`, :math:`H_z`, and :math:`H_D` in your workspace or notebook, simply create the full Hamiltonian matrix as follows:

.. code-block::
    H = (Kx * Hxx) + (Ky * Hyy) + (Kz * Hzz) + (D * HD)


8. Finding ground sate and physical properties
----------------------------------------------

Now we can pass the full Hamiltonian matrix :math:`H` as an argument to the function ``ground_state`` to obtain the ground state ``psi_0`` and
the ground state energy per site ``E_0``. We can also pass the ground state ``psi_0`` as an argument to the functions ``s_squared`` and
``entanglement_entropy`` to get the expectation value of :math:`(S^x + S^y + S^z)^2`, and the bipartite entanglement entropy:

.. code-block::
    E_0, psi_0 = ground_state(H, L)
    Ss = s_squared(HD, L, psi_0)
    entropy = entanglement_entropy(L, psi_0, kept_ints, state_map, n_unique_list)

Saving these results to a .txt file is also straightforward:

.. code-block::
    f = open('results.txt', 'w')
    f.write('D_111, E_gs per site, <(Sx + Sy + Sz)^2>, entropy \n')
    f.write(f'{D}, {E_0}, {Ss}, {entropy} \n')
    f.close()

If desired, you may want to saved the ground state ``psi_0`` after calling the function ``ground_state``, and later load it when calculating
the entanglement entropy or :math:`\langle(S^x + S^y + S^z)^2\rangle`. This can be performed with the commands:

.. code-block::
    # Saving ground state as .npy file:
    np.save('psi_0', psi_0)

    # Loading ground state:
    np.load('psi_0.npy')


8. Freeing memory after calculations complete
---------------------------------------------

If desired, you may want to free up memory after your calculations are complete. If you no longer need to store the matrices you have loaded or
the ground state, this can be done as follows:

.. code-block::
    del Hxx
    del Hyy
    del Hzz
    del HD
    del H
    del psi_0
    gc.collect()