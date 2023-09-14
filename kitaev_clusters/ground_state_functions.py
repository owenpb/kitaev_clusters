import numpy as np
import scipy
from scipy import sparse
import scipy.sparse.linalg

from kitaev_clusters.symmetry_functions import ternary, ternary_pad, tern_to_base10


def ground_state(H, L, k=1, ncv=20):

    """
    Finds the ground state psi_0 and the ground state energy E_0, by applying Lanczos algorithm to matrix Hamiltonian H.

    Parameters
    ----------
    H : csr_matrix or lil_matrix
        The full sparse matrix Hamiltonian.
    L : int
        The total number of lattice sites. Used to calculate the energy per site.
    k : int, optional
        Number of eigenvalues and eigenvectors desired. By default k=1 to find state only.
    ncv : int, optional
        Number of Lanczos vectors generated.


    Returns
    -------
    E_0 : float
        The ground state energy per site.
    psi_0 : ndarray
        The ground state, i.e. a 1d array with ndim complex coefficients.

    """

    E, V = scipy.sparse.linalg.eigsh(H, k=k, which="SA", return_eigenvectors=True, ncv=ncv)

    # Get ground state energy per site (i.e. divide by L) and corresponding eigenvector
    E_0 = E[0]/L
    psi_0 = V[:, 0]

    print('Ground state energy (per site): ', E_0)

    return E_0, psi_0


def s_squared(HD, L, psi_0):

    """
    Finds the expectation value of (Sx + Sy + Sz)^2 in the ground state (per site).

    Parameters
    ----------
    HD : csr_matrix or lil_matrix
        The sparse matrix HD corresponding to the single-ion anisotropy term in the Hamiltonian.
    L : int
        The total number of lattice sites. Used to calculate <(Sx + Sy + Sz)^2> per site.
    psi_0 : ndarray
        The ground state, i.e. a 1d array with ndim complex coefficients.


    Returns
    -------
    Ss: float
        The expectation value of (Sx + Sy + Sz)^2 in the ground state (per site).

    """

    Ss = np.transpose(np.conjugate(psi_0)) @ HD @ psi_0

    Ss = np.real(Ss) / L  # Value per site

    return Ss


def entanglement_entropy(L, psi, state_map, n_unique_list):

    """
    Finds the bipartite entanglement entropy when the system is divided into two equal halves. Performs a
    singular value decomposition (SVD) to obtain the singular values.

    Parameters
    ----------
    L : int
        The total number of lattice sites. Used to calculate <(Sx + Sy + Sz)^2> per site.
    psi : ndarray
        The state of the system, i.e. a 1d array with ndim complex coefficients. Typically the ground state psi_0.
    state_map : ndarray
        An array containing the representative state (in base-10) for each of the 3^L possible states. This is
        the second value returned by the function get_representative_states.
    n_unique_list : ndarray
        An array containing the number of unique mirror states for each representative state. This is the third value
        returned by the function get_representative_states.


    Returns
    -------
    entropy: float
        The bipartite entanglement entropy when the system is divided into two equal halves.

    """

    M_dim = 3**(L//2)

    M = np.zeros((M_dim, M_dim), dtype='complex')

    print(f'Constructing {M_dim}x{M_dim} matrix for SVD...\n')

    for i in range(M_dim):
        for j in range(M_dim):

            state_left = ternary_pad(i, L//2)
            state_right = ternary_pad(j, L//2)

            state = state_left + state_right

            state_int = tern_to_base10(state)

            rep_state_index = int(state_map[state_int])

            ci = psi[rep_state_index]
            ri = n_unique_list[rep_state_index]

            M[i][j] = ci / np.sqrt(ri)

    print('Performing SVD...\n')

    # Get array of singular values:
    S = np.linalg.svd(M, compute_uv=False)

    print('Calculating entanglement entropy...\n')

    entropy = 0
    for value in S:
        entropy += -1*(value**2)*(np.log(value**2))

    print('Entropy calculation complete \n')

    return entropy
