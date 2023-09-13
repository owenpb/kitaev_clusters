import numpy as np
import scipy
from scipy import sparse
import scipy.sparse.linalg

from symmetry_functions import ternary, ternary_pad, tern_to_base10


def ground_state(H, L, k=1, ncv=20):

    E, V = scipy.sparse.linalg.eigsh(H, k=k, which="SA", return_eigenvectors=True, ncv=ncv)

    # Get ground state energy per site (i.e. divide by L) and corresponding eigenvector
    E_0 = E[0]/L
    psi_0 = V[:, 0]

    print('Ground state energy (per site): ', E_0)

    return E_0, psi_0


def s_squared(HD, L, psi_0):

    Ss = np.transpose(np.conjugate(psi_0)) @ HD @ psi_0

    Ss = np.real(Ss) / L  # Value per site

    return Ss


def entanglement_entropy(L, psi, kept_ints, state_map, n_unique_list):

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
