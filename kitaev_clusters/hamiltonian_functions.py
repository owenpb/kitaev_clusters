import numpy as np
import scipy
from scipy import sparse
import scipy.sparse.linalg
from scipy.sparse import lil_matrix

from kitaev_clusters.symmetry_functions import ternary, ternary_pad, tern_to_base10


def SmSm(state, j, k):

    """
    Applies spin lowering operator S- to sites j and k, i.e. acts S-(j) S-(k) on a state.

    Parameters
    ----------
    state : str
        A string representing a state in ternary.
    j : int
        An integer in the range [0, L-1] (where L is the total number of lattice sites) denoting a site j to lower.
    k : int
        An integer in the range [0, L-1] (where L is the total number of lattice sites) denoting a site k to lower.


    Returns
    -------
    new_state : str
        The ternary representation of the new state after the spin lowering operator S- has acted on sites j and k.

    """

    # Note: lowering the z-component of a spin = increasing the ternary number from e.g. 0 (ms=+1) to 1 (ms=0),
    # or from 1 (ms=0) to 2 (ms=-1).

    state_rev = state[::-1]

    list_sites = list(state_rev)

    list_sites[j] = str(int(list_sites[j]) + 1)
    list_sites[k] = str(int(list_sites[k]) + 1)

    list_sites = ''.join(list_sites)

    new_state = list_sites[::-1]

    return new_state


def SmSp(state, j, k):

    """
    Applies spin lowering operator S- to site j, and spin raising operator S+ to site k, i.e. acts S-(j) S+(k) on a state.

    Parameters
    ----------
    state : str
        A string representing a state in ternary.
    j : int
        An integer in the range [0, L-1] (where L is the total number of lattice sites) denoting a site j to lower.
    k : int
        An integer in the range [0, L-1] (where L is the total number of lattice sites) denoting a site k to raise.


    Returns
    -------
    new_state : str
        The ternary representation of the new state after the spin lowering operator S- has acted on sites j, and the spin raising operator S+ has acted on site k.

    """

    # Note: lowering the z-component of a spin = increasing the ternary number from e.g. 0 (ms=+1) to 1 (ms=0),
    # or from 1 (ms=0) to 2 (ms=-1).

    state_rev = state[::-1]

    list_sites = list(state_rev)

    list_sites[j] = str(int(list_sites[j]) + 1)
    list_sites[k] = str(int(list_sites[k]) - 1)

    list_sites = ''.join(list_sites)

    new_state = list_sites[::-1]

    return new_state


def SpSm(state, j, k):

    """
    Applies spin raising operator S+ to site j, and spin lowering operator S- to site k, i.e. acts S+(j) S-(k) on a state.

    Parameters
    ----------
    state : str
        A string representing a state in ternary.
    j : int
        An integer in the range [0, L-1] (where L is the total number of lattice sites) denoting a site j to raise.
    k : int
        An integer in the range [0, L-1] (where L is the total number of lattice sites) denoting a site k to lower.


    Returns
    -------
    new_state : str
        The ternary representation of the new state after the spin raising operator S+ has acted on site j, and the spin lowering operator S- has acted on site k.

    """

    # Note: lowering the z-component of a spin = increasing the ternary number from e.g. 0 (ms=+1) to 1 (ms=0),
    # or from 1 (ms=0) to 2 (ms=-1).

    state_rev = state[::-1]

    list_sites = list(state_rev)

    list_sites[j] = str(int(list_sites[j]) - 1)
    list_sites[k] = str(int(list_sites[k]) + 1)

    list_sites = ''.join(list_sites)

    new_state = list_sites[::-1]

    return new_state


def SpSp(state, j, k):

    """
    Applies spin raising operator S+ to sites j and k, i.e. acts S+(j) S+(k) on a state.

    Parameters
    ----------
    state : str
        A string representing a state in ternary.
    j : int
        An integer in the range [0, L-1] (where L is the total number of lattice sites) denoting a site j to raise.
    k : int
        An integer in the range [0, L-1] (where L is the total number of lattice sites) denoting a site k to raise.


    Returns
    -------
    new_state : str
        The ternary representation of the new state after the spin raising operator S+ has acted on sites j and k.

    """

    # Note: raising the z-component of a spin = decreasing the ternary number from e.g. 2 (ms=-1) to 1 (ms=0),
    # or from 1 (ms=0) to 0 (ms=1).

    state_rev = state[::-1]

    list_sites = list(state_rev)

    list_sites[j] = str(int(list_sites[j]) - 1)
    list_sites[k] = str(int(list_sites[k]) - 1)

    list_sites = ''.join(list_sites)

    new_state = list_sites[::-1]

    return new_state


def SzSz(state, j, k):

    """
    Applies Sz operator to sites j and k, and returns the coefficient multiplying the resultant (identical) state.

    Parameters
    ----------
    state : str
        A string representing a state in ternary.
    j : int
        An integer in the range [0, L-1] (where L is the total number of lattice sites) denoting a site j where we apply the Sz operator.
    k : int
        An integer in the range [0, L-1] (where L is the total number of lattice sites) denoting a site k where we apply the Sz operator.


    Returns
    -------
    new_state : str
        The ternary representation of the new state after the Sz operator has acted on sites j and k.

    """

    state_rev = state[::-1]

    sj = int(state_rev[j])
    sk = int(state_rev[k])

    ms_values = [1, 0, -1]  # Spin z-components (possible m_s values) for spin-1

    coeff_j = ms_values[sj]
    coeff_k = ms_values[sk]

    coeff = coeff_j*coeff_k

    return coeff


def Ss(state, i):

    """
    Applies squared spin operator (Sx + Sy + Sz)^2 to a site j, and returns the resultant states and their coefficients.

    Parameters
    ----------
    state : str
        A string representing a state in ternary.
    j : int
        An integer in the range [0, L-1] (where L is the total number of lattice sites) denoting a site j where we apply the operator (Sx + Sy + Sz)^2.


    Returns
    -------
    list
        A list of tuples of the form (state, coefficient), which result from applying the operator (Sx + Sy + Sz)^2 to our initial state at site j.

    """

    rev_state = state[::-1]     # Reverse string so lattice site 0 at beginning
    list_revstate = list(rev_state)

    spin_val = rev_state[i]
    insq2 = 1/(np.sqrt(2))

    bprime1 = list_revstate.copy()
    bprime2 = list_revstate.copy()
    bprime3 = list_revstate.copy()

    bprime1_options = ['1', '0', '0']
    bprime2_options = ['2', '2', '1']

    coeff1_options = [insq2*(1 + 1.0j), insq2*(1 - 1.0j), -1.0j]
    coeff2_options = [1.0j, insq2*(-1 - 1.0j), insq2*(-1 + 1.0j)]

    bprime1[i] = bprime1_options[int(spin_val)]
    bprime2[i] = bprime2_options[int(spin_val)]

    coeff1 = coeff1_options[int(spin_val)]
    coeff2 = coeff2_options[int(spin_val)]

    # Rejoin lists to strings, and reverse to get in proper ternary form

    bprime1 = ''.join(bprime1)
    bprime1 = bprime1[::-1]

    bprime2 = ''.join(bprime2)
    bprime2 = bprime2[::-1]

    bprime3 = ''.join(bprime3)
    bprime3 = bprime3[::-1]

    coeff3 = 2

    return [(bprime1, coeff1), (bprime2, coeff2), (bprime3, coeff3)]


def construct_Hxx(L, x_neighbors, kept_ints, state_map, n_unique_list, filename, as_csr=True):

    """
    Constructs Hxx, i.e. the component of the Kitaev Hamiltonian which sums over all x-direction bonds of the
    honeycomb lattice.

    Parameters
    ----------
    L : int
        The total number of lattice sites.
    x_neighbors : list
        A list of tuples of site pairs which each constitute an x-bond. This is returned by the function get_neighbors
        with the parameter bond="x".
    kept_ints : ndarray
        An array of the base-10 integers corresponding to the representative states. This is the first value returned by
        the function get_representative_states.
    state_map : ndarray
        An array containing the representative state (in base-10) for each of the 3^L possible states. This is
        the second value returned by the function get_representative_states.
    n_unique_list : ndarray
        An array containing the number of unique mirror states for each representative state. This is the third value
        returned by the function get_representative_states.
    filename : str
        Filename for the Hxx matrix which will be saved as a .npz file.
    as_csr : bool
        If True (default), the Hxx sparse matrix is converted from List of Lists (LIL) format to Compressed Sparse Row
        (CSR) format before saving.


    Returns
    -------
    Hxx : csr_matrix
        The matrix Hxx in Compressed Sparse Row (CSR) format, i.e. the component of the Kitaev Hamiltonian which sums
        over all x-direction bonds of the honeycomb lattice. It will have dimensions ndim x ndim, where ndim is the
        number of representative states.

    """

    ndim = kept_ints.size

    Hxx = sparse.lil_matrix((ndim, ndim))

    print('Constructing Hxx...\n')

    for i in range(ndim):

        a = ternary_pad(kept_ints[i], L)

        for nx in x_neighbors:

            j = nx[0]
            k = nx[1]

            bprime1 = SpSp(a, j, k)
            if ('3' not in bprime1) & ('-' not in bprime1):

                row = int(state_map[tern_to_base10(bprime1)])
                col = i

                Ra = n_unique_list[i]
                Rb = n_unique_list[row]

                Hxx[row, col] = Hxx[row, col] + 0.25*2*np.sqrt(Ra/Rb)

            bprime2 = SpSm(a, j, k)
            if ('3' not in bprime2) & ('-' not in bprime2):

                row = int(state_map[tern_to_base10(bprime2)])
                col = i

                Ra = n_unique_list[i]
                Rb = n_unique_list[row]

                Hxx[row, col] = Hxx[row, col] + 0.25*2*np.sqrt(Ra/Rb)

            bprime3 = SmSp(a, j, k)
            if ('3' not in bprime3) & ('-' not in bprime3):

                row = int(state_map[tern_to_base10(bprime3)])
                col = i

                Ra = n_unique_list[i]
                Rb = n_unique_list[row]

                Hxx[row, col] = Hxx[row, col] + 0.25*2*np.sqrt(Ra/Rb)

            bprime4 = SmSm(a, j, k)
            if ('3' not in bprime4) & ('-' not in bprime4):

                row = int(state_map[tern_to_base10(bprime4)])
                col = i

                Ra = n_unique_list[i]
                Rb = n_unique_list[row]

                Hxx[row, col] = Hxx[row, col] + 0.25*2*np.sqrt(Ra/Rb)

    if as_csr:
        print('Converting Hxx to csr format...\n')
        Hxx = Hxx.tocsr()

    print('Saving Hxx... \n')
    scipy.sparse.save_npz(filename, Hxx)

    return Hxx


def construct_Hyy(L, y_neighbors, kept_ints, state_map, n_unique_list, filename, as_csr=True):

    """
    Constructs Hyy, i.e. the component of the Kitaev Hamiltonian which sums over all y-direction bonds of the
    honeycomb lattice.

    Parameters
    ----------
    L : int
        The total number of lattice sites.
    y_neighbors : list
        A list of tuples of site pairs which each constitute a y-bond. This is returned by the function get_neighbors
        with the parameter bond="y".
    kept_ints : ndarray
        An array of the base-10 integers corresponding to the representative states. This is the first value returned by
        the function get_representative_states.
    state_map : ndarray
        An array containing the representative state (in base-10) for each of the 3^L possible states. This is
        the second value returned by the function get_representative_states.
    n_unique_list : ndarray
        An array containing the number of unique mirror states for each representative state. This is the third value
        returned by the function get_representative_states.
    filename : str
        Filename for the Hyy matrix which will be saved as a .npz file.
    as_csr : bool
        If True (default), the Hxx sparse matrix is converted from List of Lists (LIL) format to Compressed Sparse Row
        (CSR) format before saving.


    Returns
    -------
    Hyy : csr_matrix
        The matrix Hxx in Compressed Sparse Row (CSR) format, i.e. the component of the Kitaev Hamiltonian which sums
        over all y-direction bonds of the honeycomb lattice. It will have dimensions ndim x ndim, where ndim is the
        number of representative states.

    """

    ndim = kept_ints.size

    Hyy = sparse.lil_matrix((ndim, ndim))

    print('Constructing Hyy...\n')

    for i in range(ndim):

        a = ternary_pad(kept_ints[i], L)

        for ny in y_neighbors:

            j = ny[0]
            k = ny[1]

            bprime1 = SpSp(a, j, k)
            if ('3' not in bprime1) & ('-' not in bprime1):

                row = int(state_map[tern_to_base10(bprime1)])
                col = i

                Ra = n_unique_list[i]
                Rb = n_unique_list[row]

                Hyy[row, col] = Hyy[row, col] - 0.25*2*np.sqrt(Ra/Rb)

            bprime2 = SpSm(a, j, k)
            if ('3' not in bprime2) & ('-' not in bprime2):

                row = int(state_map[tern_to_base10(bprime2)])
                col = i

                Ra = n_unique_list[i]
                Rb = n_unique_list[row]

                Hyy[row, col] = Hyy[row, col] + 0.25*2*np.sqrt(Ra/Rb)

            bprime3 = SmSp(a, j, k)
            if ('3' not in bprime3) & ('-' not in bprime3):

                row = int(state_map[tern_to_base10(bprime3)])
                col = i

                Ra = n_unique_list[i]
                Rb = n_unique_list[row]

                Hyy[row, col] = Hyy[row, col] + 0.25*2*np.sqrt(Ra/Rb)

            bprime4 = SmSm(a, j, k)
            if ('3' not in bprime4) & ('-' not in bprime4):

                row = int(state_map[tern_to_base10(bprime4)])
                col = i

                Ra = n_unique_list[i]
                Rb = n_unique_list[row]

                Hyy[row, col] = Hyy[row, col] - 0.25*2*np.sqrt(Ra/Rb)

    if as_csr:
        print('Converting Hyy to csr format...\n')
        Hyy = Hyy.tocsr()

    print('Saving Hyy... \n')
    scipy.sparse.save_npz(filename, Hyy)

    return Hyy


def construct_Hzz(L, z_neighbors, kept_ints, filename, as_csr=True):

    """
    Constructs Hzz, i.e. the component of the Kitaev Hamiltonian which sums over all z-direction bonds of the
    honeycomb lattice.

    Parameters
    ----------
    L : int
        The total number of lattice sites.
    z_neighbors : list
        A list of tuples of site pairs which each constitute a z-bond. This is returned by the function get_neighbors
        with the parameter bond="z".
    kept_ints : ndarray
        An array of the base-10 integers corresponding to the representative states. This is the first value returned by
        the function get_representative_states.
    filename : str
        Filename for the Hzz matrix which will be saved as a .npz file.
    as_csr : bool
        If True (default), the Hzz sparse matrix is converted from List of Lists (LIL) format to Compressed Sparse Row
        (CSR) format before saving.


    Returns
    -------
    Hzz : csr_matrix
        The matrix Hxx in Compressed Sparse Row (CSR) format, i.e. the component of the Kitaev Hamiltonian which sums
        over all z-direction bonds of the honeycomb lattice. It will have dimensions ndim x ndim, where ndim is the
        number of representative states.

    """

    ndim = kept_ints.size

    Hzz = sparse.lil_matrix((ndim, ndim))

    print('Constructing Hzz...\n')

    for i in range(ndim):

        a = ternary_pad(kept_ints[i], L)

        for nz in z_neighbors:

            j = nz[0]
            k = nz[1]

            coeff = SzSz(a, j, k)

            Hzz[i, i] = Hzz[i, i] + coeff

    if as_csr:
        print('Converting Hzz to csr format...\n')
        Hzz = Hzz.tocsr()

    print('Saving Hzz... \n')
    scipy.sparse.save_npz(filename, Hzz)

    return Hzz


def construct_HD(L, kept_ints, state_map, n_unique_list, filename, as_csr=True):

    """
    Constructs HD, i.e. the component of the Hamiltonian which describes a single-ion anisotropy in the [1, 1, 1]
    direction. The HD operator involves a sum over all sites of D * (Sx + Sy + Sz)^2, where D is the magnitude of the
    single-ion anisotropy.

    Parameters
    ----------
    L : int
        The total number of lattice sites.
    kept_ints : ndarray
        An array of the base-10 integers corresponding to the representative states. This is the first value returned by
        the function get_representative_states.
    state_map : ndarray
        An array containing the representative state (in base-10) for each of the 3^L possible states. This is
        the second value returned by the function get_representative_states.
    n_unique_list : ndarray
        An array containing the number of unique mirror states for each representative state. This is the third value
        returned by the function get_representative_states.
    filename : str
        Filename for the HD matrix which will be saved as a .npz file.
    as_csr : bool
        If True (default), the HD sparse matrix is converted from List of Lists (LIL) format to Compressed Sparse Row
        (CSR) format before saving.


    Returns
    -------
    HD : csr_matrix
        The matrix HD in Compressed Sparse Row (CSR) format, i.e. the component of the Hamiltonian which describes a
        single-ion anisotropy in the [1, 1, 1] direction.

    """

    ndim = kept_ints.size

    HD = sparse.lil_matrix((ndim, ndim), dtype='complex64')

    print('Constructing HD...\n')

    for i in range(ndim):

        a = ternary_pad(kept_ints[i], L)

        for l in range(L):

            Ss_result = Ss(a, l)

            bprime1, coeff1 = Ss_result[0]

            col = i
            row = int(state_map[tern_to_base10(bprime1)])

            Ra = n_unique_list[i]
            Rb = n_unique_list[row]

            HD[row, col] = HD[row, col] + coeff1*np.sqrt(Ra/Rb)

            bprime2, coeff2 = Ss_result[1]

            row = int(state_map[tern_to_base10(bprime2)])

            Rb = n_unique_list[row]

            HD[row, col] = HD[row, col] + coeff2*np.sqrt(Ra/Rb)

            # bprime3 = a, so row = col, and coeff3 = 2

            HD[col, col] = HD[col, col] + 2

    if as_csr:
        print('Converting HD to csr format...\n')
        HD = HD.tocsr()

    print('Saving HD... \n')
    scipy.sparse.save_npz(filename, HD)

    return HD


def construct_hamiltonian(Kx, Ky, Kz, D, Hxx_file, Hyy_file, Hzz_file, HD_file):

    """
     Loads from file separate components of the Hamiltonian (Hxx, Hyy, Hzz, HD) and constructs the full Hamiltonian
     matrix.

     Parameters
     ----------
     Kx : float or int
         The Kitaev coupling along x-bonds.
     Ky : float or int
         The Kitaev coupling along y-bonds.
     Kz : float or int
         The Kitaev coupling along z-bonds.
     D : float or int
         The magnitude of the single-ion anisotropy term, i.e. the coefficient which multiplies HD.
     Hxx_file: .npz file
         Saved file (in .npz format) for the Hxx matrix.
     Hyy_file: .npz file
         Saved file (in .npz format) for the Hyy matrix.
     Hzz_file: .npz file
         Saved file (in .npz format) for the Hzz matrix.
     HD_file: .npz file
         Saved file (in .npz format) for the HD matrix.


     Returns
     -------
     H : csr_matrix
         The full sparse matrix Kitaev Hamiltonian corresponding to H = (Kx * Hxx) + (Ky * Hyy) + (Kz * Hzz) + (D * HD).

     """

    print('Loading Hxx... \n')
    Hxx = scipy.sparse.load_npz(Hxx_file)

    print('Loading Hyy... \n')
    Hyy = scipy.sparse.load_npz(Hyy_file)

    print('Loading Hzz... \n')
    Hzz = scipy.sparse.load_npz(Hzz_file)

    print('Loading HD... \n')
    HD = scipy.sparse.load_npz(HD_file)

    print('Constructing full Hamiltonian... \n')
    H = (Kx * Hxx) + (Ky * Hyy) + (Kz * Hzz) + (D * HD)

    print('Hamiltonian constructed \n')

    return H
