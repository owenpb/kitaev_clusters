import numpy as np
import scipy
from scipy import sparse
import scipy.sparse.linalg
from scipy.sparse import lil_matrix

from symmetry_functions import ternary, ternary_pad, tern_to_base10


def SmSm(state, j, k):

    # state: an L-digit ternary string e.g. "111100002222102100"
    # j and k: sites to lower
    # returns string with sites j and k lower
    # NOTE: lowering the z component of spin = increasing the saved number from 0 (up) to 1 (down)

    state_rev = state[::-1]  # Reverse the string to get sites in order: 0, 1, 2, 3, 4, ...

    list_sites = list(state_rev)  # Turn string into a list, e.g. [0, 0, 1, 2, 0, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1]

    list_sites[j] = str(int(list_sites[j]) + 1)
    list_sites[k] = str(int(list_sites[k]) + 1)

    list_sites = ''.join(list_sites)

    new_state = list_sites[::-1]

    return new_state


def SmSp(state, j, k):

    # a: an L-digit ternary string e.g. "111100002222102100"
    # j and k: sites to lower and raise respectively
    # returns string with site j lowered and k raised
    # NOTE: lowering the z component of spin = increasing the saved number from 0 (up) to 1 (down)

    state_rev = state[::-1] # Reverse the string to get sites in order: 0, 1, 2, 3, 4, ...

    list_sites = list(state_rev) # Turn string into a list, e.g. [0, 0, 1, 2, 0, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1]

    list_sites[j] = str(int(list_sites[j]) + 1)
    list_sites[k] = str(int(list_sites[k]) - 1)

    list_sites = ''.join(list_sites)

    new_state = list_sites[::-1]

    return new_state


def SpSm(state, j, k):

    # state: an L-digit ternary string e.g. "111100002222102100"
    # j and k: sites to raise and lower respectively
    # returns string with sites j raised and k lowered
    # NOTE: lowering the z component of spin = increasing the saved number from 0 (up) to 1 (down)

    state_rev = state[::-1]  # Reverse the string to get sites in order: 0, 1, 2, 3, 4, ...

    list_sites = list(state_rev)  # Turn string into a list, e.g. [0, 0, 1, 2, 0, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1]

    list_sites[j] = str(int(list_sites[j]) - 1)
    list_sites[k] = str(int(list_sites[k]) + 1)

    list_sites = ''.join(list_sites)

    new_state = list_sites[::-1]

    return new_state


def SpSp(state, j, k):

    # state: an L-digit ternary string e.g. "111100002222102100"
    # j and k: sites to raise
    # returns string with sites j and k raised
    # NOTE: lowering the z component of spin = increasing the saved number from 0 (up) to 1 (down)

    state_rev = state[::-1] # Reverse the string to get sites in order: 0, 1, 2, 3, 4, ...

    list_sites = list(state_rev)  # Turn string into a list, e.g. [0, 0, 1, 2, 0, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1]

    list_sites[j] = str(int(list_sites[j]) - 1)
    list_sites[k] = str(int(list_sites[k]) - 1)

    list_sites = ''.join(list_sites)

    new_state = list_sites[::-1]

    return new_state


def SzSz(state, j, k):

    # state: an L-digit ternary string e.g. "111100002222102100"
    # j and k: sites upon which Sz_j and Sz_k operators act on
    # returns coefficient which multiplies the resulting (identical) state

    # E.g. for spin-1 system: Sz_5 Sz_6 |10211111> = (-1)(1) |10211111>, so coeff = -1 x 1 = -1.
    # (Note "0" corresponds to a spin z-component m_s=1, "1" has m_s=0, and "2" has m_s=-1)

    state_rev = state[::-1]  # Reverse the string to get sites in order: 0, 1, 2, 3, 4, ...

    sj = int(state_rev[j])
    sk = int(state_rev[k])

    ms_values = [1, 0, -1]  # Spin z-components (possible m_s values) for spin-1

    coeff_j = ms_values[sj]
    coeff_k = ms_values[sk]

    coeff = coeff_j*coeff_k

    return coeff


def Ss(state, i):

    # state: an L-digit ternary string e.g. "111100002222102100"
    # i: site at which Ss operator is applied

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
