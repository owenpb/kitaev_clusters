import numpy as np


def tern_to_base10(n):

    """
    Converts a ternary string to a base-10 integer.

    Parameters
    ----------
    n : str
        A string representing a number in ternary, e.g. '1002'.


    Returns
    -------
    tot : int
        The base-10 integer corresponding to the ternary string, e.g. 29 for '1002'

    """

    rev = n[::-1]
    tot = 0

    for i in range(len(rev)):
        tot = tot + (int(rev[i])*(3**i))

    return tot


def ternary(n):

    if n == 0:
        return '0'

    nums = []

    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))

    return ''.join(reversed(nums))


def ternary_pad(n, L):

    diff = L - len(ternary(n))
    pad = '0'*diff

    return pad+ternary(n)


def mirror_states(state, translation_order_list):

    L = len(state)

    rev_state = state[::-1]

    state_translations = [state]
    state_translation_ints = [tern_to_base10(state)]

    for ordering in translation_order_list:
        translated_state = ''.join(rev_state[i] for i in ordering)
        state_translations.append(translated_state[::-1])
        state_translation_ints.append(tern_to_base10(translated_state[::-1]))

    state_inversions = []
    state_inversion_ints = []

    for state in state_translations:
        state_inversions.append(state[::-1])
        state_inversion_ints.append(tern_to_base10(state[::-1]))

    mirror_ints = state_translation_ints + state_inversion_ints
    sorted_ints = sorted(mirror_ints)
    n_unique = len(np.unique(sorted_ints))

    rep_int = sorted_ints[0]
    rep_state = ternary_pad(rep_int, L)

    return sorted_ints, n_unique, rep_state, rep_int


def get_representative_states(L, translation_order_list, ints_filename='kept_ints', states_filename='state_map', nunique_filename='n_unique_list'):

    kept_ints = []
    n_unique_list = []
    state_map = -1*np.ones(3**L, dtype=int)
    index = -1

    for i in range(3**L):

        if i % 500000 == 0:
            print(f'Finding representative for state: {i}')

        if state_map[i] != -1:
            continue

        index += 1

        kept_ints.append(i)

        mirror_ints, n_unique, rep_state, rep_int = mirror_states(ternary_pad(i, L), translation_order_list)

        for j in mirror_ints:

            state_map[j] = index

        n_unique_list.append(n_unique)

    ndim = len(kept_ints)

    kept_ints = np.asarray(kept_ints, dtype=int)
    n_unique_list = np.asarray(n_unique_list, dtype=int)

    np.save(ints_filename, kept_ints)
    np.save(states_filename, state_map)
    np.save(nunique_filename, n_unique_list)

    return kept_ints, state_map, n_unique_list, ndim


def lattice_translations(Lx, Ly):

    # Lx: number of unit cells in x-direction
    # Ly: number of unit cells in y-direction

    # Total number of sites in lattice: L = Lx * Ly * 2

    unit_cells = np.zeros(shape=(Lx, Ly, 2), dtype=int)

    translation_order_list = []

    for x in range(Lx):
        for y in range(Ly):

            left_site = (x*(2*Ly)) + y
            right_site = left_site + Ly

            unit_cells[x][y] = [left_site, right_site]

    for j in range(Ly):
        for i in range(Lx):

            shifted_lattice = np.roll(unit_cells, shift=(i,j), axis=(0,1))

            site_sequence = []

            for x1 in range(Lx):
                for y1 in range(Ly):
                    site_sequence.append(shifted_lattice[x1][y1][0])

                for y2 in range(Ly):
                    site_sequence.append(shifted_lattice[x1][y2][1])

            translation_order_list.append(site_sequence)

    # Return all non-zero translations of lattice

    return translation_order_list[1:]


def get_neighbors(Lx, Ly, bond):

    # Lx: number of unit cells in x-direction
    # Ly: number of unit cells in y-direction
    # Total number of sites in lattice: L = Lx * Ly * 2

    # bond: 'x', 'y', or 'z'

    unit_cells = np.zeros(shape=(Lx, Ly, 2), dtype=int)

    for x in range(Lx):
        for y in range(Ly):

            left_site = (x*(2*Ly)) + y
            right_site = left_site + Ly

            unit_cells[x][y] = [left_site, right_site]

    neighbors = []

    if bond == 'x':

        for i in range(Lx):
            for j in range(Ly):

                x_bond_left = unit_cells[i][j][0]
                x_bond_right = unit_cells[i][j][1]
                x_pair = tuple(sorted((x_bond_left, x_bond_right)))
                neighbors.append(x_pair)

    if bond == 'y':

        for i in range(Lx):
            for j in range(Ly):

                y_bond_left = unit_cells[i][j][1]
                y_bond_right = unit_cells[(i+1)%Lx][j][0]
                y_pair = tuple(sorted((y_bond_left, y_bond_right)))
                neighbors.append(y_pair)

    if bond == 'z':

        for i in range(Lx):
            for j in range(Ly):

                z_bond_right = unit_cells[i][j][1]
                z_bond_left = unit_cells[i][(j+1)%Ly][0]
                z_pair = tuple(sorted((z_bond_left, z_bond_right)))
                neighbors.append(z_pair)

    return neighbors
