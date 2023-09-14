import numpy as np


def tern_to_base10(state):

    """
    Converts a ternary string to a base-10 integer.

    Parameters
    ----------
    state : str
        A string representing a number in ternary.


    Returns
    -------
    integer_rep : int
        The base-10 integer corresponding to the ternary string.

    """

    rev = state[::-1]
    integer_rep = 0

    for i in range(len(rev)):
        integer_rep = integer_rep + (int(rev[i])*(3**i))

    return integer_rep


def ternary(n):

    """
    Converts a base-10 integer to a ternary string.

    Parameters
    ----------
    n : int
        A base-10 integer.


    Returns
    -------
    str
        A string representing an integer in ternary, e.g. "1002" for 29.

    """

    if n == 0:
        return '0'

    nums = []

    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))

    return ''.join(reversed(nums))


def ternary_pad(n, L):

    """
    Converts a base-10 integer to a ternary string of length L.
    Will left-pad ternary string with zeros such that the string has length L.

    Parameters
    ----------
    n : int
        A base-10 integer.
    L : int
        Desired total length of ternary string. The string will be left-padded with zeros to have length L.


    Returns
    -------
    str
        A string of length L representing an integer in ternary, e.g. "000000001002" for n=29, L=12.

    """

    diff = L - len(ternary(n))
    pad = '0'*diff

    return pad+ternary(n)


def lattice_translations(Lx, Ly):

    """
    For a specified honeycomb lattice (Lx unit cells in x-direction, Ly units cells in y-direction), returns a list of
    site orderings which are equivalent to it via translational symmetry (with periodic boundary conditions).
    For example, with Lx=3, Ly=2 there are 6 unit cells and L=12 total sites, and for any ordered labelling of sites,
    there are 5 equivalent orders which are generated by the translations (1,0), (2,0), (0,1), (1,1), and (2,1).

    Parameters
    ----------
    Lx : int
        Number of units cells in the x-direction.
    Ly : int
        Number of unit cells in the y-direction.


     Returns
    -------
    translation_order_list : list
        A list of all equivalent site orderings generated via non-zero translations (with periodic boundary conditions).

    """

    # Note: Total number of sites in lattice: L = Lx * Ly * 2

    unit_cells = np.zeros(shape=(Lx, Ly, 2), dtype=int)

    translation_order_list = []

    for x in range(Lx):
        for y in range(Ly):

            left_site = (x*(2*Ly)) + y
            right_site = left_site + Ly

            unit_cells[x][y] = [left_site, right_site]

    for j in range(Ly):
        for i in range(Lx):

            shifted_lattice = np.roll(unit_cells, shift=(i, j), axis=(0, 1))

            site_sequence = []

            for x1 in range(Lx):
                for y1 in range(Ly):
                    site_sequence.append(shifted_lattice[x1][y1][0])

                for y2 in range(Ly):
                    site_sequence.append(shifted_lattice[x1][y2][1])

            translation_order_list.append(site_sequence)

    # Return all non-zero translations of lattice

    return translation_order_list[1:]


def mirror_states(state, translation_order_list):

    """
    For a single state represented by a ternary string, this functions finds all equivalent 'mirror' states
    related by symmetry, and their base-10 integer representation. The state with the smallest base-10 integer
    representation is chosen to be the 'representative state'. The number of unique mirror states for a given
    state, n_unique, is also found.


    Parameters
    ----------
    state : str
        A state represented by a ternary string.
    translation_order_list : list
        A list of translationally equivalent site orderings. This is returned by the function lattice_translations.

     Returns
    -------
    sorted_ints : list
        A list of the base-10 integers (in ascending order) representing the equivalent mirror states.
    n_unique : int
        The number of unique mirror states.
    rep_state : str
        A ternary string corresponding to the state's representative state.
    rep_int : int
        A base-10 integer corresponding to the state's representative state.

    """

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

    """"
    For a lattice with L sites and thus 3^L possible configurations, this function finds all the representative
    states we need to keep, thus reducing the Hilbert space dimension from 3^L to a much smaller number ndim.


    Parameters
    ----------
    L : int
        The total number of lattice sites.
    translation_order_list : list
        A list of translationally equivalent site orderings. This is returned by the function lattice_translations.
    ints_filename: str
        Filename for the .npy file created which stores the ndim representative states (in base-10).
    states_filename : str
        Filename for the .npy file created which stores the representative states (in base-10) for each of
        the 3^L possible lattice configurations, i.e. a mapping from a state to its representative state.
    nunique_filename : str
        Filename for the .npy file created which stores the number of unique mirror states for each
        representative state.

     Returns
    -------
    kept_ints : ndarray
        An array of the base-10 integers corresponding to the representative states.
    state_map : ndarray
        An array containing the representative state (in base-10) for each of the 3^L possible states.
    n_unique_list : ndarray
        An array containing the number of unique mirror states for each representative state.
    ndim : int
        The total number of representative states we keep, i.e. the Hilbert space dimension is reduced
        from 3^L to ndim.

    """

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


def get_neighbors(Lx, Ly, bond):

    """
    For a given bond-direction in the honeycomb lattice (x, y, or z), this finds all the pairs of sites
    along this type of bond.

    Parameters
    ----------
    Lx : int
        Number of units cells in the x-direction.
    Ly : int
        Number of units cells in the y-direction.
    bond : str
        A string which is either "x", "y", or "z" corresponding to the desired honeycomb bond direction.

     Returns
    -------
    neighbors : list
        A list of tuples of site pairs which each constitute a bond of the chosen type.

    """

    # Note: Total number of sites in lattice: L = Lx * Ly * 2

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
                y_bond_right = unit_cells[(i+1) % Lx][j][0]
                y_pair = tuple(sorted((y_bond_left, y_bond_right)))
                neighbors.append(y_pair)

    if bond == 'z':

        for i in range(Lx):
            for j in range(Ly):

                z_bond_right = unit_cells[i][j][1]
                z_bond_left = unit_cells[i][(j+1) % Ly][0]
                z_pair = tuple(sorted((z_bond_left, z_bond_right)))
                neighbors.append(z_pair)

    return neighbors
