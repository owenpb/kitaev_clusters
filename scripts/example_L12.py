########################################################################################
# Example script for 12-site spin-1 Kitaev honeycomb model with single-ion anisotropy  #
########################################################################################

###############
# 1. Imports: #
###############

import numpy as np
import scipy
import gc
import time

from kitaev_clusters.symmetry_functions import get_representative_states, lattice_translations, get_neighbors
from kitaev_clusters.hamiltonian_functions import construct_Hxx, construct_Hyy, construct_Hzz, construct_HD, construct_hamiltonian
from kitaev_clusters.ground_state_functions import ground_state, s_squared, entanglement_entropy


###################################
# 2. Defining lattice parameters: #
###################################

L = 12  # Total number of lattice sites (L = Lx * Ly * 2)
Lx = 3  # Unit cells in horizontal direction
Ly = 2  # Unit cells in vertical direction

x_neighbors = get_neighbors(Lx, Ly, bond='x')
y_neighbors = get_neighbors(Lx, Ly, bond='y')
z_neighbors = get_neighbors(Lx, Ly, bond='z')

translation_order_list = lattice_translations(Lx, Ly)

print(f'x-bond neighbors: {x_neighbors}')
print(f'y-bond neighbors: {y_neighbors}')
print(f'z-bond neighbors: {z_neighbors}')
print('\n')
print(f'Lattice translations of sites {list(range(L))} are: \n')

for i in translation_order_list:
    print(f'{i}')


#######################################################################################################################
# 3. Constructing and storing arrays of the representative states, the state map, and number of unique mirror states: #
#######################################################################################################################

kept_ints, state_map, n_unique_list, ndim = get_representative_states(L, translation_order_list,
                                                                      ints_filename='kept_ints_L12',
                                                                      states_filename='state_map_L12',
                                                                      nunique_filename='n_unique_list_L12')


################################################################################################################
# 4. (If needed)                                                                                                #
#    Loading the arrays of representative states, the state map, and number of unique mirror states from file: #
################################################################################################################

# print(f'Hilbert space dimension has been reduced from {3**L} to {ndim}. \n')

# print('Loading kept_ints...\n')
# kept_ints = np.load('kept_ints_L12.npy')

# print('Loading state_map...\n')a
# state_map = np.load('state_map_L12.npy')

# print('Loading n_unique_list...\n')
# n_unique_list = np.load('n_unique_list_L12.npy')


################################################################
# 5. Constructing components of Hamiltonian: Hxx, Hyy, Hzz, HD #
################################################################

start = time.time()
Hxx = construct_Hxx(L, x_neighbors, kept_ints, state_map, n_unique_list, filename='Hxx_L12')
# Hxx = Hxx.astype(dtype='float32')
end = time.time()
print(f'Hxx constructed in time: {round(end - start, 2)} seconds \n')

start = time.time()
Hyy = construct_Hyy(L, y_neighbors, kept_ints, state_map, n_unique_list, filename='Hyy_L12')
# Hyy = Hyy.astype(dtype='float32')
end = time.time()
print(f'Hyy constructed in time: {round(end - start, 2)} seconds \n')

start = time.time()
Hzz = construct_Hzz(L, z_neighbors, kept_ints, filename='Hzz_L12')
# Hzz = Hzz.astype(dtype='float32')
end = time.time()
print(f'Hzz constructed in time: {round(end - start, 2)} seconds \n')

start = time.time()
HD = construct_HD(L, kept_ints, state_map, n_unique_list, filename='HD_L12')
end = time.time()
print(f'HD constructed in time: {round(end - start, 2)} seconds \n')


################################################################################
# 6. (If needed) Loading previously saved components of Hamiltonian from file: #
################################################################################

# print('Loading Hxx...\n')
# Hxx = scipy.sparse.load_npz('Hxx_L12.npz')
# Hxx = Hxx.astype(dtype='float32')  # Optional conversion to float32 if desired

# print('Loading Hyy...\n')
# Hyy = scipy.sparse.load_npz('Hyy_L12.npz')
# Hyy = Hyy.astype(dtype='float32')  # Optional conversion to float32 if desired

# print('Loading Hzz...\n')
# Hzz = scipy.sparse.load_npz('Hzz_L12.npz')
# Hzz = Hzz.astype(dtype='float32')  # Optional conversion to float32 if desired

# print('Loading HD...\n')
# HD = scipy.sparse.load_npz('HD_L12.npz')


###################################################################################################
# 7. Specifying Kitaev model couplings, single-ion anisotropy, and constructing full Hamiltonian: #
###################################################################################################

Kx = 1  # Kitaev coupling along x-bonds
Ky = 1  # Kitaev coupling along y-bonds
Kz = 1  # Kitaev coupling along z-bonds
D = 0.0   # Single-ion anisotropy in [1, 1, 1] direction

H = construct_hamiltonian(Kx, Ky, Kz, D, 'Hxx_L12.npz', 'Hyy_L12.npz', 'Hzz_L12.npz', 'HD_L12.npz')

# If you are not loading Hamiltonian components from previously saved files, can instead use:
# H = (Kx * Hxx) + (Ky * Hyy) + (Kz * Hzz) + (D * HD)


#################################################################################################################
# 8. Calculating ground state (psi_0), ground state energy (E_0), entanglement entropy, and <(Sx + Sy + Sz)^2>: #
#################################################################################################################

print('Finding ground state...\n')
E_0, psi_0 = ground_state(H, L)
print('\n')
print('Calculating <(Sx + Sy + Sz)^2> \n')
Ss = s_squared(HD, L, psi_0)
print(f'<(Sx + Sy + Sz)^2> (per site): {Ss}')
print('\n')
print('Calculating entanglement entropy...\n')
entropy = entanglement_entropy(L, psi_0, kept_ints, state_map, n_unique_list)
print(f'Entanglement entropy: {entropy}')

# Saving results for energy, S-squared, entropy:
f1 = open('results.txt', 'w')
f1.write('D_111, E_gs per site, <(Sx + Sy + Sz)^2>, entropy \n')
f1.write(f'{D}, {E_0}, {Ss}, {entropy} \n')
f1.close()

# Saving ground state:
np.save('psi_0', psi_0)

# If desired, can instead load a previously saved ground state for calculation of entanglement entropy via:
# psi_loaded = np.load('psi_0.npy')


###########################################################
# 9. (If needed) Free memory after calculations complete: #
###########################################################

del Hxx
del Hyy
del Hzz
del HD
del H
del psi_0
gc.collect()
