import sys
import os
from kitaev_clusters import symmetry_functions

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")


def test_tern_to_base10():

    assert symmetry_functions.tern_to_base10('120120011221112020') == 223302561


def test_ternary():

    assert symmetry_functions.ternary(30) == '1010'


def test_ternary_pad():

    assert symmetry_functions.ternary_pad(30, 12) == '000000001010'


def test_x_neighbors():

    x_neighbors = symmetry_functions.get_neighbors(Lx=3, Ly=3, bond='x')
    x_pairs = [(0, 3), (2, 5), (1, 4), (6, 9), (7, 10), (8, 11), (12, 15), (13, 16), (14, 17)]

    assert set(x_neighbors) == set(x_pairs)


def test_y_neighbors():

    y_neighbors = symmetry_functions.get_neighbors(Lx=3, Ly=3, bond='y')
    y_pairs = [(3, 6), (4, 7), (5, 8), (9, 12), (10, 13), (11, 14), (0, 15), (1, 16), (2, 17)]

    assert set(y_neighbors) == set(y_pairs)


def test_z_neighbors():

    z_neighbors = symmetry_functions.get_neighbors(Lx=3, Ly=3, bond='z')
    z_pairs = [(1, 3), (2, 4), (0, 5), (7, 9), (8, 10), (6, 11), (13, 15), (14, 16), (12, 17)]

    assert set(z_neighbors) == set(z_pairs)


test_tern_to_base10()
test_ternary()
test_tern_to_base10()
test_x_neighbors()
test_y_neighbors()
test_z_neighbors()


