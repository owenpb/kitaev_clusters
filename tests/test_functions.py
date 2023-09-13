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


test_tern_to_base10()
test_ternary()
test_tern_to_base10()

