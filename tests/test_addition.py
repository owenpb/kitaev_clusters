import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from kitaev_clusters import symmetry_functions


def test_addition():

    assert symmetry_functions.add_one(3) == 4


test_addition()
