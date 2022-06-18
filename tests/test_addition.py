import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from example_package import example


def test_addition():

    assert example.add_one(3) == 4


test_addition()
