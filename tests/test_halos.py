"""
Tests the functions in the halos submodule.
"""

from ltcaesar.halos import parse_halos_and_coordinates

import numpy as np


def test_parse_halos_and_coordinates():
    """
    Checks that we are recovering the correct output format
    """

    halos = np.array([0, 1, 0, 1])
    coordinates = np.random.rand(12).reshape((3, 4))
    expected_indicies = np.array([[0, 2], [1, 3]], dtype=int)

    output, indicies = parse_halos_and_coordinates(halos, coordinates)

    c = coordinates.T
    expected_output = np.array([[c[0], c[2]], [c[1], c[3]]])

    assert (expected_output == output).all()
    assert (expected_indicies == indicies).all()
