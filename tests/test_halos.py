"""
Tests the functions in the halos submodule.
"""

from ltcaesar.halos import \
    parse_halos_and_coordinates, \
    find_all_halo_centers

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


def test_find_all_halo_centers():
    """
    This is quite easy to test, we just need to generate things in 
    1D.
    """
    
    halos = np.array([0, 0, 0, 1, 1, 1])
    coordinates = np.array([
        # Halo 0
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        # Halo 1
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]).T

    output, indicies = parse_halos_and_coordinates(halos, coordinates)

    expected_centers = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    expected_radii = np.array([1.0, 1.5])

    centers, radii = find_all_halo_centers(halos, coordinates)

    assert (expected_centers == centers).all()
    assert (expected_radii == radii).all()

