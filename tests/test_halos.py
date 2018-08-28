"""
Tests the functions in the halos submodule.
"""

from ltcaesar.halos import \
    parse_halos_and_coordinates, \
    find_all_halo_centers, \
    find_particles_in_halo 

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


def test_find_particles_in_halo():
    """
    This test is performed in 2D.
    """

    x_coordinates = np.linspace(0, 100, 100)
    y_coordinates = np.linspace(0, 100, 100)
    z_coordinates = np.zeros(100)

    coordinates = np.array([x_coordinates, y_coordinates, z_coordinates]).T
    center = np.array([50.0, 50.0, 0.0])
    radius = 10.0

    mask = find_particles_in_halo(coordinates, center, radius)
    
    # Generate expected answers

    dx = coordinates - center
    expected_radii = np.array([sum(x*x)**(0.5) for x in dx])
    expected_mask = expected_radii <= radius

    assert (expected_mask == mask).all()


