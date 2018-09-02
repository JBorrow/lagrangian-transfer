"""
Tests the functions in the halos submodule.
"""

from ltcaesar.halos import (
    parse_halos_and_coordinates,
    find_all_halo_centers,
    change_virial_radius,
    find_particles_in_halo,
    create_new_halo_catalogue,
)

import numpy as np

import unittest.mock as mock


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
    expected_radii = np.array([sum(x * x) ** (0.5) for x in dx])
    expected_mask = expected_radii <= radius

    assert (expected_mask == mask).all()


def test_find_all_halo_centers():
    """
    This is quite easy to test, we just need to generate things in 
    1D.
    """

    halos = np.array([0, 0, 0, 1, 1, 1])
    coordinates = np.array(
        [
            # Halo 0
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            # Halo 1
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    ).T

    output, indicies = parse_halos_and_coordinates(halos, coordinates)

    expected_centers = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    expected_radii = np.array([1.0, 1.5])

    centers, radii = find_all_halo_centers(halos, coordinates)

    assert (expected_centers == centers).all()
    assert (expected_radii == radii).all()


def test_change_virial_radius():
    """
    Tests to see if the change_virial_radius function actually recovers the
    original solution.
    """

    particles = np.random.rand(3000).reshape((1000, 3))
    halo_radius = 0.2
    halo_center = np.array([0.5, 0.5, 0.5])

    dx = particles - halo_center
    r = np.sqrt(np.sum(dx * dx, axis=1))

    halos = np.empty(1000)
    halos[r <= halo_radius] = 0
    halos[r > halo_radius] = -1

    new_halos = change_virial_radius(
        halos=halos,
        coordinates=particles,
        centers=[halo_center],
        radii=[halo_radius],
        factor=1.0,
    )

    assert (halos == new_halos).all()


def test_create_new_halo_catalogue(contamination=0.01, make_plot=False):
    """
    Tests the creation of a new halo catalogue whilst appropriately mocking
    up data.

    Note that this test runs in full 3D.
    """

    boxsize = 1.0

    # Fake data generation
    particles = np.random.rand(300000).reshape((100000, 3))
    halo_radiis = [0.2, 0.1, 0.2]
    halo_centers = [
        np.array([0.3, 0.3, 0.3]),
        np.array([0.8, 0.8, 0.8]),
        np.array([0.0, 1.0, 0.5]),
    ]
    halos = np.empty(100000, dtype=int)
    halos[...] = -1
    expected_indicies = []

    for n, (r, c) in enumerate(zip(halo_radiis, halo_centers)):
        dx = particles - c

        dx -= (dx > boxsize * 0.5) * boxsize
        dx += (dx <= -boxsize * 0.5) * boxsize

        rads = np.sqrt(np.sum(dx * dx, axis=1))

        mask = rads <= r
        expected_indicies.append(np.where(mask)[0])
        halos[mask] = n

    # Mock up the simulation class

    simulation = mock.MagicMock()
    simulation.dark_matter.halos = halos
    simulation.dark_matter.coordinates = particles
    simulation.baryonic_matter.gas_halos = np.array([])
    simulation.baryonic_matter.star_halos = np.array([])
    simulation.baryonic_matter.gas_coordinates = np.array([])
    simulation.baryonic_matter.star_coordinates = np.array([])

    output = create_new_halo_catalogue(
        simulation, factor=1.0, n_threads=2, boxsize=boxsize
    )

    if make_plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(0, 1)

        for indicies, halo, n in zip(expected_indicies, output.halos, [0, 1, 2]):
            x = particles[indicies].T
            ax[0].scatter(x[0], x[1], s=0.2, lw=0, label=n)
            y = particles[halo.dmlist].T
            ax[1].scatter(y[0], y[1], s=0.2, lw=0, label=n)

        ax[0].set_title("Initial")
        ax[1].set_title("From create_new_halo_catalogue")
        fig.legend(frameon=False)
        fig.savefig("test_create_new_halo_catalogue.png", dpi=300)

    # Now we have to compare the expected indicies.

    for indicies, halo in zip(expected_indicies, output.halos):
        # These will not give exactly pairwise equal results because of the
        # way that we crudely find halos. Hence we need a criteria.
        diff = len(np.setdiff1d(indicies, halo.dmlist))
        total = len(indicies)

        assert diff / total < contamination


def test_coorect_overlap(contamination=0.01, make_plot=False):
    """
    Tests the creation of a new halo catalogue whilst appropriately mocking
    up data.

    Note that this test runs in full 3D.
    """

    boxsize = 1.0

    # Fake data generation
    particles = np.random.rand(300000).reshape((100000, 3))
    halo_radiis = [0.3, 0.1]
    halo_centers = [
        np.array([0.5, 0.5, 0.5]),
        np.array([0.4, 0.5, 0.5]),
    ]
    halos = np.empty(100000, dtype=int)
    halos[...] = -1
    expected_indicies = []

    for n, (r, c) in enumerate(zip(halo_radiis, halo_centers)):
        dx = particles - c

        dx -= (dx > boxsize * 0.5) * boxsize
        dx += (dx <= -boxsize * 0.5) * boxsize

        rads = np.sqrt(np.sum(dx * dx, axis=1))

        mask = rads <= r
        expected_indicies.append(np.where(mask)[0])
        halos[mask] = n

    # Mock up the simulation class

    simulation = mock.MagicMock()
    simulation.dark_matter.halos = halos
    simulation.dark_matter.coordinates = particles
    simulation.baryonic_matter.gas_halos = np.array([])
    simulation.baryonic_matter.star_halos = np.array([])
    simulation.baryonic_matter.gas_coordinates = np.array([])
    simulation.baryonic_matter.star_coordinates = np.array([])

    output = create_new_halo_catalogue(
        simulation, factor=1.0, n_threads=2, boxsize=boxsize
    )

    if make_plot:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(0, 1)

        for indicies, halo, n in zip(expected_indicies, output.halos, [0, 1, 2]):
            x = particles[indicies].T
            ax[0].scatter(x[0], x[1], s=0.2, lw=0, label=n, alpha=0.5)
            y = particles[halo.dmlist].T
            ax[1].scatter(y[0], y[1], s=0.2, lw=0, label=n, alpha=0.5)

        ax[0].set_title("Initial")
        ax[1].set_title("From create_new_halo_catalogue")
        fig.legend(frameon=False)
        fig.savefig("test_create_new_halo_catalogue.png", dpi=300)

    # Now we have to compare the expected indicies.

    for indicies, halo in zip(expected_indicies, output.halos):
        # These will not give exactly pairwise equal results because of the
        # way that we crudely find halos. Hence we need a criteria.
        diff = len(np.setdiff1d(indicies, halo.dmlist))
        total = len(indicies)

        assert diff / total < contamination
