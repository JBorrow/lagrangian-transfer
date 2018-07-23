"""
This unit test looks at some specific particles within the simulation and
by hand checks that they really do end up at the distance to their nearest
neighbour as we say they do.

This is because of the currently very dubious claim presented in the
histogram plot produced by plot.find_distances_to_nearest_neighbours_plot.
"""

import numpy as np

import ltcaesar as lt
from test_halo_finder import test_load_data


def wrap_array(coordinates, boxsize):
    """
    Wraps according to periodic boundary conditions.
    """
    coordinates -= (coordinates > boxsize * 0.5) * boxsize
    coordinates += (coordinates <= -boxsize * 0.5) * boxsize

    return coordinates


def test_distance_to_data(test=None):
    """
    Tests if we are getting the correct distances to the data by hand
    for a few choice particles.

    You can supply a list of gas IDs in the above, but if none are supplied
    then we just check 10 randomly.
    """

    data = test_load_data()

    boxsize = data.snapshot_ini.header["BoxSize"]

    if test is None:
        # We need to choose the IDs to test specifically. These cannot be ones
        # that have formed stars for simplicity purposes.
        available_indicies = np.where(
            data.snapshot_end.baryonic_matter.gas_ids.astype(int)
            - max(data.snapshot_ini.baryonic_matter.gas_ids)
            < 0
        )

        # available_indicies is a tuple that contains the array
        test = np.sort(np.random.choice(available_indicies[0], 10))

    # Read the relevant data for these particles and do some ID matching
    particle_end_ids = data.snapshot_end.baryonic_matter.gas_ids[test]
    particle_end_coordinates = data.snapshot_end.baryonic_matter.gas_coordinates[test]

    # This is the hard part, doing the ID matching.
    particle_ini_indicies = np.searchsorted(
        data.snapshot_ini.baryonic_matter.gas_ids, particle_end_ids, side="left"
    )

    # Sort indicies by ID
    sorted_indicies_for_indicies = np.argsort(
        data.snapshot_ini.baryonic_matter.gas_ids[particle_ini_indicies]
    )
    particle_ini_indicies = particle_ini_indicies[sorted_indicies_for_indicies]

    particle_ini_ids = data.snapshot_ini.baryonic_matter.gas_ids[particle_ini_indicies]
    particle_ini_coordinates = data.snapshot_ini.baryonic_matter.gas_coordinates[
        particle_ini_indicies
    ]

    assert (particle_ini_ids == particle_end_ids).all()

    # Now we need to get the "real" data from the actual data function.

    r_ini, r_end, pair_indices = lt.plot.find_distances_to_nearest_neighbours_data(data)

    # We actually only care about 10 of these numbers...
    relevant_r_ini = r_ini[particle_ini_indicies]
    relevant_r_end = r_end[test]
    relevant_pair_ids = pair_indices[particle_ini_indicies]

    # This is the "brute-force" run.
    for (
        this_r_ini,
        this_r_end,
        this_pair_id,
        this_particle_ini_coordinate,
        this_particle_end_coordinate,
    ) in zip(
        relevant_r_ini,
        relevant_r_end,
        relevant_pair_ids,
        particle_ini_coordinates,
        particle_end_coordinates,
    ):

        # Brute force find
        distances_to_ini = (
            data.snapshot_ini.dark_matter.coordinates - this_particle_ini_coordinate
        )
        distances_to_ini = wrap_array(distances_to_ini, boxsize)

        radii_to_ini = np.sqrt(np.sum(distances_to_ini * distances_to_ini, axis=1))

        minimal_distance_ini = np.min(radii_to_ini)
        assert np.isclose(minimal_distance_ini, this_r_ini)

        # Now we can find the index of our friendly neighbourhood closest
        # particle!

        mask = radii_to_ini == minimal_distance_ini
        neighbour_end_coordinate = data.snapshot_end.dark_matter.coordinates[mask]

        distance_to_end = neighbour_end_coordinate - this_particle_end_coordinate
        distance_to_end = wrap_array(distance_to_end, boxsize)

        radii_to_end = np.sqrt(np.sum(distance_to_end * distance_to_end))

        assert np.isclose(this_r_end, radii_to_end)

    return
