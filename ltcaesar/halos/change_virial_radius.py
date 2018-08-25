"""
Contains a helper function that changes the virial radius of the halos
to include more particles.

TODO: Actually write documentation

Note that we do things in "top-down" fashion; largest halos get their
stuff re-assigned first so that smaller halos can "steal" from them.
"""

import numpy as np

from tqdm import tqdm


def parse_halos_and_coordinates(halos: np.array, coordinates: np.ndarray):
    """
    Parses all of the halos.

    This returns a structure which has length equal to the number of distinct
    halos and contains an array of coordinates for each halo. It also returns
    a similar structure but that contains the indicies of these coordinates
    in the original array, for when it comes time to re-find and re-place
    them.
    """

    # Cut out all of the particles not currently in a halo
    halo_mask = halos != -1
    cut_halos = halos[halo_mask]
    cut_indicies = np.arange(len(halos))[halo_mask]
    cut_coordinates = coordinates.T[halo_mask]

    coordinates_dtype = cut_coordinates.dtype
    indicies_dtype = cut_indicies.dtype

    # Do first pass to find out how many particles are in each halo
    _, number_of_particles_in_each_halo = np.unique(cut_halos, return_counts=True)

    # Now we can allocate our list of arrays ready to fill it up
    output = np.array(
        [
            np.empty((x, 3), dtype=coordinates_dtype)
            for x in number_of_particles_in_each_halo
        ]
    )
    output_indicies = np.array(
        [np.empty(x, dtype=indicies_dtype) for x in number_of_particles_in_each_halo]
    )
    current_position_in_output = np.zeros(len(output), dtype=int)

    # We can probably replace this loop with some smart arary manipulations
    for coordinate, index, halo in zip(cut_coordinates, cut_indicies, cut_halos):
        current_position = current_position_in_output[halo]

        output[halo][current_position] = coordinate
        output_indicies[halo][current_position] = index

        current_position_in_output[halo] += 1

    return output, output_indicies


def find_all_halo_centers(halos: np.array, coordinates: np.ndarray):
    """
    This function finds all halo centers as well as radii.
    """

    raise NotImplementedError

    return centers, radii


def find_particles_in_halo(coordinates: np.ndarray, center: np.array, radius: float):
    """
    Finds all particles (returns a boolean mask) that live in the sphere defined
    by center, radius.
    """

    raise NotImplementedError

    return mask

