"""
Contains a helper function that changes the virial radius of the halos
to include more particles.

TODO: Actually write documentation

Note that we do things in "top-down" fashion; largest halos get their
stuff re-assigned first so that smaller halos can "steal" from them.
"""

import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, *args, **kwargs):
        return x


from typing import Tuple

from ltceasar.objects import Snapshot
from halos import FakeCaesar, FakeHalo


def parse_halos_and_coordinates(
    halos: np.array, coordinates: np.ndarray
) -> Tuple[np.ndarray]:
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

    It does this by looking for the most extreme values (lowest and highest
    in cartesian coordinates), and then assinging the center of these as the
    center of the halo. Note that these coordinates need not belong to the same
    particle; we can take the x-coordinate from one, the y-coordinate from
    another. The radius of the halo is then determined as the maximal distance
    from the center to one of these extreme points.
    """

    output, output_indicies = parse_halos_and_coordinates(halos, coordinates)

    coordinates_dtype = coordinates.dtype

    centers = np.empty((len(output), 3), dtype=coordinates_dtype)
    radii = np.empty(len(output), dtype=coordinates_dtype)

    for index, halo_coordinates in enumerate(output):
        # Grab extreme values in all dimensions
        max_values = halo_coordinates.max(axis=0)
        min_values = halo_coordinates.min(axis=0)
        center = 0.5 * (max_values + min_values)

        # This could be vectorised but would be much less memory efficient
        max_radius = np.sqrt(np.sum((max_values - center) ** 2))
        min_radius = np.sqrt(np.sum((min_values - center) ** 2))

        centers[index] = center
        radii[index] = max([max_radius, min_radius])

    return centers, radii


def find_particles_in_halo(
    coordinates: np.ndarray, center: np.array, radius: float
) -> np.array[bool]:
    """
    Finds all particles (returns a boolean mask) that live in the sphere defined
    by center, radius.
    """

    # First, we will chop out the cube that is defined by the center and radius.
    cube_mask = np.logical_and(
        coordinates <= (center + radius), coordinates >= (center - radius)
    ).all(axis=1)  # (this generates 3xn array)

    coordinates_in_cube = coordinates[cube_mask]

    # Now we can do the "brute force" search to chop out the sphere
    vectors_from_center = coordinates_in_cube - center
    radius_from_center = np.sqrt(
        np.sum(vectors_from_center * vectors_from_center, axis=1)
    )

    radius_mask = radius_from_center <= radius

    # Now we need to kill the areas in the cube mask that have been selected out
    cube_mask[cube_mask] = radius_mask

    return cube_mask


def change_virial_radius(
    halos: np.array,
    coordinates: np.ndarray,
    centers: np.ndarray,
    radii: np.array,
    factor: float,
) -> np.array:
    """
    Change the virial radius of the coordinates and halos such that they have 
    been increased to r_vir -> factor * r_vir.

    Returns a new "halos" array that corresponds to the same ordering as the
    original coordinates.
    
    Assumes that the halos scale with mass; i.e. halo 0 is the most massive,
    with halo -1 the least massive. The re-radii-ing is done top-down (i.e.
    with the most massive halos _first_) so that smaller halos can 'steal'
    particles back from their larger neighbours.
    """

    new_halos = np.empty_like(halos)

    for halo, (center, radius) in enumerate(
        zip(centers, tqdm(radii, desc="Searching for particles in halos"))
    ):
        mask = find_particles_in_halo(coordinates, center, radius * factor)

        new_halos[mask] = halo

    return new_halos


def create_new_halo_catalogue(snapshot: Snapshot, factor: float) -> FakeCaesar:
    """
    Takes a snapshot object, and uses the information in it to re-create a
    halo catalogue with the virial radius increased by "factor".
    """

    # First, we'll find the centers and radii of all of the halos based on their
    # Dark matter component.

    centers, radii = find_all_halo_centers(
        snapshot.dark_matter.halos, snpashot.dark_matter.coordinates
    )

    # Now we have to grab masks for each component individually and add them to a
    # halo list once processed

    halos = []

    for halo, (center, radius) in enumerate(
        zip(centers, tqdm(radii, desc="Searching for particles in halos"))
    ):
        dm_mask = find_particles_in_halo(
            snapshot.dark_matter.coordinates, center, radius * factor
        )
        gas_mask = find_particles_in_halo(
            snapshot.baryonic_matter.gas_coordinates, center, radius * factor
        )
        star_mask = find_particles_in_halo(
            snapshot.baryonic_matter.star_coordinates, center, radius * factor
        )

        # Now we need to populate a FakeHalo object with this information

        halos.append(
            FakeHalo(
                dmlist=np.where(dm_mask),
                ndm=dm_mask.sum(),
                glist=np.where(gas_mask),
                ngas=gas_mask.sum(),
                slist=np.where(star_mask),
                nstar=star_mask.sum(),
                GroupID=halo,
            )
        )

    # We can now convert to a FakeCaesar object

    halo_catalogue = FakeCaesar(halos=halos, nhalos=len(halos))
    halo_catalogue.check_valid()

    return halo_catalogue
