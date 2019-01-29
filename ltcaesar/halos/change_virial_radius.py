"""
Contains a helper function that changes the virial radius of the halos
to include more particles.

Note that we do things in "top-down" fashion; largest halos get their
stuff re-assigned first so that smaller halos can "steal" from them.

The most important funciton in this file is create_new_halo_catalogue.
Check the documentation for that for how to run this analysis.
"""

import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, *args, **kwargs):
        return x


from scipy.spatial import cKDTree as KDTree
from typing import Tuple

from .halos import FakeCaesar, FakeHalo


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
    unique_halos, number_of_particles_in_each_halo_raw = np.unique(cut_halos, return_counts=True)
    
    # We need to make sure that unique halos is contiguous.
    number_of_particles_in_each_halo = np.zeros(unique_halos.max() + 1, dtype=int)
    number_of_particles_in_each_halo[unique_halos] = number_of_particles_in_each_halo_raw

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


def find_all_halo_centers(halos: np.array, coordinates: np.ndarray, boxsize=None):
    """
    This function finds all halo centers as well as radii.

    It does this by looking for the most extreme values (lowest and highest
    in cartesian coordinates), and then assinging the center of these as the
    center of the halo. Note that these coordinates need not belong to the same
    particle; we can take the x-coordinate from one, the y-coordinate from
    another. The radius of the halo is then determined as the maximal distance
    from the center to one of these extreme points.

    Boxsize, if given, allows this to work over periodic boundaries.
    """

    output, output_indicies = parse_halos_and_coordinates(halos, coordinates)

    coordinates_dtype = coordinates.dtype

    centers = np.empty((len(output), 3), dtype=coordinates_dtype)
    radii = np.empty(len(output), dtype=coordinates_dtype)

    for index, halo_coordinates in enumerate(output):
        # Grab extreme values in all dimensions
        if boxsize is None:
            max_values = halo_coordinates.max(axis=0)
            min_values = halo_coordinates.min(axis=0)
            center = 0.5 * (max_values + min_values)

            # Let's just brute force this...
            dx = halo_coordinates - center
            r = np.sqrt(np.sum(dx * dx, axis=1))

            centers[index] = center
            radii[index] = r.max()
        else:
            # Periodic boundary conditions, oh jeez.
            # First, we select one of the particles to act as a reference
            # point for all relative/periodic calculations.
            try:
                relative_coord = halo_coordinates[0]
            except IndexError:
                # Must be an empty boi
                continue
            # This gets us the coordinates in some semblence of "real space"
            # where e.g. a sphere split over the boundary becomes a sphere again
            relative_offsets = halo_coordinates - relative_coord
            relative_offsets -= (relative_offsets > boxsize * 0.5) * boxsize
            relative_offsets += (relative_offsets <= -boxsize * 0.5) * boxsize
            # In this space, we need to do our actual calculations
            max_values = relative_offsets.max(axis=0)
            min_values = relative_offsets.min(axis=0)
            relative_center = 0.5 * (max_values + min_values)

            # Let's just brute force this...
            dx = relative_offsets - relative_center
            r = np.sqrt(np.sum(dx * dx, axis=1))

            # Now we need to unwrap our center
            wrapped_center = relative_center + relative_coord

            centers[index] = wrapped_center
            radii[index] = r.max()

    return centers, radii


def find_particles_in_halo(coordinates: np.ndarray, center: np.array, radius: float):
    """
    Finds all particles (returns a boolean mask) that live in the sphere defined
    by center, radius. (Slow; for many particles you should use a tree!)
    """

    # First, we will chop out the cube that is defined by the center and radius.
    cube_mask = np.logical_and(
        coordinates <= (center + radius), coordinates >= (center - radius)
    ).all(
        axis=1
    )  # (this generates 3xn array)

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

    # Default state: all particles outside halos
    new_halos = np.zeros_like(halos) - 1

    for halo, (center, radius) in enumerate(
        zip(centers, tqdm(radii, desc="Searching for particles in halos"))
    ):
        mask = find_particles_in_halo(coordinates, center, radius * factor)

        new_halos[mask] = halo

    return new_halos


def create_new_halo_catalogue(
    snapshot,
    factor: float,
    n_threads=-1,
    boxsize=None,
    unsort_dm=None,
    unsort_gas=None,
    unsort_star=None,
) -> FakeCaesar:
    """
    Takes a snapshot object, and uses the information in it to re-create a
    halo catalogue with the virial radius increased by "factor".

    Ensure that if you are using a periodic box that boxsize is passed;
    otherwise this fucntion will not correctly find halos and you will end
    up with all of your particles in a single halo.

    The unsort arrays are required if your data has been sorted w.r.t. the
    way that it will be read in. These are essentially the inverse of the
    sort() function. You can get this from searchsorted.
    """

    # First, we'll find the centers and radii of all of the halos based on their
    # Dark matter component.

    if unsort_dm is not None:
        centers, radii = find_all_halo_centers(
            snapshot.dark_matter.halos[unsort_dm],
            snapshot.dark_matter.coordinates[unsort_dm].T,
            boxsize=boxsize,
        )

        # We need to build trees for each of the particle types
        dm_tree = KDTree(snapshot.dark_matter.coordinates[unsort_dm], boxsize=boxsize)
    else:
        centers, radii = find_all_halo_centers(
            snapshot.dark_matter.halos,
            snapshot.dark_matter.coordinates.T,
            boxsize=boxsize,
        )

        # We need to build trees for each of the particle types
        dm_tree = KDTree(snapshot.dark_matter.coordinates, boxsize=boxsize)

    # Re-set radii to be at larger distance
    radii *= factor

    # If there are no particles in gas, etc. we fail out!
    try:
        if unsort_gas is not None:
            gas_tree = KDTree(
                snapshot.baryonic_matter.gas_coordinates[unsort_gas], boxsize=boxsize
            )
        else:
            gas_tree = KDTree(snapshot.baryonic_matter.gas_coordinates, boxsize=boxsize)
    except ValueError:
        gas_tree = None
    try:
        if unsort_star is not None:
            star_tree = KDTree(
                snapshot.baryonic_matter.star_coordinates[unsort_star], boxsize=boxsize
            )
        else:
            star_tree = KDTree(
                snapshot.baryonic_matter.star_coordinates, boxsize=boxsize
            )
    except ValueError:
        star_tree = None

    halos = []

    for halo, (center, radius) in enumerate(zip(centers, radii)):
        dmlist = np.array(
            dm_tree.query_ball_point(x=center, r=radius, n_jobs=n_threads)
        )
        if gas_tree is not None:
            glist = np.array(
                gas_tree.query_ball_point(x=center, r=radius, n_jobs=n_threads)
            )
        else:
            glist = np.array([])
        if star_tree is not None:
            slist = np.array(
                star_tree.query_ball_point(x=center, r=radius, n_jobs=n_threads)
            )
        else:
            slist = np.array([])

        halos.append(
            FakeHalo(
                dmlist=dmlist,
                ndm=len(dmlist),
                glist=glist,
                ngas=len(glist),
                slist=slist,
                nstar=len(slist),
                GroupID=halo,
            )
        )

    # There may be some problem here with a particle being assigned to two
    # halos, but hopefully that will be sorted out in the processing loop later
    # when the data is re-imported.

    # We can now convert to a FakeCaesar object

    halo_catalogue = FakeCaesar(halos=halos, nhalos=len(halos))

    return halo_catalogue
