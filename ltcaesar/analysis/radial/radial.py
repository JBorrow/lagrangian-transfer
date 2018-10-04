"""
Contains functions for exploring radial trends within halos
with data from LTCaesar.

The main function in this file is run_analysis_on_mass_bin.

What this function does is take the following objects:

    + Simulation
    + A lower and upper halo mass bound (M_sun)
    + A set of radial bins (normalized between 0 and 1, as the data will
      be set to be within the virial radius)
    + Particle type (string)
    + Conversion factor for simulation units to M_sun

and returns a 3xn_radial_bins array which contains the fraction of mass
as a function of radius that comes from the halo's own lagrangian
region, another halos lagrangian region, and from outside any lagrangian
region. 

Example:

    sim = ltcaesar.read_data_from_file("")
    bin_edges = np.linspace(0, 1, 50)
    from_own_lr, from_other_lr, from_outside_lr = run_analysis_on_mass_bin(
        simulation=sim,
        lower=1e12,
        uppwer=1e13,
        [[x, y] for x, y in zip(bin_edges[:-1], bin_edges[1:])]
    )
"""

import numpy as np

from typing import Tuple
from tqdm import tqdm

from ltcaesar.objects import Simulation


class NoMaskError(Exception):
    """
    Custom exception for when the mask fails to find any particles
    """

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def get_masks(halo, simulation: Simulation) -> Tuple[np.ndarray]:
    """
    Get the masks for the ID, coordinate, mass arrays for a given halo.

    Three boolean arrays are returned.
    
    Returns gas, star, then DM.
    """

    baryonic_masks = {}

    for ptype in ["gas", "star"]:
        baryonic_masks[f"{ptype}_mask"] = (
            getattr(simulation.snapshot_end.baryonic_matter, f"{ptype}_halos") == halo
        )

    dark_matter_mask = simulation.snapshot_end.dark_matter.halos == halo

    return baryonic_masks["gas_mask"], baryonic_masks["star_mask"], dark_matter_mask


def get_center(coordinates: np.ndarray) -> np.ndarray:
    """
    Gets the center of some coordinates in 3d space.

    This simply finds center of the minimal and maximal position.
    """

    output = np.empty(3)

    for i, coords in enumerate(coordinates.T):
        output[i] = 0.5 * (np.max(coords) + np.min(coords))

    return output


def get_relative_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """
    Returns the relative coordinates to the center of the object.

    Given a set of coordinates, this finds the center of them using
    get_center and then finds relative offsets to that.
    """

    halo_center = get_center(coordinates)
    dx = coordinates - halo_center

    return dx


def get_radii_to_halo_center(coordinates: np.ndarray) -> np.array:
    """
    Returns the radii of the coordinates given to the center of those
    coordinates.
    """

    dx = get_relative_coordinates(coordinates)
    radii = np.sqrt(np.sum(dx * dx, axis=1))

    return radii


def get_relevant_baryonic_quantities(
    halo, simulation: Simulation, ptype="gas"
) -> Tuple[np.ndarray]:
    """
    Extract all of the relevant baryonic quantities for the radial
    analysis from the simulation data object.
    
    Returns radii, masses, lagrangian regions, halos, for a given halo 
    and particle type.
    """

    gas_mask, star_mask, dark_matter_mask = get_masks(halo, simulation)
    mask = locals()[f"{ptype}_mask"]

    coordinates = getattr(
        simulation.snapshot_end.baryonic_matter, f"{ptype}_coordinates"
    )[mask]
    masses = getattr(simulation.snapshot_end.baryonic_matter, f"{ptype}_masses")[mask]
    lagrangian_regions = getattr(
        simulation.snapshot_end.baryonic_matter, f"{ptype}_lagrangian_regions"
    )[mask]
    halos = getattr(simulation.snapshot_end.baryonic_matter, f"{ptype}_halos")[mask]

    radii = get_radii_to_halo_center(coordinates)

    return radii, masses, lagrangian_regions, halos


def get_extra_bayryonic_quantity(
    halo, simulation: Simulation, quantity: str, ptype="gas"
) -> np.npdarray:
    """
    Extracts any extra quantity you may wish to use, and indexes it
    for the halo in a similar way for get_relevant_baryonic_quantities.
    """

    gas_mask, star_mask, dark_matter_mask = get_masks(halo, simulation)
    mask = locals()[f"{ptype}_mask"]

    return simulation.baryonic_matter.read_extra_array(quantity, ptype)[mask]


def find_halos_in_bin(
    lower: float, upper: float, simulation: Simulation, conversion=1e10 / 0.7
) -> np.ndarray:
    """
    Gets a mask for <x>.halos based on their dark matter halo mass.

    Conversion is the conversion factor from internal units to
    solar masses (as it is expected that lower, upper mass bins are
    given in solar masses).
    """

    return np.logical_and(
        simulation.dark_matter_mass_in_halo * conversion < upper,
        simulation.dark_matter_mass_in_halo * conversion > lower,
    )


def mass_fraction(mask, halo, simulation, ptype):
    """
    Finds the fraction of mass from inside, outside, and from other
    lagrangian regions, in a given mass bin.
    """

    radii, masses, lagrangian_regions, halos = get_relevant_baryonic_quantities(
        halo, simulation, ptype
    )

    relevant_lr = lagrangian_regions[mask]
    relevant_halos = halos[mask]
    # Use fractional masses
    relevant_masses = masses[mask]
    # These need to be normalized on a bin-by-bin basis
    relevant_masses /= np.sum(relevant_masses)

    assert np.isclose(
        np.sum(relevant_masses), 1
    ), f"Sum: {np.sum(relevant_masses)}, length: {len(relevant_masses)}"

    from_own_lr = 0.0
    from_other_lr = 0.0
    from_outside_lr = 0.0

    for lr, halo, mass in zip(relevant_lr, relevant_halos, relevant_masses):
        if lr == halo:
            from_own_lr += mass
        elif lr == -1:
            from_outside_lr += mass
        elif lr != -1:
            from_other_lr += mass
        else:
            raise Exception("Particle unassigned")

    sum_of_all = from_own_lr + from_other_lr + from_outside_lr

    assert np.isclose(sum_of_all, 1), (
        f"From Own: {from_own_lr}, From Other: {from_other_lr} ",
        f"From Outside: {from_outside_lr}, sum: {sum_of_all}.",
    )

    return from_own_lr, from_other_lr, from_outside_lr


def run_analysis_on_individual_halo(
    simulation: Simulation, halo, radial_bins, ptype="gas", bin_func=mass_fraction
) -> Tuple[np.ndarray]:
    """
    Run the analysis based on an individual halo to find the mass
    fractions from each component, the own lagrangian region, the other lagrangian
    regions, and from outside any lagrangian region, as a function of radius.

    This analysis is very slow and naive.

    Returns two arrays; the first being a len(radial_bins) x 3 array
    that contains the fraction of mass in that bin coming from a halo's
    own lagrangian region, from another lagrangian region, and from
    outside any lagrangian region. The second array is essentially a
    boolean that tells us whether or not that bin contained _any_ particles;
    some bins will be particle-less and must be excluded from the eventual
    normalisation. 
    """

    radii, _, _, _ = get_relevant_baryonic_quantities(halo, simulation, ptype)

    # Normalize our radii by the virial radius
    radii /= np.max(radii)

    def find_fraction(this_bin):
        """
        Finds the fractions of mass from each component
        in a given bin.
        """
        mask = np.logical_and(radii > this_bin[0], radii < this_bin[1])

        # We have found no particles! How sad.
        if np.sum(mask) == 0:
            raise NoMaskError

        return bin_func(mask, halo, simulation, ptype)

    output = []
    output_used = []

    for this_bin in radial_bins:
        try:
            output.append(find_fraction(this_bin))
            output_used.append(1)
        except NoMaskError:
            output.append([0.0, 0.0, 0.0])
            output_used.append(0)

    output = np.array(output).T
    output_used = np.array(output_used)

    return output, output_used


def run_analysis_on_mass_bin(
    simulation: Simulation,
    lower: float,
    upper: float,
    radial_bins,
    ptype="gas",
    conversion=1e10 / 0.7,
    bin_func=mass_fraction
):
    """
    Run the analysis on an individual halo mass bin.
    
    Simulation is the simulation data object, lower is the lower mass bin (in
    solar masses), upper is the upper mass bin (in solar masses), and
    radial_bins is the set of radial bins between 0 and 1 (these are normalized
    and stacked with the virial radius). Also returns the standard deviation

    This is a _very_ slow for loop over all halos.
    """

    halo_mask = find_halos_in_bin(lower, upper, simulation, conversion=conversion)
    halos = np.arange(len(halo_mask))[halo_mask]

    profiles = []
    normalization_factor = np.zeros([3, len(radial_bins)])

    for halo in tqdm(halos):
        try:
            individual_analyis, individual_analysis_used = run_analysis_on_individual_halo(
                simulation, halo, radial_bins, ptype, bin_func=bin_func
            )

            if np.sum(individual_analyis) == 0.0:
                raise ValueError
            else:
                profiles.append(individual_analyis)
                # This factor is required in case we don't find any particles in a given bin
                # In that case we don't want that bin to contribute to the normalization
                normalization_factor += individual_analysis_used
        except ValueError:
            # This halo has no gas in it, R.I.P. Skip it.
            continue

    full_output = np.sum(profiles, axis=0) / normalization_factor
    # Get standard errors by assuming gaussianity
    standard_deviation = np.std(profiles, axis=0) / np.sqrt(normalization_factor)

    return full_output, standard_deviation
