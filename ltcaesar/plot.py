"""
Includes a bunch of plotting funtions that are useful for analysing the LTCaesar
outputs.

Each of these need only be passed a simulation object. There are also
corresponding analysis functions that do the heavy lifting and return the raw
data should you wish to analyse it yourself.
"""

from .objects import Simulation

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm


def bin_x_by_y(x, y, xbins, average_func=np.mean):
    """
    Takes two quantities, x, y, and bins them in xbins w.r.t. x.

    Returns the centers of each x bin, the means of y in each bin, and the
    standard devaitions of y in each bin which can be used as "errors".

    Average_func is the function used to "average" the data; this defaults to
    np.mean but it should be realtively easy to swap this out for np.median,
    for instance.
    """

    output_means = []
    output_center_bin = []
    output_stdev = []

    bin_edges = [[x, y] for x, y in zip(xbins[:-1], xbins[1:])]

    for this_bin in bin_edges:
        this_mask = np.logical_and(x < this_bin[1], x > this_bin[0])
        this_data = y[this_mask]

        # Check for emptiness
        if 0 == len(this_data):
            continue

        output_center_bin.append(this_bin[0] + 0.5 * (this_bin[1] - this_bin[0]))

        output_means.append(this_data.mean())
        output_stdev.append(this_data.std())

    return np.array(output_center_bin), np.array(output_means), np.array(output_stdev)


def get_protected_fraction(x, y):
    """
    Takes two arrays and returns a protected fraction (as well as a mask) where
    y == 0. This prevents div/0 errors.
    """

    mask = y == 0
    x_masked = x[mask]
    y_masked = y[mask]

    return x / y, mask


def plot_errorbars_and_filled_region(ax, x, y, yerr, **kwargs):
    """
    Plot both the errorbars and filled region on ax.

    Kwargs are passed to ax.errorbar()
    """

    ax.errorbar(x, y, yerr, **kwargs)
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    return


def mass_fraction_transfer_from_lr_data(sim: Simulation, bins=None, average_func=np.mean):
    """
    Gets reduced data in the following format: fraction of mass that comes from the
    halo's own lagrangian region, from other lagrangian regions, and from outside
    any lagrangian region, all as a function of halo mass.

    The halo mass is taken as log10(M)

    Gets a dictionary of the following form:

    return = {
      "halo_mass": centre of each halo mass bin,
      "mass_fraction_from_lr": (mean) mass coming from each lagrangian region,
      "mass_fraction_from_other_lr": (mean) mass coming from another lagrangian region,
      "mass_fraction_from_outside_lr": (mean) mass fraction coming from outside any
                                       lagrangian region,
      "mass_fraction_from_lr_stddev": standard devation, rather than the mean,
      "mass_fraction_from_other_lr_stddev": same as above,
      "mass_fraction_from_outside_lr_stddev": same as above
    }

    These fractions come directly from the Simulation object that is given.

    Bins should probably be given. If they are not, it is left up to numpy.
    """

    # First, we need to mask out the halos with no gas or stellar content.

    mask = np.logical_and(sim.stellar_mass_in_halo != 0, sim.gas_mass_in_halo != 0)

    # Grab a bunch of masked arrays that are going to be used in the analysis
    masked_gas_mass = sim.gas_mass_in_halo[mask]
    masked_stellar_mass = sim.stellar_mass_in_halo[mask]
    masked_gas_lagrangian_mass = sim.gas_mass_in_halo_from_lagrangian[mask]
    masked_stellar_lagrangian_mass = sim.stellar_mass_in_halo_from_lagrangian[mask]
    masked_gas_other_lagrangian_mass = sim.gas_mass_in_halo_from_other_lagrangian[mask]
    masked_stellar_other_lagrangian_mass = sim.stellar_mass_in_halo_from_other_lagrangian[
        mask
    ]
    masked_gas_outside_lagrangian_mass = sim.gas_mass_in_halo_from_outside_lagrangian[
        mask
    ]
    masked_stellar_outside_lagrangian_mass = sim.stellar_mass_in_halo_from_outside_lagrangian[
        mask
    ]

    # Now reduce the data into fractions
    total_mass = masked_gas_mass + masked_stellar_mass
    total_mass_from_lr = masked_gas_lagrangian_mass + masked_stellar_lagrangian_mass
    total_mass_from_other_lr = (
        masked_gas_other_lagrangian_mass + masked_stellar_other_lagrangian_mass
    )
    total_mass_from_outside_lr = (
        masked_gas_outside_lagrangian_mass + masked_stellar_outside_lagrangian_mass
    )

    fraction_of_mass_from_lr = total_mass_from_lr / total_mass
    fraction_of_mass_from_other_lr = total_mass_from_other_lr / total_mass
    fraction_of_mass_from_outside_lr = total_mass_from_outside_lr / total_mass

    masked_halo_masses = np.log10(sim.dark_matter_mass_in_halo[mask])

    # If no bins, get them!!!

    if bins is None:
        _, bins = np.histogram(masked_halo_masses)

    # Use local routine to fully reduce the data into a single line

    halo_mass, mass_fraction_from_lr, mass_fraction_from_lr_stddev = bin_x_by_y(
        masked_halo_masses, fraction_of_mass_from_lr, bins, average_func
    )

    _, mass_fraction_from_other_lr, mass_fraction_from_other_lr_stddev = bin_x_by_y(
        masked_halo_masses, fraction_of_mass_from_other_lr, bins, average_func
    )

    _, mass_fraction_from_outside_lr, mass_fraction_from_outside_lr_stddev = bin_x_by_y(
        masked_halo_masses, fraction_of_mass_from_outside_lr, bins, average_func
    )

    return {
        "halo_mass": halo_mass,
        "mass_fraction_from_lr": mass_fraction_from_lr,
        "mass_fraction_from_other_lr": mass_fraction_from_other_lr,
        "mass_fraction_from_outside_lr": mass_fraction_from_outside_lr,
        "mass_fraction_from_lr_stddev": mass_fraction_from_lr_stddev,
        "mass_fraction_from_other_lr_stddev": mass_fraction_from_other_lr_stddev,
        "mass_fraction_from_outside_lr_stddev": mass_fraction_from_outside_lr_stddev,
    }


def mass_fraction_transfer_from_lr_plot(sim: Simulation, bins=None, average_func=np.mean):
    """
    Sets up and returns a figure, ax object based on the above data reduction.
    """

    fig, ax = plt.subplots(1)

    data = mass_fraction_transfer_from_lr_data(sim, bins, average_func)

    plot_errorbars_and_filled_region(
        ax,
        data["halo_mass"],
        data["mass_fraction_from_lr"],
        data["mass_fraction_from_lr_stddev"],
        label="From LR",
    )
    plot_errorbars_and_filled_region(
        ax,
        data["halo_mass"],
        data["mass_fraction_from_other_lr"],
        data["mass_fraction_from_other_lr_stddev"],
        label="From other LR",
    )
    plot_errorbars_and_filled_region(
        ax,
        data["halo_mass"],
        data["mass_fraction_from_outside_lr"],
        data["mass_fraction_from_outside_lr_stddev"],
        label="From outside LR",
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("log$_{10}$(M$_{halo}$ (code units))")
    ax.set_ylabel("Fraction of mass at $z=0$")

    ax.legend(frameon=False, loc=7)

    fig.tight_layout()

    return fig, ax


def find_distances_to_nearest_neighbours_data(sim: Simulation, particle_type="gas"):
    """
    Creates a kdtree of all of the dark matter particles, and looks for their
    nearest <x> neighbour. If you pass in dark_matter, we look for the second
    closest neighbour (as the closest neighbour, of course, is the particle
    itself).

    A number of caveats to this function are explained inline. Note that it assumes
    that dark matter is never created or destroyed.
    """

    boxsize = sim.snapshot_ini.header["BoxSize"]
    tree = KDTree(sim.snapshot_ini.dark_matter.coordinates, boxsize=boxsize)

    if particle_type == "dark_matter":
        particle_ini_coordinates = sim.snapshot_ini.dark_matter.coordinates
        particle_ini_ids = sim.snapshot_ini.dark_matter.ids

        radii, ids = tree.query(particle_ini_coordinates, k=2, n_jobs=-1)

        # Now cut out everything but the second neighbour (we don't care about ourselves)
        radii = radii[:, 1]
        ids = ids[:, 1]

        # We do this because we actually want a list of nearest neighbours, not
        # the real current distances. Note that these are strictly still "sorted"
        # by ID!
        particle_ini_neighbour_indicies = ids
        particle_ini_neighbour_ids = sim.snapshot_ini.dark_matter.ids[ids]
        paritcle_ini_neighbour_coordinates = sim.snapshot_ini.dark_matter.coordinates[
            ids
        ]

    elif particle_type == "gas":
        particle_ini_coordinates = sim.snapshot_ini.baryonic_matter.gas_coordinates
        particle_ini_ids = sim.snapshot_ini.baryonic_matter.gas_ids

        radii, ids = tree.query(particle_ini_coordinates, k=1, n_jobs=-1)

        particle_ini_neighbour_indicies = ids
        particle_ini_neighbour_ids = sim.snapshot_ini.dark_matter.ids[ids]
        particle_ini_neighbour_coordinates = sim.snapshot_ini.dark_matter.ids[ids]
    else:
        raise AttributeError(
            (
                "Unable to use {} particle type. Please supply gas or dark_matter "
                "-- note that the lack of stars at z=inf means that supplying "
                "stars is insignificant."
            ).format(particle_type)
        )

    # Now we need to find the distance to the same particles but at z=0.
    # This _should_ be a fairly simple thing to do, but somebody decided
    # that we should have sub-grid physics. That means that:
    # a) some particles may have been destroyed
    # b) some particles may have been turned into _more than one particle_
    # c) particles must have their distances wrapped correctly.
    # For this reason, we simply go through the particle id list instead of using
    # numpy routines. We will try to match everything available with its redshift
    # 100 counterpart.
    # This may introduce some bias, which needs to be taken care of, but for now
    # I am afraid this is the best that we can do -- we cannot really track what
    # died in a black hole.

    # Combine the two ID arrays, if we're using the gas/stars.
    if particle_type == "gas":
        # We need to truncate.
        truncate = sim.snapshot_end.baryonic_matter.truncate_ids + 1

        particle_ids_end = np.concatenate(
            (
                sim.snapshot_end.baryonic_matter.gas_ids % truncate,
                sim.snapshot_end.baryonic_matter.star_ids % truncate,
            )
        )

        particle_coordinates_end = np.concatenate(
            (
                sim.snapshot_end.baryonic_matter.gas_coordinates,
                sim.snapshot_end.baryonic_matter.star_coordinates,
            )
        )

        # Now we need to re-sort the ID's to "mix" them up.
        index_of_unique = np.argsort(particle_ids_end)

        particle_ids_end = particle_ids_end[index_of_unique]
        particle_coordiantes_end = particle_coordinates_end[index_of_unique]
    elif particle_type == "dark_matter":
        # This one is a little easier...
        particle_ids_end = sim.snapshot_end.dark_matter.ids
        particle_coordinates_end = sim.snapshot_end.dark_matter.coordinates

    # This is where we assume no DM has been destroyed, and that it is currently
    # still sorted by ID.
    particle_end_neighbour_coordinates = sim.snapshot_end.dark_matter.coordinates

    # Now we can do the main processing loop.

    final_radii = np.empty_like(particle_ids_end)

    current_index = 0
    current_id = particle_ids_end[current_index]
    current_coordinate = particle_coordinates_end[current_index]

    for this_particle_ini_id, this_neighbour_ini_index in zip(
        tqdm(particle_ini_ids), particle_ini_neighbour_indicies
    ):
        # Check if it's survived, and if so we can operate on it.
        while this_particle_ini_id == current_id:
            # Grab neighbouring particle
            neighbour_coordinate = particle_end_neighbour_coordinates[
                this_neighbour_ini_index
            ]

            dx = neighbour_coordinate - current_coordinate

            # Make sure that we wrap correctly
            dx -= (dx > boxsize * 0.5) * boxsize
            dx += (dx <= -boxsize * 0.5) * boxsize

            r = np.sum(dx * dx, axis=0)

            final_radii[current_index] = r

            # Iterate; we _need_ to do this in the while loop just in case we have repeated
            # particles.
            try:
                current_index += 1
                current_id = particle_ids_end[current_index]
                current_coordinate = particle_coordinates_end[current_index]
            except IndexError:
                # We've reached the end, team!
                final_index = current_index
                # Kill the loop
                current_index = -1
                pass

    # final_radii is actually full of r^2 -- this is to enable us to use vectorized sqrt.
    final_radii = np.sqrt(final_radii)
    # We're done!

    assert final_index == len(final_radii), "Current Index: {}, length: {}".format(
        current_index, len(final_radii)
    )

    return radii, final_radii


def find_distances_to_nearest_neighbours_plot(sim: Simulation, bins=100):
    """
    Makes a histogram plot of the distribution of distances between particles at
    redshift 0 and redshift infinity. See the data function for more.

    This takes a really long time!
    """

    boxsize = sim.snapshot_ini.header["BoxSize"]

    gas_data = find_distances_to_nearest_neighbours_data(sim, "gas")
    dark_matter_data = find_distances_to_nearest_neighbours_data(sim, "dark_matter")

    fig, ax = plt.subplots(1)

    ax.hist(gas_data[1], bins=bins, range=(0, boxsize), histtype="stepfilled", alpha=0.5, label="Gas")
    ax.hist(dark_matter_data[1], bins=bins, range=(0, boxsize), histtype="stepfilled", alpha=0.5, label="Dark Matter")

    ax.legend(frameon=False)

    ax.set_xlabel("$r_{ngb}$, distance to initial nearest neighbour (simulation units)")

    fig.tight_layout()

    return fig, ax

