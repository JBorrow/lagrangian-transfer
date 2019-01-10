"""
Includes a bunch of plotting funtions that are useful for analysing the LTCaesar
outputs.

Each of these need only be passed a simulation object. There are also
corresponding analysis functions that do the heavy lifting and return the raw
data should you wish to analyse it yourself.
"""

from ltcaesar.objects import Simulation

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


def mass_fraction_transfer_from_lr_data(
    sim: Simulation, bins=None, average_func=np.mean, use=("stellar", "gas")
):
    """
    Gets reduced data in the following format: fraction of mass that comes from the
    halo's own lagrangian region, from other lagrangian regions, and from outside
    any lagrangian region, all as a function of halo mass.

    The tuple use=("stellar", "gas") is the components that we should include in
    the final calculation.

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

    masks = [getattr(sim, "{}_mass_in_halo".format(x)) != 0 for x in use]
    mask = np.logical_and.reduce(masks)

    # Grab a bunch of masked arrays that are going to be used in the analysis
    masked_mass = sum([getattr(sim, "{}_mass_in_halo".format(x))[mask] for x in use])
    masked_lagrangian_mass = sum(
        [getattr(sim, "{}_mass_in_halo_from_lagrangian".format(x))[mask] for x in use]
    )
    masked_other_lagrangian_mass = sum(
        [
            getattr(sim, "{}_mass_in_halo_from_other_lagrangian".format(x))[mask]
            for x in use
        ]
    )
    masked_outside_lagrangian_mass = sum(
        [
            getattr(sim, "{}_mass_in_halo_from_outside_lagrangian".format(x))[mask]
            for x in use
        ]
    )

    # Now reduce the data into fractions

    fraction_of_mass_from_lr = masked_lagrangian_mass / masked_mass
    fraction_of_mass_from_other_lr = masked_other_lagrangian_mass / masked_mass
    fraction_of_mass_from_outside_lr = masked_outside_lagrangian_mass / masked_mass

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


def mass_fraction_transfer_to_halo_data(
    sim: Simulation, bins=None, average_func=np.mean
):
    """
    Gets reduced data in the following format: fraction of mass in a given
    lagrangian region that ends up in the corresponding halo, the fraction
    of mass that ends up _outside_ any halo, and the fraction of mass that
    ends up in _another_ halo.

    There is no 'use' here as it only makes sense to combine all baryonic
    components together.

    The halo mass is taken as log10(M)

    Gets a dictionary of the following form:

    return = {
      "lr_mass": centre of each halo mass bin,
      "mass_fraction_to_halo": (mean) mass coming from each lagrangian region,
      "mass_fraction_to_other_halo": (mean) mass coming from another lagrangian region,
      "mass_fraction_to_outside_halo": (mean) mass fraction ending up outside any
                                       lagrangian region,
      "mass_fraction_to_halo_stddev": standard devation, rather than the mean,
      "mass_fraction_to_other_halo_stddev": same as above,
      "mass_fraction_to_outside_halo_stddev": same as above
    }

    These fractions come directly from the Simulation object that is given.

    Bins should probably be given. If they are not, it is left up to numpy.
    """

    lagrangian_dm_mass = np.log10(sim.dark_matter_mass_in_lagrangian)

    lagrangian_gas_mass = sim.gas_mass_in_lagrangian[mask]
    mass_in_halo_from_lr = (
        sim.gas_mass_in_halo_from_lagrangian + sim.stellar_mass_in_halo_from_lagrangian
    )[mask]
    mass_outside_halo_from_lr = (
        sim.gas_mass_outside_halo_from_lagrangian
        + sim.stellar_mass_outside_halo_from_lagrangian
    )[mask]
    # By definition this must be true (now we've combined baryonic mass)
    mass_in_other_halo_from_lagrangian = (
        sim.gas_mass_in_other_halos_from_lagrangian
        + sim.stellar_mass_in_other_halos_from_lagrangian
    )[mask]

    # Now reduce the data into fractions

    fraction_to_halo = mass_in_halo_from_lr / lagrangian_gas_mass
    fraction_to_other_halo = mass_in_other_halo_from_lagrangian / lagrangian_gas_mass
    fraction_to_outside_halo = mass_outside_halo_from_lr / lagrangian_gas_mass

    # If no bins, get them!!!

    if bins is None:
        _, bins = np.histogram(masked_halo_masses)

    # Use local routine to fully reduce the data into a single line

    lr_mass, mass_fraction_to_halo, mass_fraction_to_halo_stddev = bin_x_by_y(
        lagrangian_dm_mass, fraction_to_halo, bins, average_func
    )

    _, mass_fraction_to_other_halo, mass_fraction_to_other_halo_stddev = bin_x_by_y(
        lagrangian_dm_mass, fraction_to_other_halo, bins, average_func
    )

    _, mass_fraction_to_outside_halo, mass_fraction_to_outside_stddev = bin_x_by_y(
        lagrangian_dm_mass, fraction_to_outside_halo, bins, average_func
    )

    return {
        "lr_mass": lr_mass,
        "mass_fraction_to_halo": mass_fraction_to_halo,
        "mass_fraction_to_other_halo": mass_fraction_to_other_halo,
        "mass_fraction_to_outside_halo": mass_fraction_to_outside_halo,
        "mass_fraction_to_halo_stddev": mass_fraction_to_halo_stddev,
        "mass_fraction_to_other_halo_stddev": mass_fraction_to_other_halo_stddev,
        "mass_fraction_to_outside_halo_stddev": mass_fraction_to_outside_stddev,
    }


def mass_fraction_transfer_from_lr_plot(
    sim: Simulation, bins=None, average_func=np.mean, use=("stellar", "gas")
):
    """
    Sets up and returns a figure, ax object based on the above data reduction.
    """

    fig, ax = plt.subplots(1)

    data = mass_fraction_transfer_from_lr_data(sim, bins, average_func, use=use)

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

    Returns:

        + Distance to nearest neighbours at z=ini
        + Distance to that same neighbour at z=end
        + Particle _indicies_ of DM neighbours (useful for testing).

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

    elif particle_type == "gas" or particle_type == "stars":
        particle_ini_coordinates = sim.snapshot_ini.baryonic_matter.gas_coordinates
        particle_ini_ids = sim.snapshot_ini.baryonic_matter.gas_ids

        radii, ids = tree.query(particle_ini_coordinates, k=1, n_jobs=-1)

        particle_ini_neighbour_indicies = ids
        particle_ini_neighbour_ids = sim.snapshot_ini.dark_matter.ids[ids]
        particle_ini_neighbour_coordinates = sim.snapshot_ini.dark_matter.coordinates[
            ids
        ]
    else:
        raise AttributeError(
            (
                "Unable to use {} particle type. Please supply gas, stars or dark_matter "
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

    # Now we can do the main processing loop.

    def single_processing_loop(
        ids_end, coords_end, ids_ini, neighbour_coords_end, neighbour_indicies, boxsize
    ):
        """
        We break this out into a single function to try to keep things general; we
        need to actually loop over gas and stars. Trying to combine the star and
        gas arrays initially before doing this loop was an abject failure.
        """

        final_dx = np.empty_like(coords_end)
        final_neighbour_indicies = np.empty(len(coords_end), dtype=int)

        current_index = 0
        current_id = ids_end[current_index]
        current_coordinate = coords_end[current_index]

        # A few consistency checks
        assert len(current_coordinate) == 3

        for (
            this_particle_ini_id,
            this_neighbour_coordinate,
            this_neighbour_index,
        ) in zip(
            tqdm(ids_ini, desc="Distance calculation [{}]".format(particle_type)),
            neighbour_coords_end,
            neighbour_indicies,
        ):
            # Check if it's survived, and if so we can operate on it.
            while this_particle_ini_id == current_id:
                # Grab neighbouring particle

                final_dx[current_index] = this_neighbour_coordinate - current_coordinate
                final_neighbour_indicies[current_index] = this_neighbour_index

                # Iterate; we _need_ to do this in the while loop just in case we have repeated
                # particles.

                try:
                    current_index += 1
                    current_id = ids_end[current_index]
                    current_coordinate = coords_end[current_index]
                except IndexError:
                    # We've reached the end, team!
                    final_index = current_index
                    # Kill the loop
                    current_index = -1
                    pass

        # Now wrap the box and convert to r.
        final_dx -= (final_dx > boxsize * 0.5) * boxsize
        final_dx += (final_dx <= -boxsize * 0.5) * boxsize

        final_radii = np.sqrt(np.sum(final_dx * final_dx, axis=1))
        # We're done!

        return final_radii, final_neighbour_indicies

    # Select out the relevant properties and run

    # This is where we assume no DM has been destroyed, and that it is currently
    # still sorted by ID. We need to re-mix it up so that we can just loop over the
    # coordinates (i.e. sort them by their pair coordinate).
    particle_end_neighbour_coordinates = sim.snapshot_end.dark_matter.coordinates[
        particle_ini_neighbour_indicies
    ]

    if particle_type == "gas":
        if sim.snapshot_end.baryonic_matter.truncate_ids is not None:
            truncate = sim.snapshot_end.baryonic_matter.truncate_ids + 1
            final_radii, final_neighbour_indicies = single_processing_loop(
                sim.snapshot_end.baryonic_matter.gas_ids % truncate,
                sim.snapshot_end.baryonic_matter.gas_coordinates,
                particle_ini_ids,
                particle_end_neighbour_coordinates,
                particle_ini_neighbour_indicies,
                boxsize,
            )
        else:
            final_radii, final_neighbour_indicies = single_processing_loop(
                sim.snapshot_end.baryonic_matter.gas_ids,
                sim.snapshot_end.baryonic_matter.gas_coordinates,
                particle_ini_ids,
                particle_end_neighbour_coordinates,
                particle_ini_neighbour_indicies,
                boxsize,
            )
    elif particle_type == "stars":
        if sim.snapshot_end.baryonic_matter.truncate_ids is not None:
            truncate = sim.snapshot_end.baryonic_matter.truncate_ids + 1
            final_radii, final_neighbour_indicies = single_processing_loop(
                sim.snapshot_end.baryonic_matter.star_ids % truncate,
                sim.snapshot_end.baryonic_matter.star_coordinates,
                particle_ini_ids,
                particle_end_neighbour_coordinates,
                particle_ini_neighbour_indicies,
                boxsize,
            )
        else:
            final_radii, final_neighbour_indicies = single_processing_loop(
                sim.snapshot_end.baryonic_matter.star_ids,
                sim.snapshot_end.baryonic_matter.star_coordinates,
                particle_ini_ids,
                particle_end_neighbour_coordinates,
                particle_ini_neighbour_indicies,
                boxsize,
            )
    if particle_type == "dark_matter":
        final_radii, final_neighbour_indicies = single_processing_loop(
            sim.snapshot_end.dark_matter.ids,
            sim.snapshot_end.dark_matter.coordinates,
            particle_ini_ids,
            particle_end_neighbour_coordinates,
            particle_ini_neighbour_indicies,
            boxsize,
        )

    return radii, final_radii, particle_ini_neighbour_indicies, final_neighbour_indicies


def find_distances_to_nearest_neighbours_plot(sim: Simulation, bins=100):
    """
    Makes a histogram plot of the distribution of distances between particles at
    redshift 0 and redshift infinity. See the data function for more.

    This takes a really long time!
    """

    boxsize = sim.snapshot_ini.header["BoxSize"]

    dark_matter_data = find_distances_to_nearest_neighbours_data(sim, "dark_matter")
    gas_data = find_distances_to_nearest_neighbours_data(sim, "gas")
    star_data = find_distances_to_nearest_neighbours_data(sim, "stars")

    fig, ax = plt.subplots(1)

    ax.hist(
        gas_data[1],
        bins=bins,
        range=(0, boxsize),
        histtype="stepfilled",
        alpha=0.5,
        label="Gas",
    )
    ax.hist(
        dark_matter_data[1],
        bins=bins,
        range=(0, boxsize),
        histtype="stepfilled",
        alpha=0.5,
        label="Dark Matter",
    )
    ax.hist(
        star_data[1],
        bins=bins,
        range=(0, boxsize),
        histtype="stepfilled",
        alpha=0.5,
        label="Stars",
    )

    ax.legend(frameon=False)

    ax.set_xlabel("$r_{ngb}$, distance to initial nearest neighbour (simulation units)")

    fig.tight_layout()

    return fig, ax
