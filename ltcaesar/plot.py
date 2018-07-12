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


def bin_x_by_y(x, y, xbins):
    """
    Takes two quantities, x, y, and bins them in xbins w.r.t. x.

    Returns the centers of each x bin, the means of y in each bin, and the
    standard devaitions of y in each bin which can be used as "errors".
    """

    output_means = []
    output_center_bin = []
    output_stdev = []

    bin_edges = [[x, y] for x, y in zip(xbins[:-1], xbins[1:])]

    for this_bin in bin_edges:
        this_mask = np.logical_and(x < this_bin[1], x > this_bin[0])
        this_data = y[this_mask]

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
    ax.fill_between(x, y-yerr, y+yerr, alpha=0.2)

    return


def mass_fraction_transfer_from_lr_data(sim: Simulation, bins=None):
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
        masked_halo_masses, fraction_of_mass_from_lr, bins
    )

    _, mass_fraction_from_other_lr, mass_fraction_from_other_lr_stddev = bin_x_by_y(
        masked_halo_masses, fraction_of_mass_from_other_lr, bins
    )

    _, mass_fraction_from_outside_lr, mass_fraction_from_outside_lr_stddev = bin_x_by_y(
        masked_halo_masses, fraction_of_mass_from_outside_lr, bins
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


def mass_fraction_transfer_from_lr_plot(sim: Simulation, bins=None):
    """
    Sets up and returns a figure, ax object based on the above data reduction.
    """

    fig, ax = plt.subplots(1)

    data = mass_fraction_transfer_from_lr_data(sim, bins)

    plot_errorbars_and_filled_region(ax, data["halo_mass"], data["mass_fraction_from_lr"], data["mass_fraction_from_lr_stddev"], label="From LR")
    plot_errorbars_and_filled_region(ax, data["halo_mass"], data["mass_fraction_from_other_lr"], data["mass_fraction_from_other_lr_stddev"], label="From LR")
    plot_errorbars_and_filled_region(ax, data["halo_mass"], data["mass_fraction_from_outside_lr"], data["mass_fraction_from_outside_lr_stddev"], label="From LR")

    ax.set_ylim(0, 1)
    ax.set_xlabel("log$_{10}$(M$_{halo} (code units))")
    ax.set_ylabel("Fraction of mass at $z=0$")

    ax.legend(frameon=False)

    fig.tight_layout()

    return fig, ax


