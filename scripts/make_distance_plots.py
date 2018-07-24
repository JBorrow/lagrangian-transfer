"""
Makes a bunch of different plots based on distance metrics.

This is converted from a python notebook is is quite messy.
"""

import ltcaesar as lt

import matplotlib.pyplot as plt
import numpy as np


sim = lt.read_data_from_file("lt_outputs.hdf5")

dark_matter_data = lt.plot.find_distances_to_nearest_neighbours_data(sim, "dark_matter")
gas_data = lt.plot.find_distances_to_nearest_neighbours_data(sim, "gas")
star_data = lt.plot.find_distances_to_nearest_neighbours_data(sim, "stars")


# The first thing to look at is which particles have been launched by winds. To do that, we'll need some more data from the snapshot file.
#
# Note that wind launches are encoded as follows (using base 10):
#
# + SF kick, `+=1` to `NWindLaunches`
# + AGN kick, `+=1000` to `NWindLaunches`

import h5py

truncate = sim.snapshot_end.baryonic_matter.truncate_ids + 1


with h5py.File("../test_data/snap_m25n256_151.hdf5", "r") as file:
    gas_ids = file["PartType0/ParticleIDs"][...]
    gas_launches = file["PartType0/NWindLaunches"][...]

    indicies = np.argsort(gas_ids % truncate)
    gas_ids = gas_ids[indicies]
    gas_launches = gas_launches[indicies]


# Parsing the above is as easy as killing some bits.

sf_launches = gas_launches % 1000
agn_launches = np.floor_divide(gas_launches - sf_launches, 1000)

plt.semilogy()
plt.hist(sf_launches, bins=28, range=(0, 28), alpha=0.5, label="Stellar")
plt.hist(agn_launches, bins=28, range=(0, 28), alpha=0.5, label="AGN")
plt.legend(frameon=False)
plt.xlim(0, 28)
plt.xlabel("Number of launches")

plt.savefig("n_gas_launches_agn_vs_sf.pdf")

mask_agn = agn_launches != 0
mask_sf = sf_launches != 0


plt.hist(
    gas_data[1],
    label="All",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.hist(
    gas_data[1][mask_sf],
    label="Stellar",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.hist(
    gas_data[1][mask_agn],
    label="AGN",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.xlim(0, 25000)
plt.semilogy()
plt.legend(frameon=False)
plt.xlabel("Distance to original neighbour")

plt.savefig("distance_to_neighbour_based_on_how_launched.pdf")


# So we can conclude that particles kicked by AGN are found the furthest out, with those kicked by stellar feedback a little less further out, followed by the background being much lower. We see this as about an order of magnitude in this tail.

# Now, let's look at the individual components - those particles that started out in their original lagrangian region and were moved.


mask_gas_same_lr = np.logical_and(
    (
        sim.snapshot_end.baryonic_matter.gas_lagrangian_regions
        == sim.snapshot_end.baryonic_matter.gas_halos
    ),
    sim.snapshot_end.baryonic_matter.gas_lagrangian_regions != -1,
)
mask_gas_different_lr = np.logical_and(
    (
        sim.snapshot_end.baryonic_matter.gas_lagrangian_regions
        != sim.snapshot_end.baryonic_matter.gas_halos
    ),
    sim.snapshot_end.baryonic_matter.gas_lagrangian_regions != -1,
)
mask_gas_outside_lr = np.logical_and(
    (
        sim.snapshot_end.baryonic_matter.gas_lagrangian_regions
        == sim.snapshot_end.baryonic_matter.gas_halos
    ),
    sim.snapshot_end.baryonic_matter.gas_lagrangian_regions == -1,
)
mask_gas_from_outside_lr = np.logical_and(
    sim.snapshot_end.baryonic_matter.gas_halos != -1,
    sim.snapshot_end.baryonic_matter.gas_lagrangian_regions == -1,
)


plt.hist(
    dark_matter_data[1],
    label="Dark Matter",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.hist(
    gas_data[1][mask_gas_same_lr],
    label="Same LR",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.hist(
    gas_data[1][mask_gas_different_lr],
    label="Different LR",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.hist(
    gas_data[1][mask_gas_outside_lr],
    label="Always Outside",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.hist(
    gas_data[1][mask_gas_from_outside_lr],
    label="From Outside",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.xlim(0, 25000)
plt.semilogy()
plt.legend(frameon=False)
plt.xlabel("Distance to original neighbour")

plt.savefig("distance_to_original_neighbour_based_on_in_lr_or_not_gas.pdf")


# There's a lot going on in this plot. We see that gas that ends up in the same LR as its Halo (i.e. it stays where it should) follows a very nice distribution that actually is significantly more tightly bound than the dark matter. This is quite unexpected.
#
# The gas that ends up always outside ends up getting mixed around a bit; I presume this is because we are not controlling for gas that in the middle of the simulation in fact _did_ end up in a halo of some kind. Perhaps we can tease this effect out by looking _just_ at star forming gas.
#
# Gas that ends up in the halos from the outside ends up with a significantly wider distribution than regular gas, which is nice and is expected;l as does the gas that ends up in a different LR to the one it originated in.

# Now we will repeat the analysis for stars!


mask_star_same_lr = np.logical_and(
    (
        sim.snapshot_end.baryonic_matter.star_lagrangian_regions
        == sim.snapshot_end.baryonic_matter.star_halos
    ),
    sim.snapshot_end.baryonic_matter.star_lagrangian_regions != -1,
)
mask_star_different_lr = np.logical_and(
    (
        sim.snapshot_end.baryonic_matter.star_lagrangian_regions
        != sim.snapshot_end.baryonic_matter.star_halos
    ),
    sim.snapshot_end.baryonic_matter.star_lagrangian_regions != -1,
)
mask_star_outside_lr = np.logical_and(
    (
        sim.snapshot_end.baryonic_matter.star_lagrangian_regions
        == sim.snapshot_end.baryonic_matter.star_halos
    ),
    sim.snapshot_end.baryonic_matter.star_lagrangian_regions == -1,
)
mask_star_from_outside_lr = np.logical_and(
    sim.snapshot_end.baryonic_matter.star_halos != -1,
    sim.snapshot_end.baryonic_matter.star_lagrangian_regions == -1,
)


plt.hist(
    dark_matter_data[1][sim.snapshot_end.dark_matter.halos != -1],
    label="Dark Matter (in halos)",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.hist(
    star_data[1][mask_star_same_lr],
    label="Same LR",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.hist(
    star_data[1][mask_star_different_lr],
    label="Different LR",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.hist(
    star_data[1][mask_star_outside_lr],
    label="Always Outside",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.hist(
    star_data[1][mask_star_from_outside_lr],
    label="From Outside",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.xlim(0, 25000)
plt.semilogy()
plt.legend(frameon=False)
plt.xlabel("Distance to original neighbour")

plt.savefig("distance_to_original_neighbour_based_on_in_lr_or_not_stars.pdf")

# Yes! This is perfect. This is the gas that traces a bunch more properties as it has formed stars - it is much less likely to have been processed and ejected. We see that gas that comes in from outside and forms stars, as well as gas from other LRs is the one that does the long-distance transfer.

# Now we need to consider combining the two plots; let's look at gas that is only in halos (at $z=0$) and how that is affected by the feedback.


mask_agn_halo = np.logical_and(
    agn_launches != 0, sim.snapshot_end.baryonic_matter.gas_halos != -1
)
mask_sf_halo = np.logical_and(
    sf_launches != 0, sim.snapshot_end.baryonic_matter.gas_halos != -1
)
mask_no_feedback = np.invert(np.logical_and(mask_agn, mask_sf))
mask_no_feedback_halo = np.logical_and(
    mask_no_feedback, sim.snapshot_end.baryonic_matter.gas_halos != -1
)


plt.hist(
    gas_data[1][mask_no_feedback_halo],
    label="No Feedback",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.hist(
    gas_data[1][mask_sf_halo],
    label="Stellar",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.hist(
    gas_data[1][mask_agn_halo],
    label="AGN",
    bins=100,
    alpha=0.5,
    range=(0, 25000),
    density=True,
    histtype="step",
)
plt.xlim(0, 25000)
plt.semilogy()
plt.legend(frameon=False)
plt.xlabel("Distance to original neighbour")

plt.savefig("distance_to_neighbour_based_on_how_launched_only_in_halos.pdf")

# We want to look on a particle-by-particle basis at the individual neighbours of dark matter particles at z=0. Thankfully, these should be pretty well organised -- we should be searching through the KDTree at the same time (i.e. we should have a list of particle indicies for the DM and Gas that are the same).
#
# The procedure is as follows:
#
# 1. We sort both the datasets by the ID of their neighbour
# 2. We only look at particles that have common neighbours (i.e. we only want particles that are
#    close to the same dark matter particle
# 3. We take the ratio between those two distances for that given DM particle
# 4. We can then plot this result.


unique_indicies_dm_gas = np.unique(dark_matter_data[3])[
    np.in1d(np.unique(dark_matter_data[3]), np.unique(gas_data[3]))
]

unique_indicies_dm_star = np.unique(dark_matter_data[3])[
    np.in1d(np.unique(dark_matter_data[3]), np.unique(star_data[3]))
]

ids_dm_g = np.take(sim.snapshot_end.dark_matter.ids, unique_indicies_dm_gas)
ids_gas = np.take(sim.snapshot_end.dark_matter.ids, unique_indicies_dm_gas)
ids_dm_s = np.take(sim.snapshot_end.dark_matter.ids, unique_indicies_dm_star)
ids_star = np.take(sim.snapshot_end.dark_matter.ids, unique_indicies_dm_star)

assert len(ids_gas) == len(ids_dm_g)
assert len(ids_star) == len(ids_dm_s)

indicies_dm_g = np.argsort(ids_dm_g)
indicies_gas = np.argsort(ids_gas)
indicies_dm_s = np.argsort(ids_dm_s)
indicies_star = np.argsort(ids_star)

radii_dm_g = dark_matter_data[1][indicies_dm_g]
radii_gas = gas_data[1][indicies_gas]
radii_dm_s = dark_matter_data[1][indicies_dm_s]
radii_star = star_data[1][indicies_star]


plt.semilogy()
plt.xlabel("$r_{i}$/$r_{DM}$ for common neighbours at $z=0$")
plt.hist(radii_gas / radii_dm_g, bins=256, range=(0, 2000), label="Gas", alpha=0.5)
plt.hist(radii_star / radii_dm_s, bins=256, range=(0, 2000), label="Star", alpha=0.5)
plt.xlim(0, 2000)
plt.legend(frameon=False)

plt.savefig("ratio_of_distances_compared_to_dark_matter.pdf")

# Now we'll take a look at the distribution of distances plot binned by halo mass.

masks = np.array([1e10, 1e11, 1e12, 1e13, 1e14]) / 1e10
bin_edges = [[x, y] for x, y in zip(masks[:-1], masks[1:])]


def format_number(number):
    """
    Formats a number into LaTeX style ready for plotting.
    """
    number *= 0.7e10
    float_str = "{0:.1g}".format(number)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def get_gas_mask_halo_mass_bin(bin_edge):
    mass_in_halos = sim.dark_matter_mass_in_halo
    halo_mask = np.logical_and(mass_in_halos > bin_edge[0], mass_in_halos < bin_edge[1])

    halo_ids = np.arange(len(halo_mask))[halo_mask]

    intersection = np.in1d(sim.snapshot_end.baryonic_matter.gas_halos, halo_ids)

    return intersection


for bin_edge in bin_edges:
    mask = get_gas_mask_halo_mass_bin(bin_edge)

    radii = gas_data[1][mask]

    plt.hist(
        radii,
        bins=100,
        range=(0, 25000),
        histtype="step",
        density=True,
        label=f"${format_number(bin_edge[0])}$-${format_number(bin_edge[1])}$ $M_\odot$",
    )

plt.semilogy()
plt.legend(frameon=False)
plt.xlim(0, 25000)

plt.savefig("gas_distance_metric_binned_by_halo_mass_only_in_halo.pdf")


def get_star_mask_halo_mass_bin(bin_edge):
    mass_in_halos = sim.dark_matter_mass_in_halo
    halo_mask = np.logical_and(mass_in_halos > bin_edge[0], mass_in_halos < bin_edge[1])

    halo_ids = np.arange(len(halo_mask))[halo_mask]

    intersection = np.in1d(sim.snapshot_end.baryonic_matter.star_halos, halo_ids)

    return intersection


for bin_edge in bin_edges:
    mask = get_star_mask_halo_mass_bin(bin_edge)

    radii = star_data[1][mask]

    plt.hist(
        radii,
        bins=100,
        range=(0, 25000),
        histtype="step",
        density=True,
        label=f"${format_number(bin_edge[0])}$-${format_number(bin_edge[1])}$ $M_\odot$",
    )

plt.semilogy()
plt.legend(frameon=False)
plt.xlim(0, 25000)

plt.savefig("star_distance_metric_binned_by_halo_mass_only_in_halo.pdf")


def get_dm_mask_halo_mass_bin(bin_edge):
    mass_in_halos = sim.dark_matter_mass_in_halo
    halo_mask = np.logical_and(mass_in_halos > bin_edge[0], mass_in_halos < bin_edge[1])

    halo_ids = np.arange(len(halo_mask))[halo_mask]

    intersection = np.in1d(sim.snapshot_end.dark_matter.halos, halo_ids)

    return intersection


for bin_edge in bin_edges:
    mask = get_dm_mask_halo_mass_bin(bin_edge)

    radii = dark_matter_data[1][mask]

    plt.hist(
        radii,
        bins=100,
        range=(0, 25000),
        histtype="step",
        density=True,
        label=f"${format_number(bin_edge[0])}$-${format_number(bin_edge[1])}$ $M_\odot$",
    )

plt.semilogy()
plt.legend(frameon=False)
plt.xlim(0, 25000)

plt.savefig("dm_distance_metric_binned_by_halo_mass_only_in_halo.pdf")
