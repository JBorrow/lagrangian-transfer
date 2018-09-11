"""
This file contains functions that make modifications to the
Simulation object to allow for distances larger than 1 rvir
to be included in the definition of which particles belong to
a given lagrangian region, but not to include those in the halo
itself.
"""

import numpy as np

from .change_virial_radius import find_all_halo_centers

from scipy.spatial import cKDTree as KDTree


def change_rvir_of_lagrangian_regions_only(simulation, factor=1.2):
    """
    Updates a simulation with a higehr virial radius definition
    for the lagrangian regions (all halos have their capture
    radius increased by factor).
    """

    # First find the halo centers and radii
    centers, radii = find_all_halo_centers(
        simulation.snapshot_end.dark_matter.halos,
        simulation.snapshot_end.dark_matter.coordinates.T,
        boxsize=simulation.snapshot_end.header["BoxSize"],
    )

    # Increase the radii
    radii *= factor

    # Now find the particles that lie within that radii
    # Need to build a tree for the dark matter
    tree = KDTree(simulation.snapshot_end.dark_matter.coordinates)

    # Now need to recover the actual halo numbers
    cut_halos = simulation.snapshot_end.dark_matter.halos[
        simulation.snapshot_end.dark_matter.halos != -1
    ]
    halos = np.unique(cut_halos)

    new_lagrangian_regions = np.empty_like(
        simulation.snapshot_end.dark_matter.halos, dtype=int
    )
    new_lagrangian_regions[...] = -1

    for halo, center, radius in zip(halos, centers, radii):
        dmlist = tree.query_ball_point(x=center, r=radius, n_jobs=-1)

        new_lagrangian_regions[dmlist] = halo

    # Set this array on the DM particles
    simulation.snapshot_end.dark_matter.lagrangian_regions = new_lagrangian_regions

    # Re-identify the lagrangian regions using the original code
    simulation.identify_lagrangian_regions()

    # Now we have to re-run all of the analysis
    simulation.run_gas_analysis()
    simulation.run_star_analysis()
    simulation.run_dark_matter_analysis()

    # Return the now changed object
    return simulation
