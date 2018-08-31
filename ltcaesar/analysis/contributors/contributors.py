"""
Contains functions for finding the lagrangian regions that contributed to
a given halo.
"""

import numpy as np


def find_contributors_to_halo(halo, simulation):
    """
    Finds the contributors to a halo, given the simulation dataset.

    Returns a dictionary which has the following structure:
        
        output = {
            "gas" : {
                halo_id : number_of_gas_particles_from_lr,
            },
            "star" : {
                halo_id : number_of_star_particles_from_lr,
            },
            "dm" : {
                halo_id : number_of_dm_particles_from_lr,
            }
        }
    """

    catalogue_entry = simulation.snapshot_end.halo_catalogue.halos[halo]
    output = {}

    for ptype in ["gas", "star"]:
        mask = getattr(
            simulation.snapshot_end.baryonic_matter,
            "{}_halos".format(ptype)
        ) == halo
        lr_array = getattr(
            simulation.snapshot_end.baryonic_matter,
            "{}_lagrangian_regions".format(ptype),
        )

        relevant_lrs = lr_array[mask]

        halo_ids, counts = np.unique(relevant_lrs, return_counts=True)

        output[ptype] = {k:v for k, v in zip(halo_ids, counts)}

    # Unforunately DM _has_ to be special.

    mask = simulation.snapshot_end.dark_matter.halos == halo
    # Defined to be the same. Note we expect that this output is just one halo
    # but it's always nice to have the check.
    lr_array = simulation.snapshot_end.dark_matter.halos
    relevant_lrs = lr_array[mask]
    halo_ids, counts = np.unique(relevant_lrs, return_counts=True)

    output["dm"] = {k:v for k, v in zip(halo_ids, counts)}

    return output
