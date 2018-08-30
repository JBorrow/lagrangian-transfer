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

    catalogue_entry = simulation.snapshot_end.halo_catalogue[halo]
    output = {}

    for ptype in ["gas", "star"]:
        list_name = "{}list".format(ptype[0])
        index_list = getattr(catalogue_entry, list_name)
        lr_array = getattr(
            simulation.snapshot_end.baryonic_matter,
            "{}_lagrangian_regions".format(ptype),
        )

        relevant_lrs = lr_array[index_list]

        halo_ids, counts = np.unique(relevant_lrs, return_counts=True)

        output[ptype] = dict(halo_ids, counts)

    # Unforunately DM _has_ to be special.

    index_list = catalogue_entry.dmlist
    lr_array = simulation.snapshot_end.dark_matter.lagrangian_regions
    relevant_lrs = lr_array[index_list]
    halo_ids, counts = np.unique(relevant_lrs, return_counts=True)

    output["dm"] = dict(halo_ids, counts)

    return output
