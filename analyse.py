"""
This script aims to do everything that lagtrans does but with the caesar
toolkit rather than AHF.

Please invoke as follows:

    python3 <name of snapshot file (ic)> <name of snapshot file (z=0) <name of caesar file>

Note that to install caesar for python3 you will need to make a few minor
changes. You will need to remove the hg info that is put into the caesar
snapshot, and you will need to run 2to3 on the caesar source code (some minor
differences, such as how to get values out of dictionaries, cause this).
"""

import ltcaesar as lt

import argparse as ap

PARSER = ap.ArgumentParser(
    description="""
    Analysis script for caesar-based halo data for LTCaesar. Running
    this script will write a file, lt_outputs.hdf5, to your current
    working directory that can be then re-loaded with LTCaesar to
    reconstruct (and re-read) all relevant data. Particle data and halo
    catalogue data will be re-read from the file paths given; it is
    best therefore to give these file paths as absolute rather than
    relative.
    
    Example usage:
    
    python3 analyse.py -i initial_filename.hdf5 -f final_filename.hdf5
                       -c catalogue_filename.hdf5 -t 0 -y 1
    """
)

PARSER.add_argument(
    "-i",
    "--initial",
    help="""
    Initial conditions file for your simulation. Usually a HDF5 file.
    """,
    required=True,
)

PARSER.add_argument(
    "-f",
    "--final",
    help="""
    Final conditions file for your simulation, usually the z=0 snapshot.
    """,
    required=True,
)

PARSER.add_argument(
    "-c",
    "--catalogue",
    help="""
    Halo catalogue filename. If this is from caesar, you are done;
    otherwise please see the documentation for -o/--otherhalofinder.
    """,
    required=True,
)

PARSER.add_argument(
    "-t",
    "--notrunc",
    help="""
    If set to a truthy value, this parameter prevents all ParticleIDs from
    being truncated. This is useful in the case where you do not trust the
    updates internally to your simulation code to correctly preserve the
    underlying ParticleID, or if your simulation never changes the ParticleIDs
    from the initial conditions. Otherwise, all baryonic particle IDs are
    taken modulo the largest ID in the initial conditions.
    """,
    required=False,
    default=False,
)

PARSER.add_argument(
    "-y",
    "--yt",
    help="""
    If set to a truthy value, the particle data is read using yt as a
    wrapper, rather than the thinner h5py wrapper usually used in the code.
    This is useful if you, for example, have a single snapshot made up of
    multiple HDF5 files, as this is not handled by LTCaesar gracefully.
    """,
    required=False,
    default=False,
)

PARSER.add_argument(
    "-o",
    "--otherhalofinder",
    help="""
    Use another halo finder; i.e. using a FakeCaesar catalogue generated
    through the use of the scripts available in the repository. In that
    case, please point -c to the file containing the information for the
    FakeCaesar catalogue.
    """,
    required=False,
    default=False,
)

PARSER.add_argument(
    "-l",
    "--lagrangianregions",
    help="""
    Perform a nearest-neighbour search for -l neighbours and set the 
    lagrangian region ID to that of the lowest mass halo in the group.
    This is usedful for "filling out" holes.
    """,
    required=False,
    default=1
)

PARSER.add_argument(
    "-a",
    "--aboveid",
    help="""
    Cut away halos above this ID and ensure that their particles have their
    halo ID set to -1. This is useful to ensure less spurious transfer from
    "tiny" halos.
    """,
    required=False,
    default=None
)


if __name__ == "__main__":
    ARGS = vars(PARSER.parse_args())

    snapshot_filename_ini = str(ARGS["initial"])
    snapshot_filename_end = str(ARGS["final"])
    caesar_filename = str(ARGS["catalogue"])
    no_trunc = bool(ARGS["notrunc"])
    use_yt = bool(int(ARGS["yt"]))
    other_halo_finder = bool(int(ARGS["otherhalofinder"]))
    lagrangian_regions = int(ARGS["lagrangianregions"])
    if ARGS["aboveid"] is None:
        above_id = None
    else:
        above_id = int(ARGS["aboveid"])

    # Print a summary of code options chosen.

    print(
        "Code was initialised with the following values:\n",
        "\t Initial filename: {}\n".format(snapshot_filename_ini),
        "\t Final filename: {}\n".format(snapshot_filename_end),
        "\t Caesar filename: {}\n".format(caesar_filename),
        "\t No truncation: {}\n".format(no_trunc),
        "\t Use yt to load data: {}\n".format(use_yt),
        "\t Use another halo finder: {}\n".format(other_halo_finder),
        "\t Smoothing LRs with a neighbour search of {} neighbours\n".format(lagrangian_regions),
        "\t Cutting halos above ID: {}.".format(above_id)
    )

    # Change out the caesar filename for the FakeCaesar catalogue data
    # if required

    if other_halo_finder:
        # Bad choice of variable name, sorry -- for consistency! This is
        # a FakeCaesar object.
        import pickle

        caesar_filename = pickle.load(open(caesar_filename, "rb"))

        # We need to make sure that the halos are contiguous.
        # We'll also make sure that the "maximum" halo id gets converted to -1
        # as this appears to be the AHF convention.

        max_halo_id = max([halo.GroupID for halo in caesar_filename.halos])

        for index, halo in enumerate(caesar_filename.halos):
            if halo.GroupID != max_halo_id:
                halo.GroupID = index
            else:
                halo.GroupID = -1

    # Load the data using our library

    simulation_ini = lt.Snapshot(snapshot_filename_ini, load_using_yt=use_yt)
    # Get the max initial gas ID to truncate by
    if no_trunc:
        truncate_id = None
        print("Note: leaving IDs un-truncated")
    else:
        truncate_id = simulation_ini.baryonic_matter.gas_ids.max()

    simulation_end = lt.Snapshot(
        snapshot_filename_end,
        caesar_filename,
        truncate_ids=truncate_id,
        load_using_yt=use_yt,
        neighbours_for_lagrangian_regions=lagrangian_regions,
        cut_halos_above_id=above_id
    )

    print("Running the simulation class")
    simulation = lt.Simulation(simulation_ini, simulation_end)

    simulation.prepare_analysis_arrays()
    print("Running gas analysis")
    simulation.run_gas_analysis()
    print("Running star analysis")
    simulation.run_star_analysis()
    print("Running DM analysis")
    simulation.run_dark_matter_analysis()

    simulation.write_reduced_data("lagrangian_transfer.txt")

    print("Writing to HDF5 file")
    lt.write_data_to_file("lt_outputs.hdf5", simulation)

    exit(0)
