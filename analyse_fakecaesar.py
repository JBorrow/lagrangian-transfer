"""
This script aims to do everything that lagtrans does but with the caesar
toolkit rather than AHF.

Please invoke as follows:

    python3 <name of snapshot file (ic)> <name of snapshot file (z=0) <name of fakecaesar pickle>

Note that to install caesar for python3 you will need to make a few minor
changes. You will need to remove the hg info that is put into the caesar
snapshot, and you will need to run 2to3 on the caesar source code (some minor
differences, such as how to get values out of dictionaries, cause this).
"""

import ltcaesar as lt

import sys
import pickle

snapshot_filename_ini = str(sys.argv[1])
snapshot_filename_end = str(sys.argv[2])
fake_caesar_filename = str(sys.argv[3])

# Bad choice of variable name, sorry -- for consistency! This is actually a loaded
# FakeCaesar object.
caesar_filename = pickle.load(open(fake_caesar_filename, "rb"))

# We need to make sure that the halos are contiguous.
for index, halo in enumerate(caesar_filename.halos):
    halo.GroupID = index

print([halo.GroupID for halo in caesar_filename.halos])

# Load the data using our library

simulation_ini = lt.Snapshot(snapshot_filename_ini)
# Get the max initial gas ID to truncate by
truncate_id = simulation_ini.baryonic_matter.gas_ids.max()
simulation_end = lt.Snapshot(
    snapshot_filename_end, caesar_filename, truncate_ids=truncate_id
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
