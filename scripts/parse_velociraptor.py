"""
Parses the velociraptor .particles file. This also requires access to the
original particle file; you will need to run it as follows: python3
parse_velociraptor.py <AHF particles file> <HDF5 file> <output pickle file>
"""

import ltcaesar as lt
import pickle
import numpy as np
import h5py
import sys

from tqdm import tqdm

input_filename = sys.argv[1]
particle_filename = sys.argv[2]
output_filename = sys.argv[3]


def read_particles(velociraptor_file, particle_file):
    """
    Reads the particles from the velociraptor dataset and splits them into a
    more intelligable structure, organised in a similar way to the ones that
    are stored alongside the particles.
    
    The big problem here is being able to match up the particle IDs with the
    location in the array that they exist; we actually need to re-sort the
    HaloID's such that they line up exactly.

    Note that you will have to pre-process the velociraptor data with the
    scripts that are provided in

    github.com/jborrow/simba-velociraptor-tools
    """

    switch = {"gas": 0, "dark_matter": 1, "stellar": 4}

    output_data = {}

    with h5py.File(velociraptor_file, "r") as file:
        for name, number in switch.items():
            try:
                output_data[name] = file[f"PartType{number}/GroupID"][...]
            except:
                pass

    return output_data


# Now we essentially have the exact same problem as the other reading script.

data = read_particles(input_filename, particle_filename)
full_output = {}

for particle_type in ["gas", "dark_matter", "stellar"]:
    try:
        this_data = data[particle_type]
    except:
        continue

    # We are going to index this dictionary with the halo data.
    # Note we need to store the current _index_ in the halo array, as there
    # are some weird behaviours with caesar that we need to emulate.
    this_output = {}

    # Main processing loop
    for index, halo_id in enumerate(tqdm(this_data)):
        if halo_id != -1:
            try:
                this_output[halo_id].append(index)
            except KeyError:
                this_output[halo_id] = [index]

    full_output[particle_type] = this_output


# Because there may be some paritcle types missing from some halos, we need to do
# this kind of janky loop.

del data

# Halo IDs are defined by the DM
halo_ids = np.array([int(x) for x in full_output["dark_matter"].keys()])
halo_list = []

for halo_id in tqdm(halo_ids):
    try:
        dmlist = np.array(full_output["dark_matter"][halo_id], dtype=int)
        ndm = len(dmlist)
    except KeyError:
        dmlist = np.array([], dtype=int)
        ndm = 0

    try:
        glist = np.array(full_output["gas"][halo_id], dtype=int)
        ngas = len(glist)
    except KeyError:
        glist = np.array([], dtype=int)
        ngas = 0

    try:
        slist = np.array(full_output["stellar"][halo_id], dtype=int)
        nstar = len(slist)
    except KeyError:
        slist = np.array([], dtype=int)
        nstar = 0

    # Now fill the object
    
    # With AHF, there are a _lot_ of empty halos. This could be optimized quite
    # heavily in the future, but right now I will just leave this.
    # Note: you could pre-build a list of only the occupied halos by running
    # through all of the halo ids.
    if (ngas != 0) and (ndm != 0):
        halo_list.append(
            lt.halos.FakeHalo(
                dmlist=dmlist,
                ndm=ndm,
                glist=glist,
                ngas=ngas,
                slist=slist,
                nstar=nstar,
                GroupID=halo_id,
            )
        )

# Now that we've got them in there, we need to sort them by halo mass.
halo_list = sorted(halo_list, key=lambda x: x.ndm, reverse=True)
# Now assign them group IDs in that order
for GroupID, halo in enumerate(halo_list): halo.GroupID = GroupID

# Now, let's try to make our FakeCaesar object.

halo_catalogue = lt.halos.FakeCaesar(halos=halo_list, nhalos=len(halo_list))

pickle.dump(halo_catalogue, open(output_filename, "wb"))

exit(0)
