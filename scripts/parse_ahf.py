"""
Parses the AHF .particles file. This also requires access to the original particle file; you will need to run it as follows: python3 parse_ahf.py <AHF particles file> <HDF5 file> <output pickle file> 
This will take a _very_ long time to run (of order hours when using a 512^3 sim).
"""

import ltcaesar as lt
import pickle
import numpy as np
import numpy_indexed as ni
import h5py
import sys
import pandas as pd

from tqdm import tqdm

input_filename = sys.argv[1]
particle_filename = sys.argv[2]
output_filename = sys.argv[3]

# First, we need to use Daniel's functions to parse the data...


def read_AHF_particles(file):
    """
    Daniel's old-school reading function for AHF halo data.

    This returns a dictionary with three values, "id", "PType", and "HaloID",
    which give the particle ID, particle type, and halo ID respectively.

    I don't want to touch this, hence why it is wrapped in read_particles(file).
    """
    f = open(file, "r")
    Nhalos = int(f.readline())
    f.close()
    df = pd.read_csv(file, delim_whitespace=True, names=["id", "Ptype"], header=0)
    Nlines = df["id"].size
    Ndata = Nlines - Nhalos
    data = {
        "id": np.zeros(Ndata, dtype="int64"),
        "Ptype": np.zeros(Ndata, dtype="int8"),
        "HaloID": np.zeros(Ndata, dtype="int32"),
    }
    jbeg = 0
    for i in range(0, Nhalos):
        jend = jbeg + df["id"][jbeg + i]
        data["id"][jbeg:jend] = df["id"][jbeg + i + 1 : jend + i + 1].values
        data["Ptype"][jbeg:jend] = df["Ptype"][jbeg + i + 1 : jend + i + 1].values
        data["HaloID"][jbeg:jend] = df["Ptype"][jbeg + i]
        jbeg = jend
    return data


def read_particles(AHF_file, particle_file):
    """
    Reads the particles from the AHF dataset and splits them into a more intelligable
    structure, organised in a similar way to the ones that are stored alongside the
    particles.

    The big problem here is being able to match up the particle IDs with the location
    in the array that they exist; we actually need to re-sort the HaloID's such that
    they line up exactly.
    """

    data = read_AHF_particles(AHF_file)

    switch = {"gas": 0, "dark_matter": 1, "stellar": 4}

    ids = {}

    with h5py.File(particle_file, "r") as handle:
        for name, particle_type in switch.items():
            full_particle_type = "PartType{}".format(particle_type)

            this_id_list = handle[full_particle_type]["ParticleIDs"][...]

            ids[name] = this_id_list

    # Now we can prepare the output arrays.
    output_data = {}

    for name, particle_type in switch.items():
        mask = data["Ptype"] == particle_type

        particle_ids = data["id"][mask]
        halo_ids = data["HaloID"][mask]

        # This finds the indicies where the two ID arrays match up
        indicies = ni.indices(ids[name], particle_ids)
        # We need to re-order the AHF data to be in the same order as
        # the actual HDF5 data. We can do that by using these indicies
        # as well as np.take.

        cleaned_halo_data = np.zeros_like(ids[name]) - 1
        cleaned_halo_data[indicies] = halo_ids

        this_data = {"HaloID": cleaned_halo_data, "ParticleIDs": ids[name]}

        output_data[name] = this_data

    return output_data


# Now we essentially have the exact same problem as the other reading script.

data = read_particles(input_filename, particle_filename)
full_output = {}

for particle_type in ["gas", "dark_matter", "stellar"]:
    this_data = data[particle_type]["HaloID"]

    # We are going to index this dictionary with the halo data.
    # Note we need to store the current _index_ in the halo array, as there
    # are some weird behaviours with caesar that we need to emulate.
    this_output = {}

    # Main processing loop
    for index, halo_id in enumerate(tqdm(this_data)):
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
    if (nstar != 0) and (ngas != 0) and (ndm != 0):
        halo_list.append(
            lt.halos.FakeHalo(
                dmlist=dmlist,
                ndm=ndm,
                glist=glist,
                ngas=ngas,
                slist=slist,
                nstar=nstar,
                GroupID=halo_id,
                center=None,
                rvir=None,
            )
        )

# Now that we've got them in there, we need to sort them by halo mass.
halo_list = sorted(halo_list, key=lambda x: x.ndm, reverse=True)
# Now assign them group IDs in that order
for GroupID, halo in enumerate(halo_list):
    halo.GroupID = GroupID

# Now, let's try to make our FakeCaesar object.

halo_catalogue = lt.halos.FakeCaesar(halos=halo_list, nhalos=len(halo_list))

pickle.dump(halo_catalogue, open(output_filename, "wb"))

exit(0)
