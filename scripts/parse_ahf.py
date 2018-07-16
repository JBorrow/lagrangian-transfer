"""
This script parses AHF-style halos so that they can be used with the FakeCaesar object.

It then pickles that object so that it can be loaded again in the future.

You should invoke it as follows:

    python3 parse_ahf.py <filename of hdf5 dataset> <filename of output pickle>

Note that AHF stores the individual particle/halo data in the snapshots themselves.

This script is probably super slow because it uses a bunch of dynamic memory allocations.
"""

import ltcaesar as lt
import pickle
import numpy as np
import h5py
import sys

from tqdm import tqdm

input_filename = sys.argv[1]
output_filename = sys.argv[2]

# First, load the data.

data = h5py.File(input_filename, "r")

particle_types = {
    "dark_matter": "PartType1",
    "gas": "PartType0",
    "stellar": "PartType1",
}

full_output = {}

for name, particle_type in particle_types.items():
    this_data = data[particle_type]["HaloID"][:50000]

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

    full_output[name] = this_output

# Now we have done the main procesisng loop, we need to fill the FakeCaesar object.

halo_list = []

# Because there may be some paritcle types missing from some halos, we need to do
# this kind of janky loop.

maximal_halo_id = max([max(x.keys()) for x in full_output.values()])

for halo_id in tqdm(range(maximal_halo_id + 1)):
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

# Now, let's try to make our FakeCaesar object.

halo_catalogue = lt.halos.FakeCaesar(halos=halo_list, nhalos=len(halo_list))

pickle.dump(halo_catalogue, open(output_filename, "wb"))

exit(0)
