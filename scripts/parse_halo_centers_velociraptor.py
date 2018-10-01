"""
This file parses the halo centers and virial radii to a FakeCaesar halo
catalogue, given that these halo centers and virial radii are presented in
the same format as velociraptor. If not, you can use this script as a guide
on how to do that yourself.

Please invoke it as follows

python3 parse_halo_centers_velociraptor.py <name of HDF5 particle file> \
                                           <name of .propertties file from VR> \
                                           <output filename.pickle>
"""

import numpy as np
import pickle
import h5py
import sys

from scipy.spatial import cKDTree as KDTree

import ltcaesar as lt

particle_filename = sys.argv[0]
halos_filename = sys.argv[1]
output_filename = sys.argv[2]

# Parse the input halos data to a usable format
print("Loading catalogue data")
with h5py.File(halos_filename) as data:
    centers = np.array(
        [
            data["Xc"][...],
            data["Yc"][...],
            data["Zc"][...]
        ]
    )

    radii = data["Rvir"][...]

    ids = np.arange(0, len(radii))

# Read in particle co-ordinates only so that we can match them
print("Loading particle data")
with h5py.File(particle_filename, "r") as particles:
    gas_coordinates = particles["PartType0/Coordinates"][...]
    dm_coordinates = particles["PartType1/Coordinates"][...]
    star_coordinates = particles["PartType4/Coordinates"][...]

    boxsize = particles["Header"].attrs["BoxSize"]

# Now we need to build the trees for the neigbour search
print("Building KDTrees for periodic particle search")
gas_tree = KDTree(gas_coordinates, boxsize=boxsize)
dm_tree = KDTree(dm_coordinates, boxsize=boxsize)
star_tree = KDTree(gas_coordinates, boxsize=boxsize)

# Now we can run through each of the halos and do our job
halos = []

print("Querying trees and building FakeHalo objects")
for halo, (center, radius) in enumerate(zip(centers, radii)):
    dmlist = np.array(
        dm_tree.query_ball_point(x=center, r=radius, n_jobs=-1)
    )
    glist = np.array(
        gas_tree.query_ball_point(x=center, r=radius, n_jobs=-1)
    )
    slist = np.array(
        star_tree.query_ball_point(x=center, r=radius, n_jobs=-1)
    )

    halos.append(
        lt.halos.FakeHalo(
            dmlist=dmlist,
            ndm=len(dmlist),
            glist=glist,
            ngas=len(glist),
            slist=slist,
            nstar=len(slist),
            GroupID=halo,
        )
    )

halo_catalogue = lt.FakeCaesar(halos=halos, nhalos=len(halos))

print("Writing to pickle file")
with open(output_filename, "rb") as file_handle:
    pickle.dump(halo_catalogue, file_handle)

exit(0)