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

particle_filename = sys.argv[1]
halos_filename = sys.argv[2]
output_filename = sys.argv[3]

# Parse the input halos data to a usable format
print("Loading catalogue data")
with h5py.File(halos_filename, "r") as data:
    centers = np.array([data["Xc"][...], data["Yc"][...], data["Zc"][...]]).T

    radii = data["R_BN98"][...]

    ids = np.arange(0, len(radii))

# Read in particle co-ordinates only so that we can match them
print("Loading particle data")
with h5py.File(particle_filename, "r") as particles:
    try:
        # For Gadget runs, we need to re-correct the h-factors in the
        # Velociraptor outputs.
        hubble_param = float(particles["Header"].attrs["HubbleParam"])
    except:
        hubble_param = 1.0

    try:
        gas_coordinates = particles["PartType0/Coordinates"][...]
    except:
        gas_coordinates = None
    dm_coordinates = particles["PartType1/Coordinates"][...]
    try:
        star_coordinates = particles["PartType4/Coordinates"][...]
    except:
        star_coordinates = None

    boxsize = particles["Header"].attrs["BoxSize"]

# Now we need to build the trees for the neigbour search
print("Building KDTrees for periodic particle search")
try:
    gas_tree = KDTree(gas_coordinates, boxsize=boxsize)
except:
    gas_tree = None
dm_tree = KDTree(dm_coordinates, boxsize=boxsize)
try:
    star_tree = KDTree(star_coordinates, boxsize=boxsize)
except:
    star_tree = None

# Now we can run through each of the halos and do our job
halos = []

print("Velociraptor halo organiser:")

centers *= hubble_param
centers %= boxsize
radii *= hubble_param

print("Particle max position: {}".format(dm_coordinates.max()))
print("Halo center max position: {}".format(centers.max()))
print("Halo max radius: {}".format(radii.max()))

diff = 0

print("Querying trees and building FakeHalo objects")
for halo, (center, radius) in enumerate(zip(centers, radii)):
    dmlist = np.array(dm_tree.query_ball_point(x=center, r=radius))

    if dmlist.size <= 0:
        diff += 1
        continue

    try:
        glist = np.array(gas_tree.query_ball_point(x=center, r=radius))
    except:
        glist = np.array([], dtype=int)

    try:
        slist = np.array(star_tree.query_ball_point(x=center, r=radius))
    except:
        slist = np.array([], dtype=int)

    halos.append(
        lt.halos.FakeHalo(
            dmlist=dmlist,
            ndm=len(dmlist),
            glist=glist,
            ngas=len(glist),
            slist=slist,
            nstar=len(slist),
            GroupID=halo - diff,
            center=center,
            rvir=radius,
        )
    )

# Now that we've got them in there, we need to sort them by halo mass.
halos = sorted(halos, key=lambda x: x.ndm, reverse=True)
# Now assign them group IDs in that order
for GroupID, halo in enumerate(halos):
    halo.GroupID = GroupID


halo_catalogue = lt.FakeCaesar(halos=halos, nhalos=len(halos))

print("Writing to pickle file")
with open(output_filename, "wb") as file_handle:
    pickle.dump(halo_catalogue, file_handle)

exit(0)
