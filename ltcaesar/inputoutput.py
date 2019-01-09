"""
Writing functions for the LTCaesar library.

These will serialise your data. Without them (i.e. if you just try to Pickle
your data), you will end up with a bunch of HDF5 errors due to file-handle
issues.
"""

import h5py
import time
import numpy as np

from .objects import Simulation, Snapshot
from .halos import FakeCaesar


class FakeSimulation(object):
    """
    A "fake" simulation object that has the same API (provided by namedtuples).

    Treat this as read only (duh, they are named tuples).
    """

    def __init__(self, handle):
        """
        handle is the H5py file handle.
        """

        self.handle = handle

        self.grab_data()

        return

    def grab_data(self):
        """
        This is a bit of a nightmare function. It tries to grab relevant
        handles from all over the place.
        """

        # First, let's grab some meta information.

        self.created_by = self.handle["header"].attrs["created_by"]
        self.time_created = self.handle["header"].attrs["time_created"]

        # Read in some global simulation properties

        self.lagrangian_regions = self.handle["simulation/lagrangian_regions"][...]

        # Now read in the reduced data.

        data_types = ["dark_matter", "gas", "stellar"]

        attributes = [
            "mass_in_halo",
            "mass_in_lagrangian",
            "mass_in_halo_from_lagrangian",
            "mass_in_halo_from_outside_lagrangian",
            "mass_in_halo_from_other_lagrangian",
            "mass_outside_halo_from_lagrangian",
            "mass_in_other_halos_from_lagrangian",
        ]

        for data_type in data_types:
            this_sim_data = self.handle["simulation/{}".format(data_type)]

            for attribute in attributes:
                setattr(
                    self,
                    "{}_{}".format(data_type, attribute),
                    this_sim_data[attribute][...],
                )

        # Load in the particle data like we're a fresh prince

        for snap in ["snapshot_ini", "snapshot_end"]:
            # Set this to false in case we don't set it
            read_reduced_data_fake_caesar = False
            truncate_ids = self.handle[snap].attrs["truncate_ids"]

            if truncate_ids == -1:
                truncate_ids = None
            else:
                truncate_ids = int(truncate_ids)

            snapshot_filename = (
                self.handle[snap].attrs["snapshot_filename"].decode("utf-8")
            )
            catalogue_filename = (
                self.handle[snap].attrs["catalogue_filename"].decode("utf-8")
            )
            load_using_yt = self.handle[snap].attrs["load_using_yt"]

            # Because we do the np.string_ conversion when writing.
            if catalogue_filename == "None":
                catalogue_filename = None
            elif catalogue_filename == "FakeCaesar":
                # We need to read reduced data _later_
                # This makes the reading logic a little complex, sorry about that.
                catalogue_filename = None
                read_reduced_data_fake_caesar = True

            snapshot = Snapshot(
                snapshot_filename, catalogue_filename, truncate_ids, load_using_yt
            )

            if read_reduced_data_fake_caesar:
                # Read the reduced data and slot it into the snapshot.

                dark_matter_halos = self.handle[snap]["fake_caesar"][
                    "dark_matter_halos"
                ][...]
                snapshot.dark_matter.halos = dark_matter_halos

                # Gas stellar, and bh halo data.
                for halo_type in ["gas", "star", "bh"]:
                    try:
                        this_data = self.handle[snap]["fake_caesar"][
                            "{}_halos".format(halo_type)
                        ][...]
                    except KeyError:
                        # Must not exist. Stick in an empty array!
                        this_data = np.array([])

                    setattr(
                        snapshot.baryonic_matter,
                        "{}_halos".format(halo_type),
                        this_data,
                    )

            setattr(self, snap, snapshot)

        # Finally, load in the lagrangian regions corresponding to each particle

        for lr in ["gas", "star"]:
            lr_name = "{}_lagrangian_regions".format(lr)

            data = self.handle["snapshot_end/baryonic_matter/{}".format(lr_name)][...]

            setattr(self.snapshot_end.baryonic_matter, lr_name, data)

        self.snapshot_end.dark_matter.lagrangian_regions = self.handle[
            "snapshot_end/dark_matter/lagrangian_regions"
        ][...]

        return


def write_data_to_file(filename, simulation: Simulation):
    """
    Writes the relevant data in simulation to a HDF5 file called "filename".

    Note that we make no claims that this is portable -- if you move the original
    files, we will loose them and you will need to fix up the files.
    """

    with h5py.File(filename, "w") as handle:
        # You got a lot of writing to do...

        header = handle.create_group("header")

        header.attrs["created_by"] = "LTCaesar"
        header.attrs["time_created"] = time.time()

        # Now write out the actual Simulation properties

        sim = handle.create_group("simulation")
        sim.create_dataset("lagrangian_regions", data=simulation.lagrangian_regions)

        # Now write out the reduced data

        data_types = ["dark_matter", "gas", "stellar"]

        attributes = [
            "mass_in_halo",
            "mass_in_lagrangian",
            "mass_in_halo_from_lagrangian",
            "mass_in_halo_from_outside_lagrangian",
            "mass_in_halo_from_other_lagrangian",
            "mass_outside_halo_from_lagrangian",
            "mass_in_other_halos_from_lagrangian",
        ]

        for data_type in data_types:
            this_sim_data = sim.create_group(data_type)

            for attribute in attributes:
                this_sim_data.create_dataset(
                    attribute,
                    data=getattr(simulation, "{}_{}".format(data_type, attribute)),
                )

        # Write out some more meta information so that the particles can be re-constructed
        # from the files later. Note we do this to conserve space (otherwise this file would
        # be huge), and sacrifice portability as a result.

        for snap in ["snapshot_ini", "snapshot_end"]:
            this_snapshot = getattr(simulation, snap)

            snap_group = handle.create_group(snap)
            snap_group.attrs.create(
                "snapshot_filename", np.string_(this_snapshot.snapshot_filename)
            )
            snap_group.attrs.create("load_using_yt", this_snapshot.load_using_yt)

            if not isinstance(this_snapshot.catalogue_filename, FakeCaesar):
                snap_group.attrs.create(
                    "catalogue_filename", np.string_(this_snapshot.catalogue_filename)
                )
            else:
                # Now we need to deal with the i/o for our fake halo catalogue.

                snap_group.attrs.create("catalogue_filename", np.string_("FakeCaesar"))

                # There are a number of properties that we need to write to a dataset.
                # Note that these are not actually used to re-create the FakeCaesar
                # object, but are instead just used to re-populate the reduced arrays
                # that we need.

                fake_caesar_group = snap_group.create_group("fake_caesar")

                # Dark matter halos
                fake_caesar_group.create_dataset(
                    "dark_matter_halos", data=this_snapshot.dark_matter.halos
                )

                # Gas stellar, and bh halo data.
                for halo_type in ["gas", "star", "bh"]:
                    try:
                        fake_caesar_group.create_dataset(
                            "{}_halos".format(halo_type),
                            data=getattr(
                                this_snapshot.baryonic_matter,
                                "{}_halos".format(halo_type),
                            ),
                        )
                    except TypeError:
                        # The array must be empty, let's just not write it and deal with that
                        # in the reading.
                        pass

            # H5py does _not_ like NoneTypes or long integers being stored as strings. We
            # have to sort this out by setting truncate_ids = -1 if we have it as "None" as this
            # is a) compatible with the storage datatype, and b) equivalent.

            truncate_ids = this_snapshot.truncate_ids

            if truncate_ids is None:
                truncate_ids = -1

            snap_group.attrs.create("truncate_ids", truncate_ids)

        # Now only for the end snapshot

        baryonic_group = handle["snapshot_end"].create_group("baryonic_matter")

        baryonic_group.create_dataset(
            "gas_lagrangian_regions",
            data=simulation.snapshot_end.baryonic_matter.gas_lagrangian_regions,
        )
        baryonic_group.create_dataset(
            "star_lagrangian_regions",
            data=simulation.snapshot_end.baryonic_matter.star_lagrangian_regions,
        )

        dm_group = handle["snapshot_end"].create_group("dark_matter")

        dm_group.create_dataset(
            "lagrangian_regions",
            data=simulation.snapshot_end.dark_matter.lagrangian_regions,
        )

    return


def read_data_from_file(filename):
    """
    Fills a Simulation-like object (with the same API) with the data loaded
    from the HDF5 file in filename.

    This object is strictly _not_ bitwise comparable (or even type-comparable) with
    the true "simulation" class. Be careful.
    """

    with h5py.File(filename, "r") as handle:
        return FakeSimulation(handle)
