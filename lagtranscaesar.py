"""
Replacement for the LagTrans library with Caesar.

This library was originally written by Daniel Angles-Alcazar to use the AHF
halo-finder library, but it was re-written in early July 2018 at the Kavli
Summer Program in Astrophysics by Josh Borrow (joshua.borrow@durham.ac.uk).

Full documentation for the library can be found in README.md.
"""

import caesar
import dill as pickle
import h5py
import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


from scipy.spatial import cKDTree as KDTree
from collections import namedtuple


class DMParticles(object):
    """
    Reads the DM particles, and sorts them by their ParticleID.
    """

    def __init__(self, particles, halo_catalogue=None):
        """
        Takes the particles in PartType1 (the reference to it from h5py) and the
        halos catalogue from caesar.

        Note that this should be passed ceasar.load().halos for halo_catalogue,
        and file["/PartType1/"] for the particles.
        """
        self.particles = particles
        self.halo_catalogue = halo_catalogue

        self.n_parts = len(particles["ParticleIDs"])

        if halo_catalogue is not None:
            self.halos = self.get_all_particle_references()

        # We may in the future need to ensure only the particles in the halos
        # have their data read.
        self.ids = self.read_array("ParticleIDs")
        self.masses = self.read_array("Masses")
        self.coordinates = self.read_array("Coordinates")

        self.sort_data()

        return

    def get_all_particle_references(self):
        """
        Gets an nparticle long array that denotes which halo all of the dark matter
        particles belongs to. Particles that do not reside in halos have their
        value set to -1.
        """

        # Allocate our final output array
        halos = np.empty(self.n_parts, dtype=int)
        halos[...] = -1  # Default value for all particles _not_ in halos

        # Grab all references
        particles_in_halos = [halo.dmlist for halo in self.halo_catalogue]
        copied_halos = [
            np.repeat(halo.GroupID, halo.ndm) for halo in self.halo_catalogue
        ]

        flattened_particles = np.concatenate(particles_in_halos)
        del particles_in_halos
        flattened_halos = np.concatenate(copied_halos)
        del copied_halos

        halos.put(flattened_particles, flattened_halos)

        return halos

    def read_array(self, name: str):
        """
        Read an array from the particle file.
        """

        return self.particles[name][...]

    def sort_data(self):
        """
        Sorts the data by ParticleID.
        """

        indicies = np.argsort(self.ids)

        # Actually perform data transformation.

        try:
            self.halos = self.halos[indicies]
        except AttributeError:
            # We must not have any halos. Oh well.
            pass

        self.ids = self.ids[indicies]
        self.masses = self.masses[indicies]
        self.coordinates = self.coordinates[indicies]

        return


class BaryonicParticles(object):
    """
    Holder class for the baryonic particles. This also does the re-jigging to try
    and figure out where the gas particles went (they formed stars or BHs).
    """

    def __init__(
        self,
        gas_particles,
        star_particles,
        bh_particles,
        halo_catalogue=None,
        truncate_ids=None,
    ):
        """
        Takes the particles in PatyType[x] (the reference to it from h5py) and
        the halos catalogue from caeasar.

        truncate_ids should be an integer above which the ParticleIDs are all
        truncated. This is helpful as in some simulation codes (e.g. Mufasa)
        star-forming particles have their higher-up bits played with.
        """

        self.gas_particles = gas_particles
        self.star_particles = star_particles
        self.bh_particles = bh_particles
        self.halo_catalogue = halo_catalogue
        self.truncate_ids = truncate_ids

        # Now we read a bunch of particle properties.
        self.n_gas_parts = len(gas_particles["ParticleIDs"])
        self.gas_ids = self.gas_particles["ParticleIDs"][...]
        self.gas_masses = self.gas_particles["Masses"][...]
        self.gas_coordinates = self.gas_particles["Coordinates"][...]

        # Try to load stars and BHs -- but they might not be there (ics)!
        try:
            self.n_star_parts = len(star_particles["ParticleIDs"])
            self.star_ids = self.star_particles["ParticleIDs"][...]
            self.star_masses = self.star_particles["Masses"][...]
            self.star_coordinates = self.star_particles["Coordinates"][...]
        except IndexError:
            self.n_star_parts = 0
            self.star_ids = np.array([])
            self.star_masses = np.array([])
            self.star_coordinates = np.array([[] * 3])

        try:
            self.n_bh_parts = len(bh_particles["ParticleIDs"])
            self.bh_ids = self.bh_particles["ParticleIDs"][...]
            self.bh_masses = self.bh_particles["BH_Mass"][...]
            self.bh_coordinates = self.bh_particles["Coordinates"][...]
        except IndexError:
            self.n_bh_parts = 0
            self.bh_ids = np.array([])
            self.bh_masses = np.array([])
            self.bh_coordinates = np.array([[] * 3])

        if halo_catalogue is not None:
            self.gas_halos, self.star_halos, self.bh_halos = (
                self.get_all_particle_references()
            )

        # This function sorts by ID to make matching easier.
        self.sort_data()

        return

    def sort_data(self):
        """
        Sorts the data by ParticleID.
        """

        if self.truncate_ids is not None:
            # We want to sort based on the NON-SF IDs as these are the ones that
            # will be matched later on
            gas_indicies = np.argsort(self.gas_ids % (self.truncate_ids + 1))
            star_indicies = np.argsort(self.star_ids % (self.truncate_ids + 1))
            bh_indicies = np.argsort(self.bh_ids % (self.truncate_ids + 1))
        else:
            gas_indicies = np.argsort(self.gas_ids)
            star_indicies = np.argsort(self.star_ids)
            bh_indicies = np.argsort(self.bh_ids)

        # Actually perform data transformation.

        try:
            self.gas_halos = self.gas_halos[gas_indicies]
            self.star_halos = self.star_halos[star_indicies]
            try:
                self.bh_halos = self.bh_halos[bh_indicies]
            except:
                # Must be empty
                pass
        except AttributeError:
            # We must not have the halo catalogue for this one
            pass

        self.gas_ids = self.gas_ids[gas_indicies]
        self.star_ids = self.star_ids[star_indicies]
        self.bh_ids = self.bh_ids[bh_indicies]

        self.gas_masses = self.gas_masses[gas_indicies]
        self.star_masses = self.star_masses[star_indicies]
        self.bh_masses = self.bh_masses[bh_indicies]

        self.gas_coordinates = self.gas_coordinates[gas_indicies]
        self.star_coordinates = self.star_coordinates[star_indicies]
        self.bh_coordinates = self.bh_coordinates[bh_indicies]

        return

    def get_all_particle_references(self):
        """
        Gets three nparticle long array that denotes which halo all of the
        particles belongs to. Particles that do not reside in halos have their
        value set to -1.

        Yes, this function breaks DRY, but that's for memory efficiency.
        """

        # Allocate our final output array
        gas_halos = np.empty(self.n_gas_parts, dtype=int)
        gas_halos[...] = -1  # Default value for all particles _not_ in halos

        # Grab all references
        particles_in_halos = [galaxy.glist for galaxy in self.halo_catalogue]
        copied_halos = [
            np.repeat(galaxy.halo.GroupID, galaxy.ngas)
            for galaxy in self.halo_catalogue
        ]

        flattened_particles = np.concatenate(particles_in_halos)
        del particles_in_halos
        flattened_halos = np.concatenate(copied_halos)
        del copied_halos

        gas_halos.put(flattened_particles, flattened_halos)

        del flattened_particles
        del flattened_halos

        # Now for stars
        star_halos = np.empty(self.n_star_parts, dtype=int)
        star_halos[...] = -1  # Default value for all particles _not_ in halos

        # Grab all references
        particles_in_halos = [galaxy.slist for galaxy in self.halo_catalogue]
        copied_halos = [
            np.repeat(galaxy.halo.GroupID, galaxy.nstar)
            for galaxy in self.halo_catalogue
        ]

        flattened_particles = np.concatenate(particles_in_halos)
        del particles_in_halos
        flattened_halos = np.concatenate(copied_halos)
        del copied_halos

        star_halos.put(flattened_particles, flattened_halos)

        del flattened_particles
        del flattened_halos

        # Now for BHs
        # This is not currently implemented
        bh_halos = None

        return gas_halos, star_halos, bh_halos

    def parse_lagrangian_regions(self, ids: np.ndarray, lagrangian_regions: np.ndarray):
        """
        Parse the Lagrangian regions so that they "line up" with the z=0 particles.
        We must do this as some particles have been changed to become stars.
        """

        # This could be quite slow. It is only a linear pass over the particles
        # though, as we can assume the arrays are sorted by ID.

        gas_lagrangian_regions = np.repeat(-1, len(self.gas_ids))
        star_lagrangian_regions = np.repeat(-1, len(self.star_ids))

        gas_current_index = 0
        star_current_index = 0

        touched_last_gas, touched_last_star = (False, False)

        try:
            gas_current_id = self.gas_ids[gas_current_index]
        except IndexError:
            raise IndexError("lagtranscaesar: Unable to index the gas ID array.")

        try:
            star_current_id = self.star_ids[star_current_index]
        except IndexError:
            print("No star particles found. Consider checking this.")
            gas_lagrangian_regions = lagrangian_regions

        truncate = self.truncate_ids + 1

        # The _first_ id needs truncating just in case.
        if self.truncate_ids is not None:
            truncated_gas_id = gas_current_id % truncate
            truncated_star_id = star_current_id % truncate
        else:
            truncated_gas_id = gas_current_id
            truncated_star_id = star_current_id

        # It is unclear if this fully works at the moment. We really need to parse
        # the stellar IDs first before doing this. We may also break things by sorting
        # in that way, but I would hope not as the gas particles are still contiguous.

        for lr, particle_id in zip(tqdm(lagrangian_regions, desc="Parsing LR"), ids):
            # We must do this on a case-by-case basis or risk 2x memory footprint.
            # This should hopefully be very fast as we already have gas_current_id in
            # the cache anyway.

            while particle_id == truncated_gas_id:
                gas_lagrangian_regions[gas_current_index] = lr
                gas_current_index += 1

                try:
                    gas_current_id = self.gas_ids[gas_current_index]

                    if self.truncate_ids is not None:
                        # We do this truncation on the fly as it is more memory efficient.
                        truncated_gas_id = gas_current_id % truncate
                    else:
                        truncated_gas_id = gas_current_id
                except IndexError as e:
                    # We must have reached the end of the current ids.
                    assert gas_current_index == len(gas_lagrangian_regions)
                    if touched_last_gas:
                        # This should _never_ happen by construction.
                        raise e
                    else:
                        touched_last_gas = True

                        # Reset values so we don't get stuck in the while loop
                        gas_current_id = -1
                        truncated_gas_id = -1
                        gas_current_index -= 1

            while particle_id == truncated_star_id:
                star_lagrangian_regions[star_current_index] = lr
                star_current_index += 1

                try:
                    star_current_id = self.star_ids[star_current_index]

                    if self.truncate_ids is not None:
                        truncated_star_id = star_current_id % truncate
                    else:
                        truncated_star_id = star_current_id
                except IndexError as e:
                    assert star_current_index == len(star_lagrangian_regions)
                    if touched_last_star:
                        raise e
                    else:
                        touched_last_star = True

                        # Reset values so we don't get stuck in the while loop
                        star_current_id = -1
                        truncated_star_id = -1
                        star_current_index -= 1

            # Black holes need to be implemented here at some point.

        if not touched_last_gas:
            print("Not parsed all gas particles. Results might be wrong.")
            print(
                "Got to index {}/{}".format(
                    gas_current_index, len(gas_lagrangian_regions)
                )
            )
        if not touched_last_star:
            print("Not parsed all star particles. Results might be wrong.")
            print(
                "Got to index {}/{}".format(
                    star_current_index, len(star_lagrangian_regions)
                )
            )

        self.gas_lagrangian_regions = gas_lagrangian_regions.astype(int)
        self.star_lagrangian_regions = star_lagrangian_regions.astype(int)

        return


class Snapshot(object):
    """
    Snapshot container for both the baryonic and DM particle data, as well as the
    halo catalogue. Pass two of these to Simulation to run the analysis.
    """

    def __init__(self, snapshot_filename, catalogue_filename=None, truncate_ids=None):
        """
        Opens the data and stuffs the information into the appropriate
        objects.

        truncate_ids should be an integer above which the ParticleIDs are all
        truncated. This is helpful as in some simulation codes (e.g. Mufasa)
        star-forming particles have their higher-up bits played with.
        """

        particle_data = h5py.File(snapshot_filename, "r")

        self.header = dict(particle_data["Header"].attrs)

        if catalogue_filename is not None:
            self.halo_catalogue = caesar.load(catalogue_filename)
        else:
            # We need halo_catalogue.halos = None and halo_catalogue.galaxies = None
            # for simplicity, and also extensibility perhaps for later.
            catalogue = namedtuple("EmptyHaloCatalogue", ["halos", "galaxies"])
            self.halo_catalogue = catalogue._make([None, None])

        self.dark_matter = DMParticles(
            particle_data["PartType1"], self.halo_catalogue.halos
        )

        try:
            self.baryonic_matter = BaryonicParticles(
                particle_data["PartType0"],
                particle_data["PartType4"],
                particle_data["PartType5"],
                self.halo_catalogue.galaxies,
                truncate_ids=truncate_ids,
            )
        except KeyError:
            self.baryonic_matter = BaryonicParticles(
                particle_data["PartType0"],
                np.array([]),
                np.array([]),
                self.halo_catalogue.galaxies,
                truncate_ids=truncate_ids,
            )

        return

    def close_file_handles(self):
        """
        Closes the file handles used for reading data so that the struct can be
        pickled.
        """

        self.particle_data.close()

        return


class Simulation(object):
    """
    Simulation container for both the z=0 and z->infty snapshots. This also
    contains the relevant routines for the data reduction.
    """

    def __init__(self, snapshot_ini: Snapshot, snapshot_end: Snapshot):
        """
        Snapshot_ini is the initial conidtions, with snapshot_end the z=0 snap.
        """

        self.snapshot_ini = snapshot_ini
        self.snapshot_end = snapshot_end

        self.identify_lagrangian_regions()

        return

    def identify_lagrangian_regions(self):
        """
        Identifies the lagrangian regions based on the given snapshots.
        
        This performs a nearest-neighbour search after building a kd-tree
        so may take some time.

        We identify the lagrangian regions by associating the z=ini gas particle
        with the GroupID of the nearest DM z=ini particle at z=end.
        """

        boxsize = self.snapshot_ini.header["BoxSize"]

        tree = KDTree(self.snapshot_ini.dark_matter.coordinates, boxsize=boxsize)

        # This returns the index of the Dark Matter particle that belongs to
        # the relevant z=0 group.
        print("Querying tree")
        _, ids = tree.query(
            self.snapshot_ini.baryonic_matter.gas_coordinates, k=1, n_jobs=-1
        )
        print("Finished querying tree")

        # This requires the data to be sorted by ParticleID.
        # First we'll assert we have the same number.
        assert len(self.snapshot_end.dark_matter.ids) == len(
            self.snapshot_ini.dark_matter.ids
        )
        assert (
            self.snapshot_end.dark_matter.ids[-1]
            == self.snapshot_ini.dark_matter.ids[-1]
        )
        self.lagrangian_regions = self.snapshot_end.dark_matter.halos[ids]

        # This could cause problems if we ever have stars in the ICs
        self.snapshot_end.baryonic_matter.parse_lagrangian_regions(
            self.snapshot_ini.baryonic_matter.gas_ids, self.lagrangian_regions
        )

        return

    def prepare_analysis_arrays(self):
        """
        Allocates the analysis arrays.
        """

        self.n_halos = self.snapshot_end.halo_catalogue.nhalos
        # Because we have -1 as the group_id, we want to have that information saved
        # in the final array element - hence we overallocate by 1.
        self.n_groups = self.n_halos + 1

        # The total mass that ends up in the z=0 halo.
        self.dark_matter_mass_in_halo = np.zeros(self.n_groups)
        self.gas_mass_in_halo = np.zeros(self.n_groups)
        self.stellar_mass_in_halo = np.zeros(self.n_groups)

        # The total mass at z=inf inside the lagrangian region of the halo
        self.dark_matter_mass_in_lagrangian = np.zeros(self.n_groups)
        self.gas_mass_in_lagrangian = np.zeros(self.n_groups)
        self.stellar_mass_in_lagrangian = np.zeros(self.n_groups)

        # The total mass at z=0 that ends up in the halo from the lagragnain region
        self.dark_matter_mass_in_halo_from_lagrangian = np.zeros(self.n_groups)
        self.gas_mass_in_halo_from_lagrangian = np.zeros(self.n_groups)
        self.stellar_mass_in_halo_from_lagrangian = np.zeros(self.n_groups)

        # The total mass at z=0 that ends up in the halo from outside the lagrangian
        # region
        self.dark_matter_mass_in_halo_from_outside_lagrangian = np.zeros(self.n_groups)
        self.gas_mass_in_halo_from_outside_lagrangian = np.zeros(self.n_groups)
        self.stellar_mass_in_halo_from_outside_lagrangian = np.zeros(self.n_groups)

        # The total mass at z=0 that ends up in the halo from other halos lagrangian
        # regions
        self.dark_matter_mass_in_halo_from_other_lagrangian = np.zeros(self.n_groups)
        self.gas_mass_in_halo_from_other_lagrangian = np.zeros(self.n_groups)
        self.stellar_mass_in_halo_from_other_lagrangian = np.zeros(self.n_groups)

        # The total mass at z=0 from the original lagrangian region that ends up
        # outside any lagrangian regions
        self.dark_matter_mass_outside_halo_from_lagrangian = np.zeros(self.n_groups)
        self.gas_mass_outside_halo_from_lagrangian = np.zeros(self.n_groups)
        self.stellar_mass_outside_halo_from_lagrangian = np.zeros(self.n_groups)

    def run_gas_analysis(self):
        """
        Runs the analysis (gas) - this is just a linear pass through all of the
        particles.
        """

        for group_id, lagrangian_region, mass in zip(
            tqdm(self.snapshot_end.baryonic_matter.gas_halos, desc="Analysing gas"),
            self.snapshot_end.baryonic_matter.gas_lagrangian_regions,
            self.snapshot_end.baryonic_matter.gas_masses,
        ):
            # First, add on the halo mass
            try:
                self.gas_mass_in_halo[group_id] += mass
            except IndexError:
                # Must be weird. Let's just forget about it.
                pass

            # Add on mass to corresponding lagrangian region
            try:
                self.gas_mass_in_lagrangian[lagrangian_region] += mass
            except IndexError:
                # Must be weird. Let's just forget about it.
                pass

            if group_id == lagrangian_region:
                # We're in the same halo as LR
                try:
                    self.gas_mass_in_halo_from_lagrangian[group_id] += mass
                except IndexError:
                    # Wow you are so uninteresting, particle.
                    pass
            elif group_id != -1:
                if lagrangian_region != -1:
                    # We've ended up in someone else's lagrangian
                    self.gas_mass_in_halo_from_other_lagrangian[group_id] += mass
                else:
                    # We must be new to the game!
                    self.gas_mass_in_halo_from_outside_lagrangian[group_id] += mass
            else:
                # We've ended up outside any lagrangian region
                if lagrangian_region != -1:
                    # We _used_ to be in a lagrangian region
                    self.gas_mass_outside_halo_from_lagrangian[
                        lagrangian_region
                    ] += mass
                else:
                    # We were never important :(
                    pass

        return

    def run_star_analysis(self):
        """
        Runs the analysis (stars) - this is just a linear pass through all of the
        particles.
        """

        for group_id, lagrangian_region, mass in zip(
            tqdm(self.snapshot_end.baryonic_matter.star_halos, desc="Analysing stars"),
            self.snapshot_end.baryonic_matter.star_lagrangian_regions,
            self.snapshot_end.baryonic_matter.star_masses,
        ):
            # First, add on the halo mass
            try:
                self.stellar_mass_in_halo[group_id] += mass
            except IndexError:
                # Must be -1
                pass

            # Add on mass to corresponding lagrangian region
            try:
                self.stellar_mass_in_lagrangian[lagrangian_region] += mass
            except IndexError:
                # Must be -1
                pass

            if group_id == lagrangian_region:
                # We're in the same halo as LR
                self.stellar_mass_in_halo_from_lagrangian[group_id] += mass
            elif group_id != -1:
                if lagrangian_region != -1:
                    # We've ended up in someone else's lagrangian
                    self.stellar_mass_in_halo_from_other_lagrangian[group_id] += mass
                else:
                    # We must be new to the game!
                    self.stellar_mass_in_halo_from_outside_lagrangian[group_id] += mass
            else:
                # We've ended up outside any lagrangian region
                if lagrangian_region != -1:
                    # We _used_ to be in a lagrangian region
                    self.stellar_mass_outside_halo_from_lagrangian[
                        lagrangian_region
                    ] += mass
                else:
                    # We were never important :(
                    pass

        return

    def run_dark_matter_analysis(self):
        """
        Runs the dark matter analysis. This is more of a check than anything -
        everything here is pretty much just a "null" test.
        """

        for group_id, lagrangian_region, mass in zip(
            tqdm(self.snapshot_end.dark_matter.halos, desc="Analysing DM"),
            self.snapshot_end.dark_matter.halos,  # By definition they are the same.
            self.snapshot_end.dark_matter.masses,
        ):
            # First, add on the halo mass
            try:
                self.dark_matter_mass_in_halo[group_id] += mass
            except IndexError:
                # Must be -1
                pass

            # Add on mass to corresponding lagrangian region
            try:
                self.dark_matter_mass_in_lagrangian[lagrangian_region] += mass
            except IndexError:
                # Must be -1
                pass

            if group_id == lagrangian_region:
                # We're in the same halo as LR
                self.dark_matter_mass_in_halo_from_lagrangian[group_id] += mass
            elif group_id != -1:
                if lagrangian_region != -1:
                    # We've ended up in someone else's lagrangian
                    self.dark_matter_mass_in_halo_from_other_lagrangian[
                        group_id
                    ] += mass
                else:
                    # We must be new to the game!
                    self.dark_matter_mass_in_halo_from_outside_lagrangian[
                        group_id
                    ] += mass
            else:
                # We've ended up outside any lagrangian region
                if lagrangian_region != -1:
                    # We _used_ to be in a lagrangian region
                    self.dark_matter_mass_outside_halo_from_lagrangian[
                        lagrangian_region
                    ] += mass
                else:
                    # We were never important :(
                    pass

        return

    def write_reduced_data(self, filename: str):
        """
        Writes out the reduced data as a table to files.

        You will be presented with three CSV files:

        + gas_{filename}
        + stellar_{filename}
        + dark_matter_{filename} (meant to be a null test)

        These can then be used for data reduction again - you can even get the
        HMF from these files.
        """

        attributes = [
            "mass_in_halo",
            "mass_in_lagrangian",
            "mass_in_halo_from_lagrangian",
            "mass_in_halo_from_outside_lagrangian",
            "mass_in_halo_from_other_lagrangian",
            "mass_outside_halo_from_lagrangian",
        ]

        header = ",".join(attributes)

        data_types = ["dark_matter", "gas", "stellar"]

        for data_type in data_types:
            if filename[-4:] == ".txt":
                this_filename = "{}_{}".format(data_type, filename)
            else:
                this_filename = "{}_{}.txt".format(data_type, filename)

            data_to_write = np.array(
                [getattr(self, "{}_{}".format(data_type, x)) for x in attributes]
            ).T

            np.savetxt(this_filename, data_to_write, header=header, delimiter=",")

        return
