"""
Replacement for the LagTrans library with Caesar.
"""

import caesar
import h5py
import numba


class DMParticles(object):
    """
    Reads the DM particles.
    """

    def __init__(self, particles, halo_catalogue=None):
        """
        Takes the particles in PartType1 (the reference to it from h5py) and the
        halos catalogue from caesar.
        """

        self.particles = particles
        self.halo_catalogue = halo_catalogue

        self.n_parts = len(particles["ParticleIDs"])

        if halo_catalogue is not None:
            self.haloes = self.get_all_particle_references()

        # We may in the future need to ensure only the particles in the haloes
        # have their data read.
        self.ids = self.read_array("ParticleIDs")
        self.masses = self.read_array("Masses")
        self.coordinates = self.read_array("Coordinates")

        return

    @numba.jit
    def get_all_particle_references(self):
        """
        Gets an nparticle long array that denotes which halo all of the dark matter
        particles belongs to. Particles that do not reside in haloes have their
        value set to -1.
        """

        # Allocate our final output array
        haloes = np.empty(self.n_parts)
        haloes[...] = -1  # Default value for all particles _not_ in halos

        # Grab all references
        particles_in_halos = [halo.dmlist for halo in self.halo_catalogue]
        copied_haloes = [
            np.repeat(halo.GroupID, halo.ndm) for halo in self.halo_catalogue
        ]

        flattened_particles = numpy.concatenate(particles_in_halos)
        del particles_in_halos
        flattened_halos = numpy.concatenate(copied_haloes)
        del copied_haloes

        haloes.put(flattened_particles, flattened_halos)

        return haloes

    def read_array(self, name: str):
        """
        Read an array from the particle file.
        """

        return self.particles[name][...]


class BaryonicParticles(object):
    """
    Holder class for the baryonic particles.
    """

    def __init__(
        self, gas_particles, star_particles, bh_particles, halo_catalogue=None
    ):
        """
        Takes the particles in PatyType[x] (the reference to it from h5py) and
        the halos catalogue from caeasar.
        """

        self.gas_particles = gas_particles
        self.star_particles = star_particles
        self.bh_particles = bh_particles
        self.halo_catalogue = halo_catalogue

        self.n_gas_parts = len(gas_particles["ParticleIDs"])
        self.n_star_parts = len(gas_particles["ParticleIDs"])
        self.n_bh_parts = len(gas_particles["ParticleIDs"])

        if halo_catalogue is not None:
            self.gas_haloes, self.star_haloes, self.bh_haloes = (
                get_all_particle_references()
            )

        # Now we read a bunch of particle properties.

        self.gas_ids = self.gas_particles["ParticleIDs"][...]
        self.star_ids = self.star_particles["ParticleIDs"][...]
        self.bh_ids = self.bh_particles["ParticleIDs"][...]

        self.gas_masses = self.gas_particles["Masses"][...]
        self.star_masses = self.star_particles["Masses"][...]
        self.bh_masses = self.bh_particles["BH_Mass"][...]

        self.gas_coordinates = self.gas_particles["Coordinates"][...]
        self.star_coordinates = self.star_particles["Coordinates"][...]
        self.bh_coordinates = self.bh_particles["Coordinates"][...]

        return

    @numba.jit
    def get_all_particle_references(self):
        """
        Gets three nparticle long array that denotes which halo all of the
        particles belongs to. Particles that do not reside in haloes have their
        value set to -1.

        Yes, this function breaks DRY, but that's for memory efficiency.
        """

        # Allocate our final output array
        gas_haloes = np.empty(self.n_gas_parts)
        gas_haloes[...] = -1  # Default value for all particles _not_ in halos

        # Grab all references
        particles_in_halos = [galaxy.glist for galaxy in self.halo_catalogue]
        copied_haloes = [
            np.repeat(galaxy.GroupID, galaxy.ngas) for galaxy in self.halo_catalogue
        ]

        flattened_particles = numpy.concatenate(particles_in_halos)
        del particles_in_halos
        flattened_halos = numpy.concatenate(copied_haloes)
        del copied_haloes

        gas_haloes.put(flattened_particles, flattened_halos)

        del flattened_particles
        del flattened_haloes

        # Now for stars
        star_haloes = np.empty(self.n_star_parts)
        star_haloes[...] = -1  # Default value for all particles _not_ in halos

        # Grab all references
        particles_in_halos = [galaxy.slist for galaxy in self.halo_catalogue]
        copied_haloes = [
            np.repeat(galaxy.GroupID, galaxy.nstar) for galaxy in self.halo_catalogue
        ]

        flattened_particles = numpy.concatenate(particles_in_halos)
        del particles_in_halos
        flattened_halos = numpy.concatenate(copied_haloes)
        del copied_haloes

        star_haloes.put(flattened_particles, flattened_halos)

        del flattened_particles
        del flattened_haloes

        # Now for BHs
        bh_haloes = np.empty(self.n_bh_parts)
        bh_haloes[...] = -1  # Default value for all particles _not_ in halos

        # Grab all references
        particles_in_halos = [galaxy.bhlist for galaxy in self.halo_catalogue]
        copied_haloes = [
            np.repeat(galaxy.GroupID, galaxy.nbh) for galaxy in self.halo_catalogue
        ]

        flattened_particles = numpy.concatenate(particles_in_halos)
        del particles_in_halos
        flattened_halos = numpy.concatenate(copied_haloes)
        del copied_haloes

        bh_haloes.put(flattened_particles, flattened_halos)

        del flattened_particles
        del flattened_haloes

        return gas_haloes, star_haloes, bh_haloes


class Simulation(object):
    """
    Simulation container for both the baryonic and DM particle data.
    """

    def __init__(self, snapshot_filename, catalogue_filename=None):
        """
        Opens the data and stuffs the information into the appropriate
        objects.
        """

        particle_data = h5py.File(snapshot_filename, "r")

        if catalogue_filename is not None:
            halo_catalogue = caesar.load(catalogue_filename)
        else:
            halo_catalogue = None

        self.dark_matter = DMParticles(particle_data["PartType1"], halo_catalogue)
        self.baryonic_matter = BaryonicParticles(
            particle_data["PartType0"],
            particle_data["PartType3"],
            particle_data["PartType5"],
            halo_catalogue,
        )

        return
