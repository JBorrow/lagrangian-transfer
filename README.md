Lagrangian Transfer
===================

Josh Borrow, Daniel Angles-Alcazar, Romeel Dave
-----------------------------------------------

This small library is used to calculate the transfer of mass between lagrangian
regions in cosmologcal simulations.

Requirements
------------

+ `python3` - no attempts will be made to ensure that this code works with
  older versions of `python`.
+ `caesar` - you will also need to generate the relevant halo catalogues for
  each snapshot that you would like to include in the analysis. The version of
  `caesar` that we recommend is the custom version available here:
  https://bitbucket.org/laskalam/caesar. You will need to run `2to3` on this,
  and then edit out references in the `setup.py` to the `hg` version. This is a
  bit of a pain at the moment, sorry.
+ `h5py` for reading snapshots.
+ `numpy` for numerical routines.
+ `scipy` for the KDTree routines.
+ `tqdm` (optional) for a status bar.

And that's it! We assume that your output files are GADGET-oid compatible, i.e.
that they are `HDF5` files, with collections for particles where:

+ `PartType0` are gas particles
+ `PartType1` are dark matter particles
+ `PartType4` are star particles
+ `PartType5` are black hole particles.

Of course, these are similar requirements to those for using `caesar`. Note that
we explicitly do _not_ use any `pandas` routines for memory efficiency.


Analysis Script
---------------

An analysis scripy, `analyse.py`, is provided for quick analysis of the data.
It can be invoked as follows:

```bash
python3 analyse.py <name of snapshot file (ic)> <name of snapshot file (z=0) <name of caesar file>
```

So for example,

```bash
python3 analyse.py snap_0000.hdf5 snap_0151.hdf5 caesar_snap_0151.hdf5
```

after having ran `caesar snap_0151.hdf5`. We hope to improve this interface in the
future should this project continue.


API
---

The API is quite simple, should you wish to script your analysis. A good example of
its usage is the above `analyse.py`.

#### The `Snapshot` class

The `Snapshot` class is a wrapper around both `caesar` and `h5py` that is used to
load the data. Simply pass in the filename of the snapshot and (optionally) the
halo catalogue.

```python
import lagtranscaesar as lt

snap_ini = lt.Snapshot(snapshot_filename="snap_0000.hdf5")
snap_end = lt.Snapshot(
    snapshot_filename="snap_0151.hdf5",
    catalogue_filename="caesar_snap_0151.hdf5"
)
```

#### The `Simulation` class

The `Simulation` class takes two `Snapshots` and runs the analysis. For example,
continuing on from above,

```python
simulation = lt.Simulation(
    snapshot_ini=snap_ini,
    snapshot_end=snap_end
)
```

This will _not_ automatically run the analysis. To do that, you will need to
call the following methods:

```python
# Prepare the arrays - this is an allocator function
simulation.prepare_analysis_arrays()
# Run the analysis for the _gas_ particles only
simulation.run_gas_analysis()
# Run the analysis now for the star particles as well
simulation.run_star_analysis()
# Run the analysis for the dark matter particles
simulation.run_dark_matter_analysis()
```

This will produce six arrays per particle type. They are described below - note
that they are actually called `<particle_type>_<array_name>`.

+ `mass_in_halo`: the total mass of that particle type within the given group/halo.
+ `mass_in_lagrangian`: the total mass of that particle type initially within the
  lagrangian region.
+ `mass_in_halo_from_lagrangian`: the total mass of that particle type that ends up
  in the final halo at z=0 that started in the lagrangian region.
+ `mass_in_halo_from_outside_lagrangian`: the total mass of that particle type that
  starts outside of any lagrangian region but still ends up in the halo/group.
+ `mass_in_halo_from_other_lagrangian`: the total mass of that particle type that
  starts in a different group's lagrangian region but still ends up in the given halo.
+ `mass_outside_halo_from_lagrangian`: the total mass from the initial lagrangian
  region that ends up outside _any_ lagrangian region.

To save this data to file, youc an call the `write_reduced_data` method on the
`Simulation` class. This will save three CSV files.


Reading and Writing
-------------------

So, you've ran your analysis -- and it took a little while!. You would like to store
the data somewhere a little more permanent than in your fragile memory (ECC or not).

We procide two top-level functions: `write_data_to_file`, which takes a filename
and an instance of the `Simulation` class. This gets written (using `h5py`) out to
file.

To read the data in again, simply use `read_data_from_file` and you will get a
`Simulation`-like object (these are _not_ bitwise comparable, they are actually of
different types) with the same API that you can use to access your data.

Note that these files are not portable as of the current version. This is because 
we hard-code the references to both the catalogue and particle files.


Using your own Halo Finder
--------------------------

Sometimes the Caesar definition is not everything that you want out of your halo
finder. You are free to use your own halo definitions with ltcaesar, but you will
need to tell the system about them. We provide an API to do this through 
`ltcaesar.halos.FakeCaesar` where you will need to re-create an object that _looks_
like the Caesar API. This will then get passed around and used as if it is a loaded
Caesar file.

This is quite experimental, so if you expereience any problems let us know. You will
need the following information (the `FakeHalo` object may be useful for you, but you
are welcome to use structured arrays).

+ `FakeCaesar.nhalos`, number of halos
+ `FakeCaesar.halos`, a list of all of the halos, use the FakeHalo object to give each
		of these the following property (or just use a structured numpy
		array, or whatever)

+ `FakeHalo.dmlist`, a list of the _indicies_ in the original HDF5 dataset for the
	       dark matter particles
+ `FakeHalo.ndm`, the number of dark matter particles in the above list
+ `FakeHalo.glist`, same as dmlist but for gas
+ `FakeHalo.ngas`, same as ndm but for gas
+ `FakeHalo.slist`, same as dmlist but for stars
+ `FakeHalo.nstar`, same as ndm but for stars
+ `FakeHalo.GroupID`, the group which this halo belongs to (i.e. its halo id)


Difficult Snapshots
-------------------

The majority of Gadget-oid snapshots should be read just fine with the thin layer
over `h5py` that is implemeted in `ltcaesar`. However, if you are having problems
(for example if you have multiple files dumped per snapshot), you will need to use
the `yt` wrapper that is provided. You can do this by passing a truthy `load_using_yt`
value to the `Snapshot` class. There is documentation available in the scripts
used for running the full analysis on how to use this feature also.


Troubleshooting
---------------

This analysis, whilst on the surface simple, actually requires some realtively
complex procedures to be ran in a reasonable computational time. It also has to
deal with some bugs in original simulation codes.

If your code does not correctly update the `ParticleID` of a progenitor gas
particle when it forms a star, then you will need to use the `notrunc` option
in the `analyse.py` script. This can be activated as follows:

```bash
python3 analyse.py <name of snapshot file (ic)> <name of snapshot file (z=0) <name of caesar file> notrunc
```

This will ignore all gas particles that have had their higher bits changed. 
You should then, naturally, expect the following error message from the code:

```
Not parsed all gas particles. Results might be wrong.
Got to index x/y
Not parsed all star particles. Results might be wrong.
Got to index x/y
```

What this is telling you is that the code has not been able to find matches for
the particles with the highest IDs (those that have been bit-shifted). These
particles are then not considered further in the analysis based on Lagrangian
Regions. 

You should ensure that these particles are ignored from any further analysis.
