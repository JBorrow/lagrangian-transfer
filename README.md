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
python3 <name of snapshot file (ic)> <name of snapshot file (z=0) <name of caesar file>
```

So for example,

```bash
python3 snap_0000.hdf5 snap_0151.hdf5 caesar_snap_0151.hdf5
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
