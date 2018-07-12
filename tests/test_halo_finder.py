"""
These tests require _any_ test data.

That is, they require a caesar file (named caesar.hdf5), its corresponding
snapshot file, snapshot_end.hdf5, and a snapshot at z=inf (or as close as
you can get, called snapshot_ini.hdf5.

Alternatively, if it exists, we will try to use the output of analyse.py,
the lt_outputs.hdf5, that lives in the same directory that you are running
your tests from.

They should be placed in the same directory that you _run_ the tests from.

Tests in this file:
    + We recover the correct halo mass
    + We recover the correct mass of gas and stars in each halo
    + We recover the correct gas fraction
"""

import ltcaesar as lt
import numpy as np
import sys

def test_load_data():
    """
    Very basic test that either loads the data or makes sure that it exists,
    by creating it.
    """

    try:
        data = lt.read_data_from_file("lt_outputs.hdf5")
    except:
        sys.call([
            "python3",
            "analyse.py",
            "snapshot_ini.hdf5",
            "snapshot_end.hdf5",
            "caesar.hdf5"
        ])

        data = lt.read_data_from_file("lt_outputs.hdf5")

    return data

def test_masses(mass_factor=(1e10 / 0.68)):
    """
    Tests that we have the correct halo masses (dm, gas, *).

    Note that in this test, we assume that the simulation units are

    M_sim = 1e10 M_sun / h
    """

    data = test_load_data()

    data_from_caesar = data.snapshot_end.halo_catalogue

    # Dark Matter
    masses_from_cesar = np.sort([x.masses["dm"] for x in data_from_caesar.halos])
    # -1th element is all DM _not_ in a halo.
    masses_from_ltcaesar = np.sort(data.dark_matter_mass_in_halo[:-1])

    masses_from_ltcaesar *= mass_factor
    
    check_dm = np.isclose(
        masses_from_cesar,
        masses_from_ltcaesar,
        1e-10
    )

    assert check_dm.all()

    # Gas
    masses_from_cesar = np.sort([x.masses["gas"] for x in data_from_caesar.halos])
    # -1th element is all gas _not_ in a halo.
    masses_from_ltcaesar = np.sort(data.gas_mass_in_halo[:-1])

    masses_from_ltcaesar *= mass_factor
    
    check_gas = np.isclose(
        masses_from_cesar,
        masses_from_ltcaesar,
        1e-10
    )

    assert check_gas.all()

    # Stars
    masses_from_cesar = np.sort([x.masses["stellar"] for x in data_from_caesar.halos])
    # -1th element is all gas _not_ in a halo.
    masses_from_ltcaesar = np.sort(data.stellar_mass_in_halo[:-1])

    masses_from_ltcaesar *= mass_factor
    
    check_stars = np.isclose(
        masses_from_cesar,
        masses_from_ltcaesar,
        1e-10
    )

    assert check_stars.all()

    return
        
    

