"""
This file contains code that is used to enable ltcaesar to work with a generic
halo finder. All that is requried is to populate a "fake" caesar object with the
same information that is used in the reduction analysis in ltcaesar.

Namely, you will need to populate the FakeCaesar object with the following information:

    FakeCaesar.nhalos == number of halos
    FakeCaesar.halos == a list of all of the halos, use the FakeHalo object to give each
                        of these the following property (or just use a structured numpy
                        array, or whatever)

    FakeHalo.dmlist == a list of the _indicies_ in the original HDF5 dataset for the
                       dark matter particles
    FakeHalo.ndm == the number of dark matter particles in the above list
    FakeHalo.glist == same as dmlist but for gas
    FakeHalo.ngas == same as ndm but for gas
    FakeHalo.slist == same as dmlist but for stars
    FakeHalo.nstar == same as ndm but for stars
    FakeHalo.GroupID == the group which this halo belongs to (i.e. its halo id)

You can then pass this to a Snapshot object and it will automatically use this instead 
of attempting to load the caesar snapshot.

Note: this will probably not work with the current i/o functions.
"""

# We need this only for typing information.
import numpy as np
from typing import Union, List


class FakeCaesarError(Exception):
    """
    Raises an error when things go wrong with FakeCaesar, such as when check_valid
    fails.
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class FakeHalo(object):
    """
    Fake halo object; this has the same API as Caesar.

    You will need to pass the following to the FakeHalo object:

    + dmlist
    + ndm
    + glist
    + ngas
    + slist
    + nstar
    + GroupID

    These are all fairly straightforward, and are defined in the header of halos.py
    if you are concerned about them.
    """

    def __init__(
        self,
        dmlist: np.ndarray,
        ndm: int,
        glist: np.ndarray,
        ngas: int,
        slist: np.ndarray,
        nstar: int,
        GroupID,
    ):
        self.dmlist = dmlist
        self.ndm = ndm
        self.glist = glist
        self.ngas = ngas
        self.slist = slist
        self.nstar = nstar
        self.GroupID = GroupID

        return

    def check_valid():
        """
        Checks if we are a valid halo!
        """

        # Note that len() is O(1) on numpy arrays and lists.
        real_ndm = len(self.dmlist)
        real_ngas = len(self.glist)
        real_nstar = len(self.slist)

        if real_ndm != self.ndm:
            raise FakeCaesarError(
                (
                    "Ndm does not match the number of DM particles "
                    "actually supplied to FakeHalo. Actually "
                    "found {} particles, when we expected {}. "
                    "GroupID = {}."
                ).format(real_ndm, self.ndm, self.GroupID)
            )
        else:
            pass

        if real_ngas != self.ngas:
            raise FakeCaesarError(
                (
                    "Ngas does not match the number of gas particles "
                    "actually supplied to FakeHalo. Actually "
                    "found {} particles, when we expected {}. "
                    "GroupID = {}."
                ).format(real_ngas, self.ngas, self.GroupID)
            )
        else:
            pass

        if real_nstar != self.nstar:
            raise FakeCaesarError(
                (
                    "Nstar does not match the number of star particles "
                    "actually supplied to FakeHalo. Actually "
                    "found {} particles, when we expected {}. "
                    "GroupID = {}."
                ).format(real_nstar, self.nstar, self.GroupID)
            )
        else:
            pass

        return  # no news is good news!


class FakeCaesar(object):
    """
    Fake Caesar object; this looks like a halo finder (i.e. it has the same API).

    Pass a list of halos (.halos), and the number of halos (.nhalos). Then call
    .check_valid(); this will also attempt to check each halo if it is valid using
    the same funciton iff it exists.
    """

    def __init__(self, halos: Union[np.ndarray, List[FakeHalo]], nhalos: int):
        """
        Pass me halos, a list of all halos, and nhalos, the total number of halos.

        You may then want to call check_valid on the resulting FakeCaesar object; this
        will ensure that things are all good and proper.
        """

        self.halos = halos
        self.nhalos = nhalos

    def check_valid(self):
        """
        Checks if the supplied data is valid. Also calls check_valid on all halos in
        the halos list, if available (catches AttributeError).
        """

        # Note that len() is O(1) on numpy arrays and lists.
        real_number_of_halos = len(self.halos)

        if real_number_of_halos != self.nhalos:
            raise FakeCaesarError(
                (
                    "Nhalos does not match the number of halos "
                    "actually supplied to FakeCaesar. Actually "
                    "found {} halos, when we expected {}."
                ).format(real_number_of_halos, self.nhalos)
            )
        else:
            pass

        # Now attempt to call check_valid on all halos.

        for halo in self.halos:
            try:
                halo.check_valid()
            except AttributeError:
                pass

        return  # no news is good news.
