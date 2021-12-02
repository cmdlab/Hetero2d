# coding: utf-8
# Copyright (c) CMD Lab Development Team.
# Distributed under the terms of the GNU License.

"""
This class instantiates the default VASP input set to simulate 2D-substrate slab hetero_interfaces.
"""

from __future__ import division, unicode_literals, print_function

import numpy as np
import os
import warnings

from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen import Structure
from pymatgen.io.vasp.sets import DictSet

__author__ = "Tara M. Boland"
__copyright__ = "Copyright 2020, CMD Lab"
__maintainer__ = "Tara M. Boland"
__email__ = "tboland1@asu.edu"
__date__ = "Jan 06, 2020"

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")


def _load_yaml_config(fname):
    config = loadfn(os.path.join(MODULE_DIR, "%s.yaml" % fname))
    config["INCAR"].update(loadfn(os.path.join(MODULE_DIR,
                                               "VASPIncarBase.yaml")))
    return config


class CMDLRelaxSet(DictSet):
    """  
    Implementation of VaspInputSet utilizing parameters in the public 2D Materials Synthesis Database.
    """
    CONFIG = _load_yaml_config("CMDLRelaxSet")

    def __init__(self, structure, **kwargs):
        super(CMDLRelaxSet, self).__init__(
            structure, CMDLRelaxSet.CONFIG, **kwargs)
        self.kwargs = kwargs


class CMDLInterfaceSet(CMDLRelaxSet):
    """
    Class for writing a set of interface vasp runs, including the 2D,
    substrate slabs, and hetero_interface (oriented along the
    c direction) as well as the bulk (3D) phase of the 2D material and the 
    un-oriented unit cells, to ensure the same K_POINTS, POTCAR and INCAR criterion.

    Args:
        structure (Structure): The structure for the calculation

    Other Parameters:
        k_product (int): default to 20, k_point number x length for a & b directions,
            also for c direction in bulk calculations
        iface (bool): Set to False for initial structure calculations to fully 
            relax the structures to calculate the adsorption energy of a 2D 
            material on the substrate slab. Defaults to True.
        vdw (str): The string representation for the vdW correction functional
            that will be used for the simulations. Defaults to optB88.
        auto_dipole (bool): Whether to set dipole corrections. Defaults to
            False.
        set_mix (bool): Whether to set the mixing parameters. Defaults to
            True.
        sort_structure (bool): Whether to sort the structure. Defaults to
            False.
        user_incar_settings (dict): A way to override the default settings
            for the incar.
        kwargs:
            Other kwargs supported by :class:`DictSet`.

    Returns:
        Vasp input set object
    """

    def __init__(self, structure, k_product=20, iface=True, vdw='optB88',
                 auto_dipole=False, set_mix=False, sort_structure=False,
                 user_incar_settings=None, **kwargs):
        super(CMDLInterfaceSet, self).__init__(structure, **kwargs)

        if sort_structure:
            structure = structure.get_sorted_structure()

        self.structure = structure
        self.k_product = k_product
        self.iface = iface
        self.vdw = vdw.lower() if vdw is not None else None
        self.auto_dipole = auto_dipole
        self.kwargs = kwargs
        self.set_mix = set_mix
        self.user_incar_settings = user_incar_settings

        iface_incar = {}
        if self.iface:
            iface_incar["ISIF"] = 2
            if self.set_mix:
                iface_incar["AMIN"] = 0.01
                iface_incar["AMIX"] = 0.2
                iface_incar["BMIX"] = 0.001
            iface_incar["NELMIN"] = 8

        if self.auto_dipole:
            weights = [s.species_and_occu.weight for s in structure]
            center_of_mass = np.average(structure.frac_coords,
                                        weights=weights, axis=0)
            iface_incar["IDIPOL"] = 3
            iface_incar["LDIPOL"] = True
            iface_incar["DIPOL"] = ' '.join(str(i) for i in center_of_mass)
        if self.vdw:
            vdw_par = loadfn(os.path.join(MODULE_DIR, "vdW_parameters.yaml"))
            try:
                self._config_dict["INCAR"].update(vdw_par[self.vdw])
            except KeyError:
                raise KeyError("Invalid or unsupported van-der-Waals "
                               "functional. Supported functionals are "
                               "%s." % vdw_par.keys())

        if user_incar_settings:
            iface_incar.update(user_incar_settings)
        self._config_dict["INCAR"].update(iface_incar)

        if self._config_dict["INCAR"]['ISPIN'] == 1:
            self._config_dict["INCAR"].pop("MAGMOM", '')

    @property
    def kpoints(self):
        """
        k_product (default to 20) is k_point number * length for a & b
            directions, also for c direction in bulk calculations. Automatic 
            mesh & Gamma is the default setting.
        """

        # To get input sets, the input structure has to has the same number
        # of required parameters as a Structure object (ie. 4). Slab
        # attributes aren't going to affect the VASP inputs anyways so
        # converting the slab into a structure should not matter

        kpt = super(CMDLInterfaceSet, self).kpoints
        kpt.comment = "Automatic mesh"
        kpt.style = 'Gamma'

        # use k_product to calculate k_points, k_product = kpts[0][0] * a
        abc = self.structure.lattice.abc
        kpt_calc = [int(self.k_product / abc[0] + 0.5),
                    int(self.k_product / abc[1] + 0.5), 1]
        self.kpt_calc = kpt_calc
        # calculate kpts (c direction) for bulk. (for slab, set to 1)
        if not self.iface:
            kpt_calc[2] = int(self.k_product / abc[2] + 0.5)

        kpt.kpts[0] = kpt_calc

        return kpt

    def as_dict(self, verbosity=2):
        """
        A JSON serializable dict representation of an object.
        """

        d = MSONable.as_dict(self)
        if verbosity == 1:
            d.pop("structure", None)
        return d
