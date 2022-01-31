# coding: utf-8
# Copyright (c) CMD Lab Development Team.
# Distributed under the terms of the GNU License.

"""
This class instantiates the default VASP input set to simulate 2D-substrate slab hetero_interfaces.
"""
from __future__ import division, unicode_literals, print_function

import os, warnings, six, numpy as np
from math import ceil, floor

from monty.json import MSONable
from monty.serialization import loadfn

from pymatgen import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.vasp.sets import DictSet, get_vasprun_outcar, get_structure_from_prev_run


__author__ = "Tara M. Boland"
__copyright__ = "Copyright 2020, CMD Lab"
__maintainer__ = "Tara M. Boland"
__email__ = "tboland1@asu.edu"
__date__ = "Jan 06, 2020"

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

warnings.filterwarnings("ignore")

def _load_yaml_config(fname):
    config = loadfn(os.path.join(MODULE_DIR, "%s.yaml" % fname))
    config["INCAR"].update(loadfn(os.path.join(MODULE_DIR, "VASPIncarBase.yaml")))
    return config

class CMDLRelaxSet(DictSet):
    """  
    Implementation of VaspInputSet utilizing parameters in the public 2D Materials Synthesis Database.
    """
    CONFIG = _load_yaml_config("CMDLRelaxSet")

    def __init__(self, structure, **kwargs):
        super(CMDLRelaxSet, self).__init__(structure, CMDLRelaxSet.CONFIG, **kwargs)
        self.kwargs = kwargs

class CMDLInterfaceSet(CMDLRelaxSet):
    """
    Class for writing a set of interface vasp runs, including the 2D, substrate 
    slabs, and hetero_interface (oriented along the c direction) as well as the 
    bulk (3D) phase of the 2D material and the un-oriented unit cells, to ensure 
    the same K_POINTS, POTCAR and INCAR criterion.

    Args:
        structure (Structure): The structure for the calculation.

    Other Parameters:
        k_product (int): Default to 20, k_point number x length for a & b directions,
            also for c direction in bulk calculations.
        iface (bool): Set to False for initial structure calculations to fully 
            relax the structures to calculate the adsorption energy of a 2D 
            material on the substrate slab. Defaults to True.
        vdw (str): The string representation for the vdW correction functional
            that will be used for the simulations. Defaults to optB88.
        auto_dipole (bool): Whether to set dipole corrections. Defaults to
            False.
        set_mix (bool): Whether to set the mixing parameters. Defaults to
            False.
        sort_structure (bool): Whether to sort the structure. Defaults to
            False.
        user_incar_settings (dict): A way to override the default settings
            for the incar.
        kwargs: Other kwargs supported by :class:`DictSet`.

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
        self.set_mix = set_mix
        self.user_incar_settings = user_incar_settings
        self.kwargs = kwargs

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
         
        # modify configuration dictionary
        self._config_dict["INCAR"].update(iface_incar)
        if self._config_dict["INCAR"]['ISPIN'] == 1:
            self._config_dict["INCAR"].pop("MAGMOM", '')
    
    @property
    def kpoints(self):
        """
        k_product (default to 20) is the number of k-points * length for a & b
        directions, also for c direction in bulk calculations. Results in 
        k_product k-points/Angstrom. Defaults to automatic mesh & Gamma.
        """
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

class CMDLElectronicSet(CMDLRelaxSet):
    """
    Class for writing vasp inputs for DOS, bader, and charge density difference
    runs from previous VASP jobs. Typically, you would use the classmethod
    from_prev_calc to initialize from a previous SCF run.

    Args:
        structure (Structure): The structure for the calculation.

    Other Parameters:
        reciprocal_density (int): For static calculations, we usually set the
            reciprocal density by volume. This is a convenience arg to change
            that, rather than using user_kpoints_settings. Defaults to 100,
            which is ~50% more than that of standard relaxation calculations.
        force_gamma (bool): Force gamma k-point mesh.
        prev_incar (Incar/string): Incar file from previous run.
        **kwargs: kwargs supported by CMDLRelaxSet.

    Returns:
        Vasp input set object
    """
    def __init__(self, structure, reciprocal_density=200, force_gamma=False,
                 prev_incar=None, **kwargs):
        super(CMDLElectronicSet, self).__init__(structure, **kwargs)
        if isinstance(prev_incar, six.string_types):
            prev_incar = Incar.from_file(prev_incar)

        self.reciprocal_density = reciprocal_density
        self.force_gamma = force_gamma
        self.prev_incar = prev_incar
        self.kwargs = kwargs

    @property
    def incar(self):
        """
        :return: Incar
        """
        parent_incar = super(CMDLElectronicSet, self).incar # get inputset parent incar
        # if no prev use parent
        incar = Incar(self.prev_incar) if self.prev_incar is not None else Incar(parent_incar)
        
        # remove incar tags that are not needed for NonSCF
        remove = ["EDIFFG", "ISIF", "LREAL", "POTIM", "KPOINT_BSE", "NELMDL", "MAGMOM", 
                  "AMIX_MAG", "BMIX_MAG", "AMIX", "BMIX", "IMIX", "AMIN", "NELMIN"]
        [ incar.pop(tag) for tag in remove if tag in incar.keys() ]

        # enforce tags for NonSCF calcs: ICHARG should be 11
        incar.update({"IBRION": -1, "ISTART": 1, "LWAVE": False, "NSW": 0, "ISYM": 0,
                      "ICHARG": 11})
        incar.update(self.kwargs.get("user_incar_settings", {}))
        return incar


    @property
    def kpoints(self):
        """
        Generate a dense k-point grid for dos/bader analysis.
        """
        return Kpoints.automatic_density_by_vol(self.structure, self.reciprocal_density, 
            force_gamma=self.force_gamma)

    @classmethod
    def from_prev_calc(cls, prev_calc_dir, dedos=0.05, grid_density=0.03,
                       dos=True, bader=True, cdd=False, small_gap_multiply=None, 
                       nbands_factor=1, dos_around_fermi=[4,6], **kwargs):
        """
        Generate a set of Vasp input files for ElectronicFW calculations from a
        directory of previous directory.

        Args:
            prev_calc_dir (str): The directory containing the outputs(
                vasprun.xml and OUTCAR) from the previous vasp run.

        Other Parameters:
            dedos (float): Automatically set nedos using the total energy range
                which will be divided by the energy step dedos. Default 0.05 eV.                
            grid_density (float): Distance between grid points for the NGXF,Y,Z grids.
                Defaults to 0.03 Angs; NGXF,Y,Z are ~2x > default. For charge 
                density difference calculations the parent grid density is used for all
                children fireworks.
            dos (bool): If True, sets INCAR tags for high quality site-orbital projected 
                density of states. Defaults to True.
            bader (bool): If True, sets INCAR tags to generate bader analysis files. 
            cdd (bool): If True, ensures the grid density matches between the parent 
                and child Fireworks. Default set to False.
            small_gap_multiply ([float, float]): If the gap is less than
               1st index, multiply the default reciprocal_density by the 2nd
               index.
            nbands_factor (float): Multiplicative factor for NBANDS. 
            dos_around_fermi (bool/list): The element projected density of states is
                calculated around the fermi level. Default range is [efermi-4, efermi+6].
                If you want a different range supply a list i.e. [4,6] or False to compute
                the entire dos range.
            **kwargs: All kwargs supported by CMDLRelaxSet, other than prev_incar
                and structure which are determined from the prev_calc_dir.
        """
        vasprun, outcar = get_vasprun_outcar(prev_calc_dir)

        incar = vasprun.incar
        structure = get_structure_from_prev_run(vasprun, outcar) # Magmom-decorated struct

        # Turn off spin when magmom for every site is smaller than 0.02.
        if outcar and outcar.magnetization:
            site_magmom = np.array([i["tot"] for i in outcar.magnetization])
            ispin = 2 if np.any(site_magmom[np.abs(site_magmom) > 0.02]) else 1
        elif vasprun.is_spin:
            ispin = 2
        else:
            ispin = 1
        
        # set nbands factor
        nbands = int(np.ceil(vasprun.parameters["NBANDS"] * nbands_factor))
        incar.update({"ISPIN": ispin, "NBANDS": nbands})

        # multiply the reciprocal density if needed:
        if small_gap_multiply:
            gap = vasprun.eigenvalue_band_properties[0]
            if gap <= small_gap_multiply[0]:
                kwargs["reciprocal_density"] = kwargs.get("reciprocal_density", 200) * small_gap_multiply[1]

        # check if the previous calc was metallic or insulating
        gap = vasprun.complete_dos.get_gap()
        if gap > 0.0: # if insulator/semi -5
            incar.update({"ISMEAR":-5})
            incar.pop("SIGMA")
        else: # ismear 0; 0.05 smear
            incar.update({"ISMEAR": 0, "SIGMA": 0.05})

        # dos settings
        if dos:
            # automatic setting of nedos using the energy range and the energy step dedos
            if dos_around_fermi:
                fermi = vasprun.efermi
                emin, emax = floor(fermi - dos_around_fermi[0]), ceil(fermi + dos_around_fermi[1]) 
                nedos = ceil(abs(emin - emax) / dedos) # compute dos spacing
                incar.update({"LORBIT": 11, "NEDOS": nedos + 1 if nedos % 2 == 0 else nedos,
                              "EMIN": emin, "EMAX": emax})
            else:
                emax, emin = max(vasprun.complete_dos.energies), min(vasprun.complete_dos.energies)
                nedos = ceil(abs(emin - emax) / dedos) # compute dos spacing
                incar.update({"LORBIT": 11, "NEDOS": nedos + 1 if nedos % 2 == 0 else nedos})
        # bader analysis settings
        if bader:
            # get new NGiF grid spacing
            a, b, c = structure.lattice.abc
            ngxf, ngyf, ngzf = [int(ceil(i/grid_density/10) * 10) for i in [a,b,c]]

            # modification to the incar
            bader_settings = {"LCHARG": True, "LAECHG": True, "NGXF": ngxf, "NGYF": ngyf,
                              "NGZF": ngzf}
            incar.update(bader_settings)

        # charge density difference settings
        if cdd:
            cdd_settings = {}

        return cls(structure=structure, prev_incar=incar, **kwargs)


