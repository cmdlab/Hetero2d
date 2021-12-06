# coding: utf-8
# Copyright (c) CMD Lab Development Team.
# Distributed under the terms of the GNU License.

"""
These classes write the vasp input sets used to control VASP tags.
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import glob, shutil, os, numpy as np

from pymatgen import Lattice, Structure, Specie
from pymatgen.core.surface import Slab, SlabGenerator
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.alchemy.transmuters import StandardTransmuter
from pymatgen.io.vasp import Incar, Poscar, Potcar, PotcarSingle
from pymatgen.transformations.site_transformations import AddSitePropertyTransformation

from importlib import import_module
from monty.serialization import dumpfn, loadfn

from atomate.utils.utils import get_logger

from fireworks import Firework
from fireworks.core.firework import FiretaskBase
from fireworks.utilities.fw_utilities import explicit_serialize

# mp_interfaces
from hetero2d.manipulate.utils import set_sd_flags
from hetero2d.manipulate.heterotransmuter import hetero_interfaces

__author__ = 'Tara M. Boland'
__email__ = 'tboland1@asu.edu'
__copyright__ = "Copyright 2020, CMD Lab"
__maintainer__ = "Tara M. Boland"

logger = get_logger(__name__)


@explicit_serialize
class WriteSlabStructureIOSet(FiretaskBase):
    """
    Apply the provided transformations to the input structure and write the
    input set for that structure. Reads structure from POSCAR if no structure 
    provided. Note that if a transformation yields many structures from one, 
    only the last structure in the list is used.

    Args:
        structure (Structure): input structure.
        transformations (list): list of names of transformation classes as defined 
            in the modules in pymatgen.transformations
        vasp_input_set (VaspInputSet): VASP input set.

    Other Parameters:
        transformation_params (list): list of dicts where each dict specifies the 
            input parameters to instantiate the transformation class in the 
            transformations list.
        override_default_vasp_params (dict): additional user input settings.
        prev_calc_dir (str): path to previous calculation if using structure 
            from another calculation.

    Returns:
       none 
    """

    required_params = ["structure", "transformations", "vasp_input_set"]
    optional_params = ["prev_calc_dir", "transformation_params", "override_default_vasp_params"]

    def run_task(self, fw_spec):
        """
        Execute the transformations and write the input files for the calculation
        """
        transformations = []
        transformation_params = self.get("transformation_params",
                                         [{} for i in range(len(self["transformations"]))])
        for t in self["transformations"]:
            found = False
            for m in ["advanced_transformations", "defect_transformations",
                      "site_transformations", "standard_transformations"]:
                mod = import_module("pymatgen.transformations.{}".format(m))
                try:
                    t_cls = getattr(mod, t)
                except AttributeError:
                    continue
                t_obj = t_cls(**transformation_params.pop(0))
                transformations.append(t_obj)
                found = True
            if not found:
                raise ValueError("Could not find transformation: {}".format(t))

        structure = self['structure'] if not self.get('prev_calc_dir', None) else \
            Poscar.from_file(os.path.join(self['prev_calc_dir'], 'POSCAR')).structure
        ts = TransformedStructure(structure)
        transmuter = StandardTransmuter([ts], transformations)
        trans_structure = transmuter.transformed_structures[-1].final_structure.copy()

        # add selective dynamics tags
        aspt = AddSitePropertyTransformation({'selective_dynamics': set_sd_flags(
            interface=trans_structure, n_layers=fw_spec.get('nlayers_sub'),
            top=True, bottom=False)})
        final_structure = aspt.apply_transformation(trans_structure)

        # update vasp input with user overrides
        vis_orig = self["vasp_input_set"]
        vis_dict = vis_orig.as_dict()
        vis_dict["structure"] = final_structure.as_dict()
        vis_dict.update(self.get("override_default_vasp_params", {}) or {})
        vis = vis_orig.__class__.from_dict(vis_dict)
        vis.write_input(".")

        # uncomment if you need to debug
        # dumpfn(transmuter.transformed_structures[-1], "transformations.json")


@explicit_serialize
class WriteHeteroStructureIOSet(FiretaskBase):
    """
    Apply the provided transformations to the input 2D structure and 
    substrate slab then write the input set for that simulation. Reads 
    structure from POSCAR if no structure provided. 

    Args:
        structure (Slab): 2D input structure to place on substrate. Valid 
            parameters include all pymatgen structure objects.

    Other Parameters:
        vasp_input_set (VaspInputSet): VASP input set for the transformed 2d on sub-slab
        override_default_vasp_params (dict): additional user input settings.
        user_incar_settings (dict): additional incar settings to add to the calculation.

    Returns:
        None
    """
    required_params = ["structure", "nlayers_sub", "nlayers_2d"]
    optional_params = ["user_incar_settings", "override_default_vasp_params",
                       "vasp_input_set"]

    def run_task(self, fw_spec):
        '''
        Write the selective dynamics tags for the 2d-substrate heterostructure
        configuration and all calculation input file to the specified directory.
        '''
        structure = self['structure']

        # set selective dynamics for the 2D sub slab iface 
        sd_flags = set_sd_flags(
            interface=structure,
            n_layers=self['nlayers_sub'] + self['nlayers_2d'],
            top=True, bottom=False)
        for idx, sd in enumerate(sd_flags):
            sd_flags[idx] = [True if j == 1.0 else False for j in sd]
        structure.add_site_property('selective_dynamics', sd_flags)

        # write vasp input
        vis_orig = self["vasp_input_set"]  # get vis passed to FW
        vis_dict = vis_orig.as_dict()  # editable
        vis_dict["structure"] = structure.as_dict()  # update structure
        vis_dict.update(self.get("override_default_vasp_params",
                                 {}) or {})  # override defaults
        vis_dict['user_incar_settings'].update(self.get("user_incar_settings", {}))  # uis
        vis = vis_orig.__class__.from_dict(vis_dict)  # update the vis
        vis.write_input(".")
