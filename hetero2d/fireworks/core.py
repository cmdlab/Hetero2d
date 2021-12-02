# coding: utf-8
# Copyright (c) CMD Lab Development Team.
# Distributed under the terms of the GNU License.

"""
These classes implements various fireworks needed to compute the thermodynamic
stability of 2D thin films adsorbed on a substrate surface.
"""

from __future__ import division, print_function, unicode_literals, absolute_import

from atomate.utils.utils import get_logger
from atomate.vasp.config import HALF_KPOINTS_FIRST_RELAX, RELAX_MAX_FORCE, VASP_CMD, DB_FILE 
from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.firetasks.run_calc import RunVaspCustodian
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet

from fireworks import Firework, FileWriteTask
from fireworks.utilities.fw_utilities import get_slug

from pymatgen import Structure
from pymatgen.core.surface import Slab

from hetero2d.firetasks.heteroiface_tasks import CreateHeterostructureTask
from hetero2d.firetasks.parse_outputs import HeteroAnalysisToDb
from hetero2d.firetasks.write_inputs import WriteHeteroStructureIOSet, WriteSlabStructureIOSet

# hetero_interfaces
from hetero2d.io.VaspInterfaceSet import CMDLInterfaceSet

__author__ = 'Tara M. Boland'
__copyright__ = "Copyright 2020, CMD Lab"
__maintainer__ = "Tara M. Boland"
__email__ = 'tboland1@asu.edu'

logger = get_logger(__name__)

# @tboland1 - update custodian handlers to better handle vdW errors and 
# implement these changes into all vasp run's. Also create a user flag handler 
# to remove exception cases. -tboland1

from custodian.vasp.handlers import VaspErrorHandler, MeshSymmetryErrorHandler, UnconvergedErrorHandler, \
    NonConvergingErrorHandler, PositiveEnergyErrorHandler, FrozenJobErrorHandler, StdErrHandler

# Set up Custodian Error Handlers
subset = list(VaspErrorHandler.error_msgs.keys())
subset.remove('brions')

handler = [VaspErrorHandler(errors_subset_to_catch=subset), MeshSymmetryErrorHandler(),
           UnconvergedErrorHandler(), NonConvergingErrorHandler(),
           PositiveEnergyErrorHandler(), FrozenJobErrorHandler(), StdErrHandler()]


class HeteroOptimizeFW(Firework):
    def __init__(self, spec, structure, name="Structure Optimization",
                 vasp_input_set=None, user_incar_settings=None, vasp_cmd=VASP_CMD,
                 ediffg=None, db_file=DB_FILE, force_gamma=True, parents=None,
                 job_type="double_relaxation_run", max_force_threshold=RELAX_MAX_FORCE,
                 half_kpts_first_relax=HALF_KPOINTS_FIRST_RELAX, auto_npar=">>auto_npar<<",
                 **kwargs):
        """
        Optimize the given structure with additional tags for the
        heterostructure workflow.

        Args:
            structure (Structure): Input structure.
            spec (dict): The specification parameters used to control the 
                workflow and pass variables.

        Other Parameters:
            name (str): Name for the Firework.
            user_incar_settings (dict): Input settings to update the settings
                for the vasp input set.
            vasp_input_set (VaspInputSet): input set to use. Defaults to 
                CMDLInterfaceSet() if None.
            vasp_cmd (str): Command to run vasp.
            ediffg (float): Shortcut to set ediffg in certain jobs.
            db_file (str): Path to file specifying db credentials to place output 
                parsing.
            force_gamma (bool): Force gamma centered kpoint generation.
            job_type (str): custodian job type (default "double_relaxation_run").
            max_force_threshold (float): max force on a site allowed at end; otherwise, 
                reject job.
            auto_npar (bool or str): whether to set auto_npar. defaults to env_chk: 
                ">>auto_npar<<".
            half_kpts_first_relax (bool): whether to use half the kpoints for the first 
                relaxation.
            parents ([Firework]): Parents of this particular Firework.
            kwargs: Other kwargs that are passed to Firework.__init__.
        """
        name = "{}: {}".format(structure.composition.reduced_formula, name)
        user_incar_settings = user_incar_settings or None
        vasp_input_set = vasp_input_set or CMDLInterfaceSet(structure,
                                                            auto_dipole=True, user_incar_settings=user_incar_settings,
                                                            vdw='optB88', iface=True)
        
        if vasp_input_set.incar["ISIF"] in (0, 1, 2, 7) and job_type == "double_relaxation":
            warnings.warn(
                "A double relaxation run might not be appropriate with ISIF {}".format(
                    vasp_input_set.incar["ISIF"]))
        t = [WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set),
             RunVaspCustodian(handler_group=handler, vasp_cmd=vasp_cmd, 
                              job_type=job_type, ediffg=ediffg, max_force_threshold=max_force_threshold,
                              auto_npar=auto_npar, wall_time=spec.get('wall_time', None),
                              half_kpts_first_relax=half_kpts_first_relax), PassCalcLocs(name=name),
             HeteroAnalysisToDb(wf_name=spec['wf_name'],
                                db_file=db_file,
                                task_label=name,
                                additional_fields={
                                    "tags": spec['tags']
                                })]

        super(HeteroOptimizeFW, self).__init__(t, spec=spec, parents=parents, 
                                               name=name, **kwargs)

class SubstrateSlabFW(Firework):
    def __init__(self, spec, structure, slab_params, vasp_input_set=None,
                 user_incar_settings=None, prev_calc_dir=None, vasp_cmd=">>vasp_cmd<<",
                 name="Slab Structure Optimization", copy_vasp_outputs=True, db_file=None,
                 parents=None, **kwargs):
        """
        Apply the transformations to the bulk structure, write the slab
        set corresponding to the transformed structure, and run vasp.  Note 
        that if a transformation yields many structures from one, only the 
        last structure in the list is used. By default all structures will
        have selective dynamics tags as one of the transformations applied
        to the input structure.
        
        Args:
            name (string): Name for the Firework.
            spec (dict): The specification parameters used to control the 
                workflow and pass variables.
            structure (Structure): Bulk input structure to apply transformations
                onto.
            slab_params (dict): A dictionary containing a list of transformations 
                and transformation_params to generate the substrate slab 
                structure for the hetero_interface.
                Example: slab_params = {'transformations': ['SlabTransformation',
                'AddSitePropertyTransformation'],'transformation_params': [{},{}]}.
                Definitions: transformations (list): list of names of transformation 
                classes as defined in the modules in pymatgen.transformations.
                transformation_params (list): list of dicts where each dict specify 
                the input parameters to instantiate the transformation 
                class in the transformations list.

        Other Parameters:
            user_incar_settings (dict): Input settings to update the settings
                for the vasp input set.
            vasp_input_set (VaspInputSet): VASP input set, used to write the 
                input set for the transmuted structure. Defaults to 
                CMDLInterfaceSet.
            vasp_cmd (string): Command to run vasp.
            copy_vasp_outputs (bool): Whether to copy outputs from previous 
                run. Defaults to True.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (string): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or 
                list of FWs.
            kwargs: Other kwargs that are passed to Firework.__init__.
        """
        fw_name = "{}: {}".format(structure.composition.reduced_formula, name)  # task_label

        # Substrate Slab Set Up
        # vasp input settings
        user_incar_settings = user_incar_settings or None
        vasp_input_set = vasp_input_set or CMDLInterfaceSet(structure,
                                                            auto_dipole=True, user_incar_settings=user_incar_settings,
                                                            vdw='optB88', iface=True)

        # slab settings
        transformations = slab_params.get('transformations')
        transformation_params = slab_params.get('transformation_params')

        # Firetasks Set Up
        t = []
        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, contcar_to_poscar=True))
            t.append(WriteSlabStructureIOSet(transformations=transformations,
                                             transformation_params=transformation_params,
                                             prev_calc_dir=".", vasp_input_set=vasp_input_set))
        elif copy_vasp_outputs:
            t.append(CopyVaspOutputs(calc_loc=True, contcar_to_poscar=True))
            t.append(WriteSlabStructureIOSet(structure=structure,
                                             transformations=transformations,
                                             transformation_params=transformation_params,
                                             vasp_input_set=vasp_input_set,
                                             prev_calc_dir="."))
        elif structure:
            t.append(WriteSlabStructureIOSet(structure=structure,
                                             transformations=transformations,
                                             transformation_params=transformation_params,
                                             prev_calc_dir=None,
                                             vasp_input_set=vasp_input_set))
        else:
            raise ValueError("Must specify structure or previous calculation")
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, handler_group=handler))
        t.append(PassCalcLocs(name=fw_name))
        t.append(HeteroAnalysisToDb(wf_name=spec['wf_name'],
                                    db_file=db_file,
                                    task_label=fw_name,
                                    additional_fields={
                                        "tags": spec['tags'],
                                        "transmuter_input": {"transformations": transformations,
                                                             "transformation_params": transformation_params}
                                    }))
        super(SubstrateSlabFW, self).__init__(t, spec=spec, parents=parents, 
                                              name=fw_name, **kwargs)

class GenHeteroStructuresFW(Firework):
    def __init__(self, spec, structure, heterotransformation_params,
                 vasp_input_set=None, user_incar_settings=None, prev_calc_dir=None, 
                 name="Generate Heterostructures", vasp_cmd=">>vasp_cmd<<",
                 db_file=">>db_file<<", copy_vasp_outputs=False,
                 parents=None, **kwargs):
        """
        Apply transformations to 2D material and substrate slab to generate 
        fireworks that create and relax hetero_interface structures using
        the vdW corrections. Note: If the transformation produces many from 
        one, all structures are simulated.

        Args:
            spec (dict): Specification of the job to run.
            structure (Structure): 2D material to align onto the substrate.
            hetero_transformation_params (dict): dictionary containing the 
                input to control the hetero2d.manipulate.hetero_transmuter
                modules.
            transformations (list): list of transformations to apply
                to the structure as defined in the modules in hetero2d.manipulate module.
            transformation_params (list): list of dicts where each dict specifies
                the input parameters to instantiate the transformation class in the 
                transformations list. Example: h_params={'transformations':
                ['hetero_interfaces'], 'transformation_params':[{hetero_interface parameters
                dictionary}]}.

        Other Parameters:
            user_incar_settings (dict): Input settings to update the settings
                for the vasp input set.
            vasp_input_set (VaspInputSet): VASP input set, used to write the 
                input set for the transmuted structure. Defaults to CMDLInterfaceSet.
            name (string): Name for the Firework. Default is "Generate 
                HeteroStructures {2D}-on-{Substrate} {unique_id}".
            vasp_cmd (string): Command to run vasp.
            copy_vasp_outputs (bool): Whether to copy outputs from previous run.
                Defaults to False.
            prev_calc_dir (str): Path to a previous calculation to copy data from.
            db_file (string): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            kwargs: Other kwargs that are passed to Firework.__init__.
        """
        struct_sub = spec["struct_sub"]  # only used to set correct composition

        # FW Name
        fw_name = "{}: {}-on-{}: {}".format(name,
                                            structure.composition.reduced_formula,
                                            struct_sub.composition.reduced_formula,
                                            spec['unique_id'])

        # VASP Input
        user_incar_settings = user_incar_settings or None
        vasp_input_set = vasp_input_set or CMDLInterfaceSet(structure,
                                                            auto_dipole=True, user_incar_settings=user_incar_settings,
                                                            vdw='optB88', iface=True)

        # Create hetero_interface Structure Firetask
        t = []
        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, contcar_to_poscar=True))
            # creates hetero_interface structures and then relaxes structures
            t.append(CreateHeterostructureTask(spec=spec, 
                                               structure=structure,
                                               heterotransformation_params=heterotransformation_params,
                                               vasp_input_set=vasp_input_set,
                                               prev_calc_dir=".",
                                               name=fw_name))
        elif copy_vasp_outputs:
            t.append(CopyVaspOutputs(calc_loc=True, contcar_to_poscar=True))
            # creates hetero_interface structures and then relaxes structures
            t.append(CreateHeterostructureTask(spec=spec, 
                                               structure=structure,
                                               heterotransformation_params=heterotransformation_params,
                                               vasp_input_set=vasp_input_set,
                                               prev_calc_dir=".",
                                               name=fw_name))
        elif structure:
            # creates hetero_interface structures and then relaxes structures
            t.append(CreateHeterostructureTask(spec=spec, 
                                               structure=structure,
                                               heterotransformation_params=heterotransformation_params,
                                               vasp_input_set=vasp_input_set,
                                               name=fw_name))
        super(GenHeteroStructuresFW, self).__init__(t, spec=spec, parents=parents,
                                                    name=fw_name, **kwargs)


class HeteroStructuresFW(Firework):
    def __init__(self, spec, structure, name=None, transformation=None,
                 vasp_input_set=None, user_incar_settings=None,
                 vasp_cmd=">>vasp_cmd<<", db_file=">>db_file<<",
                 copy_vasp_outputs=False, prev_calc_dir=None,
                 parents=None, **kwargs):
        """
        Relax the hetero_structures generated by CreateHeterostructureTask 
        and perform various energetic analysis to store in database.
        Reads structure from POSCAR if no structure provided. 

        Args:
            spec (dict): The specification dictionary used to control the
                firework analysis. Spec file boolean triggers, Binding_Energy
                and Adsorption_Energy, must be provided to perform the analysis.
            structure (Slab): Heterostructure slab structure.

        Other Parameters:
            transformation (dict): The list of dictionaries containing the
                input used to create the hetero_interfaces. Defaults to none.
            name (str): The name for this firework. Defaults to wf_name + 
                heterostructure optimization (used as the fw_name).
            transformation (dict): transformation parameters used to create
                the heterostructure.
            user_incar_settings (dict): Input settings to update the settings
                for the vasp input set.
            vasp_input_set (VaspInputSet): VASP input set for the transformed 
                2d on sub-slab. Default CMDLInterfaceSet.
            prev_calc_dir: path to previous calculation if using structure 
                from another calculation.
        """
        fw_name = "{}: {}".format("Heterostructure Optimization", name)

        # Update VASP input params
        user_incar_settings = user_incar_settings or None
        vasp_input_set = vasp_input_set or CMDLInterfaceSet(structure,
                                                            auto_dipole=False, iface=True, vdw='optB88',
                                                            user_incar_settings=user_incar_settings)

        # Analysis Flags for HeteroAnalysisToDb
        formation_energy = spec.get('Formation_Energy', True)
        binding_energy = spec.get('Binding_Energy', True)
        adsorption_energy = spec.get('Adsorption_Energy', True)

        # QueueAdapter: SET WALL_TIME
        spec_wall_time = spec.get('wall_time_hetero', None)
        if not spec_wall_time:
            wall_time = 172800
        else:
            h, m, s = spec_wall_time.split(':')
            wall_time = int(h) * 3600 + int(m) * 60 + int(s)

        # Set FW for simulations
        t = [FileWriteTask(files_to_write=[{"filename": get_slug(fw_name), "contents": ""}])]
        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, contcar_to_poscar=True))
            t.append(WriteHeteroStructureIOSet(structure=structure,
                                               nlayers_sub=transformation["nlayers_sub"],
                                               nlayers_2d=transformation["nlayers_2d"],
                                               vasp_input_set=vasp_input_set))
        elif copy_vasp_outputs:
            t.append(CopyVaspOutputs(calc_loc=True, contcar_to_poscar=True))
            t.append(WriteHeteroStructureIOSet(structure=structure,
                                               nlayers_sub=transformation["nlayers_sub"],
                                               nlayers_2d=transformation["nlayers_2d"],
                                               vasp_input_set=vasp_input_set))
        elif structure:
            t.append(WriteHeteroStructureIOSet(structure=structure,
                                               nlayers_sub=transformation["nlayers_sub"],
                                               nlayers_2d=transformation["nlayers_2d"],
                                               vasp_input_set=vasp_input_set))
        else:
            raise ValueError("Must specify structure or previous calculation")
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, handler_group=handler, wall_time=wall_time))
        t.append(PassCalcLocs(name=fw_name))
        t.append(HeteroAnalysisToDb(wf_name=spec['wf_name'],
                                    db_file=db_file,
                                    task_label=fw_name,
                                    Formation_Energy=formation_energy,
                                    Binding_Energy=binding_energy,
                                    Adsorption_Energy=adsorption_energy,
                                    additional_fields={
                                        "tags": spec['tags'],
                                        "heterotransformation": transformation
                                    }))

        super(HeteroStructuresFW, self).__init__(t, spec=spec, parents=parents,
                                                 name=fw_name, **kwargs)
