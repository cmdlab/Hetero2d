# coding: utf-8
# Copyright (c) CMD Lab Development Team.
# Distributed under the terms of the GNU License.

"""
These modules are helper functions for the workflow, fireworks, and firetasks to simulate 
the thermodynamic stability and electronic properties of 2D thin films adsorbed on a substrate surface.
Most functions here are not intented to be used outside of specific fireworks.
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import glob
import os
import re
from copy import deepcopy
from importlib import import_module

from atomate.utils.utils import get_logger, env_chk
from fireworks.core.firework import FiretaskBase, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize
from monty.json import jsanitize
from monty.serialization import dumpfn
from pymatgen.core import Structure
from six.moves import range

from hetero2d.manipulate.utils import get_mongo_client

__author__ = 'Tara M. Boland'
__copyright__ = "Copyright 2022, CMD Lab"
__maintainer__ = "Tara M. Boland"
__email__ = 'tboland1@asu.edu'
__date__ = "June 09, 2022"

logger = get_logger(__name__)

# Helper Functions
def update_spec(additional_spec):
    """
    Controller function for Hetero2d workflow to update spec, override 
    default settings, and manage data passing between workflows, fireworks,
    and firetasks.

    Args:
        additional_spec (dict): user spec settings.
        struct_2d (Structure): 2D structure.
        struct_sub (Structure): Substrate structure. Can
            be un-relaxed/relaxed bulk or the relaxed substrate slab.
        orient (list): Orientation (hkl) of the substrate. Defaults
            to orientation specified in the transformation params.

    Other Parameters:
        duplicate (bool): Enable duplicate checking 
            for the workflow. Defaults to True.
        is_bulk_optimized (bool): If True, the given
            bulk structure is assumed to be optimized.
        is_sub_optimized (bool): If True, the given 
            substrate slab is assumed to be optimized. 
            Default is False. If True, you must set 
            E_sub key in additions['analysis_info'] in
            order to calculate the E_bind. 
        is_2d_optimized (bool): If True, the given 
            struct_2d is assumed to be optimized. 
            Default is False. If True, you must provide 
            E_2d and N_2d keys in the 
            additions['analysis_info'] dict.
        is_3d2d_optimized (bool): If True, the given 
            struct_3d2d is assumed to be optimized. 
            Default is False. If True, you must provide E_3d2d
            and N_3d2d in the additions['analysis_info'] dict.
        analysis_info (dict): Dictionary that contains previous
            job information used for the analysis later.
        wf_name (str): An appropriate and unique name for 
            the workflow. This is the base name for all WFs
            choose wisely. Default to:
            HeteroInterfaces-{substrate-orient}:{2D.comp}-on-{sub.comp}.
        wall_time_hetero (str): WallTime for the heterostructure jobs.
            Must be entered in "HR:MIN:SEC" format. Defaults to 48 hrs.
        sub_nodes (int): Set the number of nodes to use for optimation
            of the substrate slab in the wf. Defaults to 1.
        iface_nodes (int): Set the number of nodes to use for optimation
            of the substrate slab in the wf. Default set to 1.
        ntasks (int): The total number of cpu-cores/processors used for 
            the calculation. Can use ntasks_per_node to set # of cpu cores
            to use per node. Used to set the appropriate number of cores 
            for the hetero_structure configurations. Defaults to ntasks 
            in my_qadpater.yaml.
        ntasks_per_node (int): The number of cpu-cores/processors per
            node. Used to set the appropriate number of cores per node
            for the hetero_structure configurations. Defaults to 
            ntasks_per_node set in my_qadpater.yaml.
        Formation_Energy (bool): To perform analysis of the formation
            energy of the 2D material. Defaults to True.
        Binding_Energy (bool): To perform analysis of the binding energy
            of the 2D/substrate slab. Defaults to True.
        Adsorption_Energy (bool): To perform analysis of the adsorption
            formation energy of the 2D/substrate slab. Defaults to True.

    Returns:
        spec dict
    """
    # copy data b/c we delete entries
    c_spec = deepcopy(additional_spec)

    # process default data passed to the function
    sub = c_spec.pop('struct_sub')
    td = c_spec.pop('struct_2d')
    orient = c_spec.pop('orient')
    unique_id = c_spec.get('unique_id')
    layers = c_spec.pop('nlayers_sub')

    # set up default spec
    default_spec = {
        "_preserve_fworker": True,
        "duplicate": True,
        "is_bulk_optimized": False,
        "is_sub_optimized": False,
        "is_2d_optimized": False,
        "is_3d2d_optimized": False,
        "Formation_Energy": True,
        "Binding_Energy": True,
        "Adsorption_Energy": True,
        "nlayers_sub": layers,
        "wf_name": "{}-{}: {}-on-{}: {}".format('Heterointerfaces', ''.join(str(i) for i in orient),
                                                td.composition.reduced_formula, sub.composition.reduced_formula,
                                                unique_id),
        "orient": ''.join(str(i) for i in orient),
        "wall_time_hetero": '48:00:00', "struct_sub": sub
    }

    # update the spec up additional specs
    default_spec.update(c_spec)

    # duplicate checking 
    if default_spec.get('duplicate'):
        default_spec.update({"_pass_job_info": True,
                             "_dupefinder": {"_fw_name": "DupeFinderExact"}})
    return default_spec

@explicit_serialize
class TransferSlabTask(FiretaskBase):
    """
    This class updates the spec from the SubstrateSlabFW
    with the relaxed oriented substrate slab.
    
    Args:
        label (str): The name of the structure in the 
            additional_spec dict - struct_sub.

    Returns:
        FWAction updating the spec
    """
    required_params = ['label']

    def run_task(self, fw_spec):
        struct_sub = fw_spec.get('struct_sub')
        # get relaxed struct_sub and update fw_spec
        # glob finds files named "" the [] tells it what file to copy
        f = glob.glob("CONTCAR*")[-1]
        # this takes the contcar and preps it for updating fw_spec
        s = Structure.from_file(f, False)
        # update the fw_spec
        update_spec = {"struct_sub": s}

        return FWAction(update_spec=update_spec)

@explicit_serialize
class CreateHeterostructureTask(FiretaskBase):
    """
    Applies provided transformations to the input 2D structure and 
    substrate slab and writes the VASP input for the calculation. Reads 
    structure from previous the directory if no structure is provided. 

    Updates the Q_Adapter settings for the hetero_interface optimization. Updates
    spec file to include to analysis_info, the previous job info, and enables
    duplicate checking.
    
    Args:
        heterotransformation_params (dict): dictionary containing the input to
            control the hetero2d.manipulate.hetero_transmuter modules. Example:
            h_params={'transformations':['hetero_interfaces'], 'transformation_params':
            [{hetero_inface input}]}.
        transformations (list): list of transformations to apply to the structure as 
            defined in the modules in hetero2d.manipulate module.
        transformation_params (list): list of dicts where each dict specifies
            the input parameters to instantiate the transformation class
            in the transformations list. The transformations input parameters should
            correspond to the dictionary keys where the values represent the input
            data.
        vasp_input_set (vis): User defined vasp input set for the heterostructure.

    Other Parameters:
        name (str): The firework name for this step of the workflow.

    Returns:
        HeterostructureFW's for each unique structure generated by the heterotransformation_params
    """
    required_params = ["vasp_input_set", "heterotransformation_params"]
    optional_params = ["name","spec","structure"]

    def run_task(self, fw_spec):
        fw_name = self.get('name', "CreateHeterostructureTask: {}".format(
            fw_spec['unique_id']))

        # optimize substrate slab structure or pull from spec
        if not fw_spec.get('is_sub_optimized'):
            locs = fw_spec.get('calc_locs')
            sub_loc = [i['path'] for i in fw_spec['calc_locs']
                       if re.search('Slab Structure Optimization-\d', i['name'].split(':')[1].strip())][0]
            struct_sub = Structure.from_file(glob.glob(os.path.join(sub_loc, "CONTCAR*"))[-1])
            logger.info('Pulling struct_sub from calc_locs:', sub_loc)
        else:
            struct_sub = fw_spec.get('struct_sub')
            logger.info('Pulling struct_sub from fw_spec.')

        # optimized 2d structure or pull from spec
        if not fw_spec.get('is_2d_optimized'):
            td_loc = [i['path'] for i in fw_spec['calc_locs']
                      if re.search(' 2D Structure Optimization', i['name'])][0]
            struct_2d = Structure.from_file(glob.glob(os.path.join(td_loc, "CONTCAR*"))[-1])
            logger.info('Pulling structure from calc_locs:', td_loc)
        else:
            pat = '{{hetero2d.firetasks.heteroiface_tasks.CreateHeterostructureTask}}'
            struct_2d = [Structure.from_dict(i['structure']) for i in fw_spec['_tasks']
                         if re.search(pat, i['_fw_name'])][0]
            logger.info('Pulling structure from fw_spec:', pat)

        # pull structures from spec file
        td = fw_spec.get('struct_2d')
        sub = fw_spec.get('struct_sub')

        # gather hetero_interface transformation params
        h_params = self['heterotransformation_params']
        nlayers_2d = h_params['transformation_params'][0]['nlayers_2d']

        # create heterotransformation_params: all structs updated
        [i.update({'struct_2d': struct_2d, 'struct_sub': struct_sub})
         for i in h_params['transformation_params']]

        # Create Hetero-Interface
        # setup heteroiface transformations
        transformations = []
        t_params = h_params.get("transformation_params",
                                [{} for i in range(len(h_params["transformations"]))])

        # store trans_params in db
        params = deepcopy(t_params[0])
        # create hetero_interfaces
        for t in h_params["transformations"]:
            found = False
            for m in ["heterotransmuter"]:
                mod = import_module("hetero2d.manipulate")
                try:
                    t_cls = getattr(mod, t)
                except AttributeError:
                    continue
                t_obj = t_cls(**t_params.pop(0))
                transformations.append(t_obj)
                found = True
            if not found:
                raise ValueError(
                    "Could not find transformation: {}".format(t))
        transformation = transformations[0].copy()
        final_structures = [deepcopy(struct) for struct in transformation]
        alignment_info = final_structures.pop(-1)

        # create json files for final structures
        for idx, struct in enumerate(final_structures):
            dumpfn(struct, "heterointerfaces_config_{}.json".format(idx))

        # Setup/Clean up FW Spec For Next Calcs
        # Q_Adapter Settings Update
        iface_nodes = fw_spec.get("iface_nodes", 1)
        wall_time_hetero = fw_spec.get("walltime_hetero", fw_spec['wall_time_hetero'])

        adapter_keys = ['ntasks', 'ntasks_per_core', 'exclude_nodes', 'cpus_per_task',
                        'gres', 'qos', 'queue', 'account', 'job_name', 'license', 'constraint', 'signal',
                        'mem', 'mem_per_cpu']
        update_queue = {key: fw_spec.get(key, False) for key in adapter_keys if fw_spec.get(key, False)}
        update_queue.update({"walltime": wall_time_hetero, "nodes": iface_nodes})
        fw_spec["_queueadapter"] = update_queue

        # clean up analysis_info in spec
        analysis_info = {}
        [analysis_info.update(entry) for entry in fw_spec.get('analysis_info', {})]
        fw_spec['analysis_info'] = [analysis_info]

        # add hetero_interface specific tags #
        fw_spec['tags'].update({'max_mismatch': str(params['max_mismatch']),
                                'max_area': str(params['max_area']), 'r1r2_tol': str(params['r1r2_tol']),
                                'alignment_info': alignment_info})

        # Create HeteroStructuresFW to Opt Structs
        # Configure VASP Input
        vis_orig = self["vasp_input_set"]
        vis_dict = vis_orig.as_dict()
        uis = vis_dict['user_incar_settings']

        # add new npar & kpar computed above
        npar = uis.get("NPAR", 2)
        uis["NPAR"] = npar
        vis_dict['user_incar_settings'] = uis
        vasp_input_set = vis_orig.__class__.from_dict(vis_dict)

        # Relaxation FW
        # create FWs for each hetero_interface configuration
        from hetero2d.fireworks.core import HeteroStructuresFW
        new_fws = []
        for idx, structure in enumerate(final_structures):
            # spec, struct, name, transformation, analysis, vis
            # uis, tags, wyc-based iface name
            w_name = structure.site_properties['name'][0]
            h_spec = deepcopy(fw_spec)
            prefix = ':'.join(fw_spec['wf_name'].split(':')[0:2])
            h_name = "{}: Config-{}: {}".format(prefix, idx, fw_spec['unique_id'])
            h_spec['tags'].update({"Interface Config": str(idx),
                                   'iface composition': structure.composition.formula,
                                   'Wyckoff Name': w_name})
            fw = HeteroStructuresFW(spec=h_spec,
                                    structure=structure, name=h_name,
                                    vasp_input_set=vasp_input_set,
                                    transformation=params)
            new_fws.append(fw)

        # Store Generate heterostructures in DB #
        f = glob.glob("heterointerfaces_config_*")
        calc_dir = os.getcwd()
        num_cfs = len(f)
        # tags
        tags = fw_spec['tags']
        tags.update({'unique_id': fw_spec['unique_id']})
        # standard database information
        heterostructure_dict = {'dir_name': calc_dir, 'number configs': num_cfs,
                                'wf_name': fw_spec['wf_name'], 'task_label': fw_name, 'tags': tags}
        heterostructure_dict = jsanitize(heterostructure_dict)

        db_file = env_chk('>>db_file<<', fw_spec)
        db_type = fw_spec.get('db_type', None)
        conn, database = get_mongo_client(db_file, db_type=db_type)
        db = conn[database]
        col = db['Generate']
        col.insert_one(heterostructure_dict)
        conn.close()
        return FWAction(additions=new_fws)
