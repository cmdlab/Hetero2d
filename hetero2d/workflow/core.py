# coding: utf-8
# Copyright (c) CMD Lab Development Team.
# Distributed under the terms of the GNU License.

"""
This module sets up the vasp input files for the workflow to simulate 
2D materials adsorbed on a substrate slab.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
from copy import deepcopy
import re

from pymatgen import Structure
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer

from fireworks import Firework
from fireworks.core.firework import Workflow
from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact

from atomate.utils.utils import get_logger
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.vasp.powerups import add_namefile
from atomate.vasp.fireworks.core import StaticFW

from hetero2d.manipulate.utils import center_slab, change_Tasks
from hetero2d.firetasks.heteroiface_tasks import _update_spec
from hetero2d.firetasks.parse_outputs import HeteroAnalysisToDb
from hetero2d.io import CMDLInterfaceSet, CMDLElectronicSet
from hetero2d.fireworks.core import HeteroOptimizeFW, SubstrateSlabFW, GenHeteroStructuresFW, \
    ElectronicFW


__author__ = 'Tara M. Boland'
__copyright__ = "Copyright 2020, CMD Lab"
__maintainer__ = "Tara M. Boland"
__email__ = 'tboland1@asu.edu'

logger = get_logger(__name__)

def get_heterostructures_stabilityWF(struct_2d, struct_sub, struct_3d2d, heterotransformation_params, 
        slab_params, user_additions, tags={}, bin_2d=VASP_CMD, bin_3d=VASP_CMD, dipole=None, uis=None,
        uis_2d=None, vis_2d=None, uis_3d2d=None, vis_3d2d=None, uis_bulk=None, vis_bulk=None, 
        uis_trans=None, vis_trans=None, uis_iface=None, vis_iface=None):
    """
    Relax reference structures to determine energetic parameters related
    to the formation of 2D films adsorbed onto substrates. Heterointerfaces 
    are created from the relaxed 2D and substrate slabs which generate a 
    symmetry matched, low lattice mismatch between the 2D material and 
    substrate slab. The user should be familiar with the behaviour of 
    heterointerface function before runnings simulations (load 
    hetero2d.manipulate.heterotransmuter import hetero_interfaces). Default 
    relaxation set CMDLInterfaceSet. Specify a complete vasp input 
    set or update the vis set using the uis_i tag to change vasp behaviour.
    It is also possible to use the general uis to update all vis.
    
    Args:
        struct_2d (Structure): The primitive unit cell of the 2D structure
            to adsorb onto the substrate slab.
        struct_sub (Structure): The substrate (bulk phase or substrate slab)
            to adsorb the 2D material.
        struct_3d2d (Structure): The bulk reference phase for the 2D 
            material.
        heterotransformation_params (list): A list of dictionaries where 
            the keys represent the arguments of the hetero_interfaces function
	          and the value is the argument excluding the structures. Dictionary
	          keys [{hetero_interfaces function args: value}]. See hetero_interfaces
	          module for args.
        slab_params (dict): Same parameter format as the TransmuterFW to 
            create a substrate slab. 
        user_additions (dict): A specification to control the workflow. 
            See firetasks.heteroiface_tasks._update_spec for a detailed list 
            of parameters. 
        bin_2d (str): VASP run command for the VASP version compiled to restrict
            vacuum spacing in z direction from shrinking artifically.
	      bin_3d (str): VASP run command for the VASP version compiled normally.
    
    Other Parameters:
        dipole (bool): If True, dipole corrections will be used for all slab 
            calculations. Defaults to True.
        tags (dict): A dictionary applying tags to - general (all), 2d, 
            3d2d, bulk, transmuter, and iface fireworks.
        uis (dict): A dictionary of general user incar settings you wish to 
            apply to every simulation. Defaults to None.
        uis_I (dict): A dictionary of INCAR settings to override the 
            VASPInputSet defaults or set additional commands for the Ith 
            structure where I = 2d, 3d2d, bulk, trans, and iface.
        vis_I (VASPInputSet): A VASP input set to relax the Ith materials 
            where I = 2d, 3d2d, bulk, trans, and iface. Defaults to 
            CMDLInterfaceSet.

    Returns:
        Heterostructure workflow
    
    """
    ### Vasp INPUT SET ###
    # set up default spec
    user_additions = user_additions or {}
    unique_id = user_additions.get('unique_id')
    sub_nodes = user_additions.pop('sub_nodes', 1)
    user_additions.update({'struct_sub': struct_sub, 
        'struct_2d': struct_2d, 'nlayers_sub': 
        heterotransformation_params[0]['nlayers_sub']})
    
    # spec workflow controls
    spec = _update_spec(additional_spec=user_additions)
    # set up the vdw corrections
    vdw = user_additions.pop('vdw', 'optB88')
    # optimization flags
    is_bulk_optimized = spec.get('is_bulk_optimized')
    is_2d_optimized = spec.get('is_2d_optimized')
    is_3d2d_optimized = spec.get('is_3d2d_optimized')
    is_sub_optimized =  spec.get('is_sub_optimized')
    
    # Vasp Input and wf input from spec
    wf_name = spec["wf_name"]
    dipole = True if dipole == None else dipole
    bin_2d = bin_2d or VASP_CMD
    bin_3d = bin_3d or VASP_CMD

    # 2d symmetry data
    sga = SpacegroupAnalyzer(struct_2d)
    sg_film = {}
    [sg_film.update({key:sga.get_symmetry_dataset()[key]}) 
        for key in ['number','hall_number','international',
        'hall','pointgroup']]
    # bulk symmetry data
    sga = SpacegroupAnalyzer(struct_sub)
    sg_sub = {}
    [sg_sub.update({key:sga.get_symmetry_dataset()[key]}) 
        for key in ['number','hall_number','international',
        'hall','pointgroup']]

    # links dict
    input_fws = []
    links_dict = {}
    
    #####################
    ### OPTIMZIE BULK ###
    if not is_bulk_optimized:
        name = "Bulk Structure Optimization: {}".format(unique_id)

        # symmetry information
        sa_bulk = SpacegroupAnalyzer(struct_sub) #slab compt struct
        struct_sub = sa_bulk.get_conventional_standard_structure() 
       
        # VASP calculation controls
        uis_bulk = uis if not uis_bulk else uis_bulk
        vis_bulk = vis_bulk or CMDLInterfaceSet(struct_sub, 
            vdw = vdw, iface = False, auto_dipole = False, 
            user_incar_settings = uis_bulk)
       
        # database tags
        tags_bulk = {'composition': struct_sub.composition.reduced_formula, 
                'task_label': name, 'wf_name': wf_name, 
                'spacegroup': sg_sub, 'unique_id': unique_id}
        [tags_bulk.update(tags.get(i,{})) for i in ['bulk','general'] ]
        
        # make bulk spec 
        spec_bulk = deepcopy(spec)
        spec_bulk['tags'] = tags_bulk

        # optimize bulk structure firework
        input_fws.append(HeteroOptimizeFW(spec=spec_bulk, 
            structure=struct_sub, name=name,
            vasp_input_set=vis_bulk, vasp_cmd=bin_3d,
            db_file=DB_FILE)) 

        # update links dict
        links_dict.update({'bulk':{'fws':input_fws[0]}})
    #####################        
    
    ###############################
    ### OPTIMIZE SUBSTRATE SLAB ###    
    if not is_sub_optimized:
        name = "Slab Structure Optimization-{}: {}".format(spec['orient'], 
            unique_id)
        
        # VASP calculation controls
        uis_trans = uis if not uis_trans else uis_trans
        vis_trans = vis_trans or CMDLInterfaceSet(struct_sub, 
            vdw = vdw, iface = True, auto_dipole = dipole, 
            user_incar_settings = uis_trans)
        
        # database tags
        tags_trans = {'substrate_composition': struct_sub.composition.reduced_formula,
            'surface_plane': spec.get('orient'), 'unique_id': unique_id,
            'wf_name': wf_name, 'spacegroup': sg_sub}
        [tags_trans.update(tags.get(i,{})) 
             for i in ['trans','general']]
        
        # make substrate slab spec 
        spec_trans = deepcopy(spec)
        spec_trans['_queueadapter']={'nodes': sub_nodes}
        spec_trans['tags'] = tags_trans

        # optimize substrate slab structure firework
        copy_vasp_outputs = False if is_bulk_optimized else True
        input_fws.append(SubstrateSlabFW(spec=spec_trans, name=name, 
                structure=struct_sub, slab_params=slab_params, 
                vasp_input_set=vis_trans, vasp_cmd=bin_3d,
                db_file=DB_FILE, copy_vasp_outputs=copy_vasp_outputs))
        
        # update links dict
        if not is_bulk_optimized:
            links_dict['bulk']['links']={input_fws[0]:
                [input_fws[len(input_fws)-1]]}
        links_dict.update({'trans':{ 'fws': input_fws[len(input_fws)-1]}})
    ###############################
    
    ######################
    ### OPTIMIZE 2D FW ###
    if not is_2d_optimized:
        name = "2D Structure Optimization: {}".format(unique_id)
       	
        # center 2d material 
        struct_2d = center_slab(struct_2d)	
	
        # VASP calculation controls
        uis_2d = uis if not uis_2d else uis_2d
        vis_2d = vis_2d or CMDLInterfaceSet(struct_2d, 
            auto_dipole = dipole, iface=False, vdw = vdw,
            user_incar_settings = uis_2d)
       
        # database tags
        tags_2d = {'composition': struct_2d.composition.reduced_formula,
            'task_label': name, 'wf_name': wf_name, 'spacegroup': sg_film,
            'unique_id': unique_id}
        [tags_2d.update(tags.get(i,{})) for i in ['2d','general']]
        
        # make 2D spec
        spec_2d = deepcopy(spec)
        spec_2d['tags'] = tags_2d
        
        # optimize 2D structure firework
        input_fws.append(HeteroOptimizeFW(spec=spec_2d, structure=struct_2d, 
            name=name, vasp_input_set=vis_2d, vasp_cmd=bin_2d, db_file=DB_FILE))
        
        # update links dict
        links_dict.update({'2d':{'fws':input_fws[len(input_fws)-1]}}) 
    ######################
     
    ########################
    ### OPTIMIZE 3D2D FW ###
    if not is_3d2d_optimized:
        name = "3D2D Structure Optimization: {}".format(unique_id)
        
        # VASP calculation controls
        uis_3d2d = uis if not uis_3d2d else uis_3d2d
        vis_3d2d = vis_3d2d or CMDLInterfaceSet(struct_3d2d, 
            vdw = vdw, iface = False, user_incar_settings =
            uis_3d2d)

        # symmetry data
        sga = SpacegroupAnalyzer(struct_3d2d)
        sg_info = {}
        [sg_info.update({key:sga.get_symmetry_dataset()[key]}) 
            for key in ['number','hall_number','international','hall','pointgroup']]
       
        # database tags
        tags_3d2d={'composition': struct_3d2d.composition.reduced_formula, 'spacegroup': sg_info,
            'task_label': name, 'wf_name': wf_name, 'unique_id': unique_id}
        [tags_3d2d.update(tags.get(i,[])) for i in ['3d2d','general']]
        
        # make 3D2D spec
        spec_3d2d = deepcopy(spec)
        spec_3d2d['tags'] = tags_3d2d

        # optimize 3D2D structure firework
        input_fws.append(HeteroOptimizeFW(spec=spec_3d2d,
            structure=struct_3d2d, name=name, 
            vasp_input_set=vis_3d2d, vasp_cmd=bin_3d,
            db_file=DB_FILE)) 

        # update links dict
        links_dict.update({'3d2d':{'fws': input_fws[len(input_fws)-1]}}) 
        if not is_2d_optimized:
            links_dict['2d']['links']=input_fws[len(input_fws)-1]
        elif is_2d_optimized:
            links_dict.update({'3d2d':{'fws':input_fws[len(input_fws)-1]}}) 
    ####################################################################################
    
    ####################################################################################
    ### OPTIMIZE Heterostructures WF ###

    # VASP calculation controls
    h_params = {'transformation_params': 
        heterotransformation_params, 'transformations':
        ['hetero_interfaces']*len(heterotransformation_params)}
    uis_iface = uis if not uis_iface else uis_iface
    vis_iface = vis_iface or CMDLInterfaceSet(struct_2d, 
        vdw = vdw, iface = True, auto_dipole = dipole, 
        user_incar_settings = uis_iface)

    # database tags
    tags_iface = {'film_composition': struct_2d.composition.reduced_formula,
        'substrate_composition': struct_sub.composition.reduced_formula,
        'surface_plane': spec.get('orient'), 'wf_name': wf_name, 'film_spacegroup':
        sg_film, 'substrate_spacegroup': sg_sub, 'unique_id': unique_id}
    [tags_iface.update(tags.get(i,{})) for i in ['iface','general']]
  
    # make iface spec
    spec_iface = deepcopy(spec)
    spec_iface['tags'] = tags_iface
    spec['struct_2d'] = struct_2d

    # Create heterointerface structure firework
    #    FW name is assigned in the FW below
    input_fws.append(GenHeteroStructuresFW(spec=spec_iface, structure=struct_2d, 
        heterotransformation_params=h_params, vasp_input_set = vis_iface,
        vasp_cmd = bin_3d, db_file = DB_FILE))

    # update links dict
    links_dict['iface']={'fws': input_fws[len(input_fws)-1]}
    ####################################
    
    ##########################
    ## create fw links dict ##
    fw_links = {}
    # bulk children
    if not spec['is_bulk_optimized']:
        fw_links.update(links_dict['bulk']['links'])
    # trans, 2d, 3d2d children
    child_iface={'is_sub_optimized': 'trans', 
        'is_2d_optimized': '2d', 'is_3d2d_optimized': '3d2d'}
    [fw_links.update(
      {links_dict[ref]['fws']:[links_dict['iface']['fws']]}) 
         for is_opt,ref in child_iface.items() if spec[is_opt]==False]
    # iface children
    fw_links[links_dict['iface']['fws']]=[]
    # formation energy links
    if not spec['is_3d2d_optimized'] and not spec['is_2d_optimized']:
        fw_links[links_dict['2d']['fws']].append(links_dict['2d']['links'])
    ##########################
    
    ################## FINAL WF NAME ###############
    name = '{}-on-{}: hkl-[{}]: {}'.format(
        struct_2d.composition.reduced_formula,
        struct_sub.composition.reduced_formula,
        spec['orient'],spec['unique_id'])     
    wf = Workflow(fireworks=input_fws,
            links_dict=fw_links,
            name=name)
    wf = add_namefile(wf)
    print('Workflow Name:', name)
    print(wf.links)

    return wf

def wf_electronic(structure, tags={}, user_additions={}, prev_calc_dir=None,
                  dos=True, bader=True, cdd=False, user_incar_settings=None,
                  vasp_input_set=None, vasp_cmd=VASP_CMD, **kwargs):
    '''
    Workflow to obtain 1) site-orbital projected DOS, 2) Bader analysis, and 3) 
    charge density difference for the given structure. If prev_calc_dir is 
    specified the StaticFW is skipped and a NonSCFFW is performed with reciprocal 
    k-point density = 100 and NGiF grids 2x the previous values to converge the 
    charge density for Bader analysis. If cdd is selected the structure spawns 3 
    separate fireworks dividing up the input structure.  
    
    Args:
        structure (Structure): Input structure. Used to name the workflow if 
            prev_calc_dir is specified.
        tags (dict): A dictionary of tags for the workflow. Can add 
            individual tags for each structure when doing charge density difference
            calculations using 'iso_1' and 'iso_2' keys.
        user_additions (dict): A dictionary specifying addtional information and 
            settings for the workflow.  Must provide the 'unique_id'. To override 
            vasp_input_set defaults for ElectronicFW use 'electronic_set_overrides'
            key. Any valid keys for MPNonSCFSet.from_prev_calc() are valid.
            Example: {'grid_density': '0.03 A spacing between points in NGiF grid',
            'unique_id': 1, 'split_idx':{'iso_1':[], 'iso_2':[]}, 'dedos': 0.05,
            'electronic_set_overrides': }.
            
    Other parameters:
        prev_calc_dir (str): A path specifying the previous calculation directory.
            Retrieves previous calculation output to perform DOS, bader analysis, 
            and write input files. If None, will create new static calculation 
            using the provided structure.
        dos (bool): If True, peforms high quality site-orbital projected density of
            states. Default set to True.
        bader (bool): If True, peforms high quality bader analysis for the given 
            structure. Set 'grid_density' in 'user_additions' to increase the NGiF 
            grids. Defaults to True with 2x NGiF grid default set by VASP.
        cdd (bool): Compute the charge density difference for the given structure. 
            If True, supply the 'split_idx' key in 'user_additions'. 'split_idx' 
            can be the dictionary output from hetero2d.manipulate.util.tag_iface or
            2 lists of atom indices under the keys 'iso_1' and 'iso_2'. The indices 
            produce 2 isolated structures whose charge density is substracted from
            the combined structure. Default set to False.
        user_incar_settings (dict): INCAR parameters to override StaticFW defaults.
        vasp_input_set (VaspInputSet): VASP input set, used to write the input set
            for the VASP calculation. Defaults to CMDLInterfaceSet for StaticFW and 
            MPNonSCFSet if prev_calc_dir.
        vasp_cmd (str): Command to run vasp. 
        **kwargs (keyword arguments): Other kwargs that are passed to 
            Firework.__init__ applied to ElectronicFW.
    '''    
    if not prev_calc_dir and not structure: # ensure struct/prev_calc present 
        raise ValueError("Must specify structure or previous calculation.")
    
    wf_name = "{}-Electronic Properties: {}".format(
        structure.composition.reduced_formula, 
        user_additions['unique_id']) # workflow name
    
    # STATIC INCAR 
    uis = {"LCHARG": True, "IBRION": -1, "ISMEAR": 0, "SIGMA": 0.05, "NSW": 0}
    uis_static = uis.update(user_incar_settings) if user_incar_settings else uis
    vis_static = vasp_input_set or CMDLInterfaceSet(structure,
        user_incar_settings=uis_static)
    electronic_set_overrides = user_additions.get('electronic_set_overrides', {})
    
    # create tags
    tags.update({'wf_name': wf_name})  # update tags with wf name
    tags_2d, tags_sub = [tags.pop(sub_tags) if sub_tags in tags.keys() else {} 
                         for sub_tags in ['iso_1', 'iso_2'] ]
    
    fws = []
    # STATIC CALCULATION: no prev data; combined system
    if not prev_calc_dir: # no prev dir; ToDb is removed
        static = StaticFW(structure=structure, 
            name='Static: {}'.format(user_additions['unique_id']), 
            vasp_input_set=vis_static, 
            vasp_cmd=vasp_cmd,
            prev_calc_loc=None,
            prev_calc_dir=prev_calc_dir,
            db_file=DB_FILE)
        scf = static
        fws.append(static)
    else:
        scf = None
    
    # NONSCF CALCULATION: from previous dir
    # a basic DOS and Bader calc
    if not cdd: 
        electronic = ElectronicFW(name='NSCF: DOS and Bader: {}'.format(user_additions['unique_id']), 
            structure=structure, 
            dedos=user_additions.get('dedos', 0.05),
            grid_density=user_additions.get('grid_density', 0.03),
            tags=tags,
            dos=dos,
            bader=bader,
            cdd=cdd,
            parents=scf, 
            prev_calc_dir=prev_calc_dir,
            vasp_cmd=vasp_cmd,
            db_file=DB_FILE, 
            electronic_set_overrides=electronic_set_overrides,
            **kwargs)
        fws.append(electronic)    
    # Charge Density Difference
    elif cdd: 
        split_idx = user_additions.get('split_idx', None)
        if split_idx == None: # raise error if cdd dict not present
            raise ValueError("Must specify dictionary 'split_idx':{} in user_additions.")
        
        # get iso_1, iso_2 atom indices
        if len(split_idx) == 2:
            idx_2d = split_idx.get('iso_1')
            idx_sub = split_idx.get('iso_2')
        else:
            # get a list of atom ids for the 2d material and substrate
            idx_2d = [el_v for k,v in split_idx.items() if re.search('2d_layer_[\d]', k)
                for el_v in v ] 
            idx_sub = list(set(range(0, structure.num_sites)).difference(set(idx_2d)))

        # generate iso_1 and iso_2 from combined system
        struct_sub = Structure.from_sites(sites=[structure[i] for i in idx_sub],
            to_unit_cell=True)
        struct_2d = Structure.from_sites(sites=[structure[i] for i in idx_2d],
            to_unit_cell=True)

        # NONSCF CALCULATION: COMBINED
        cdd_combined = ElectronicFW(name='Combined NSCF: {}'.format(user_additions['unique_id']), 
            structure=structure, 
            dedos=user_additions.get('dedos', 0.05),
            grid_density=user_additions.get('grid_density', 0.03),
            tags=tags,
            dos=dos,
            bader=bader,
            cdd=False,
            parents=scf, 
            prev_calc_dir=prev_calc_dir,
            vasp_cmd=vasp_cmd,
            db_file=DB_FILE, 
            electronic_set_overrides=electronic_set_overrides,
            **kwargs)
        
        # STATIC CALC: ISO_1 ISO_2 (2d and substrate)
        tags_2d.update(tags)
        tags_sub.update(tags)
        static_2d = StaticFW(structure=struct_2d, 
            name='ISO 1 Static: {}'.format(user_additions['unique_id']), 
            vasp_input_set=vis_static, 
            vasp_cmd=vasp_cmd,
            prev_calc_loc=None,
            prev_calc_dir=None,
            parents=None,
            db_file=DB_FILE)
        static_sub = StaticFW(structure=struct_sub, 
            name='ISO 2 Static: {}'.format(user_additions['unique_id']), 
            vasp_input_set=vis_static, 
            vasp_cmd=vasp_cmd,
            prev_calc_loc=None,
            prev_calc_dir=None,
            parents=None,
            db_file=DB_FILE)
        
        # NONSCF CALCULATION: ISO_1 ISO_2
        cdd_2d = ElectronicFW(name='ISO 1 NSCF: {}'.format(user_additions['unique_id']), 
            structure=struct_2d, 
            dedos=user_additions.get('dedos', 0.05),
            grid_density=user_additions.get('grid_density', 0.03),
            tags=tags_2d,
            dos=dos,
            bader=bader,
            cdd=False,
            parents=[cdd_combined, static_2d],
            prev_calc_dir=None,
            vasp_cmd=vasp_cmd,
            db_file=DB_FILE,
            electronic_set_overrides=electronic_set_overrides,
            **kwargs)
        cdd_sub = ElectronicFW(name='ISO 2 NSCF: {}'.format(user_additions['unique_id']),
            structure=struct_sub, 
            dedos=user_additions.get('dedos', 0.05),
            grid_density=user_additions.get('grid_density', 0.03),
            tags=tags_sub,
            dos=dos,
            bader=bader,
            cdd=False, 
            parents=[cdd_combined, static_sub],
            prev_calc_dir=None,
            vasp_cmd=vasp_cmd,
            db_file=DB_FILE,
            electronic_set_overrides=electronic_set_overrides,
            **kwargs)
        
        cdd_analysis = Firework(
            HeteroAnalysisToDb(db_file=DB_FILE,
                task_label="Charge Density Difference Analysis",
                dos=False,
                bader=False,
                cdd=cdd,
                additional_fields={}),
            name="Charge Density Difference Analysis", 
            spec={"_allow_fizzled_parents": False},
            parents=[cdd_2d, cdd_sub, cdd_combined])
        [fws.append(i) for i in [static_2d, static_sub, cdd_2d, cdd_sub, 
                                 cdd_combined, cdd_analysis]]

    # CREATE WORKFLOW
    wf = Workflow(fireworks=fws, name=wf_name)
    wf = add_namefile(wf)
    
    # remove VaspToDb for static calculations
    wf = change_Tasks(wf, mode='remove', fw_name_constraint=None,
            task_name_constraint='VaspToDb')
    
    print(wf.name)
    print(wf.links)
    return wf

