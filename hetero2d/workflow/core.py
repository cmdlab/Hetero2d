# coding: utf-8
# Copyright (c) CMD Lab Development Team.
# Distributed under the terms of the GNU License.

"""
This module sets up the vasp input files for the workflow to simulate 
2D materials adsorbed on a substrate slab.
"""
from __future__ import division, print_function, unicode_literals, absolute_import

from uuid import uuid4
from copy import deepcopy

from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer

from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.vasp.powerups import add_namefile
from atomate.utils.utils import get_logger

from fireworks import Firework
from fireworks.core.firework import Workflow
from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact

from hetero2d.firetasks.heteroiface_tasks import _update_spec
from hetero2d.manipulate.utils import center_slab
from hetero2d.io.VaspInterfaceSet import CMDLInterfaceSet
from hetero2d.fireworks.core import HeteroOptimizeFW, SubstrateSlabFW,  GenHeteroStructuresFW


__author__ = 'Tara M. Boland'
__copyright__ = "Copyright 2020, CMD Lab"
__maintainer__ = "Tara M. Boland"
__email__ = 'tboland1@asu.edu'

logger = get_logger(__name__)

def get_heterostructures_stabilityWF(struct_2d, struct_sub, struct_3d2d, heterotransformation_params, slab_params, user_additions, tags={}, bin_2d=VASP_CMD, bin_3d=VASP_CMD, dipole=None, uis=None, uis_2d=None, vis_2d=None, uis_3d2d=None, vis_3d2d=None, uis_bulk=None, vis_bulk=None, uis_trans=None, vis_trans=None, uis_iface=None, vis_iface=None):
    """
    Relax reference structures to determine energetic parameteres related
    to the formation of 2D films adsorbed onto substrates. Heterinterfaces 
    are created from the relaxed 2D and substrate slabs which generate a 
    symmetry matched, low lattice mismatch between the 2D material and 
    substrate slab. The user should be familiar with the behaviour of 
    heterointerface function before runnings simulations (load 
    hetero2d.manipulate.heterotransmuter import hetero_interfaces). Default 
    relaxation set CMDLInterfaceSet. Specify a complete vasp input 
    set or update the vis set using the uis_i tag to change vasp behaviour.
    It is also possible to use the general uis to updated all vis.
    
    Args:
        struct_2d (Structure): The primitive unit cell of the 2D structure
            to adsorb onto the substrate. Supply tags['2d']={ref_db_id:entry}
            if structure has never been relaxed. Additionally provided UID_opt2d
            if it has.
        struct_sub (Structure): The substrate to adsorb the 2D material.
        struct_3d2d (Structure): The bulk reference phase for the 2D 
            material.
        heterotransformation_params (list): A list transformation 
            parameters to create heterointerface configurations.
            Dictionary keys:
            * max_mismatch (int): Maximum percentage which the lattice will
            be allowed to be strained. Values range from 0-100.
            * max_area (int): The size of the supercell area you want to 
            search for potential matching lattices. Typical values 
            100-800.
            * max_angle_diff (int): The angle deviation in the c direciton. 
            Typical values around 1 percent.
            * r1r2_tol (int): The tolerance in mismatch between the new 
            lattice vectors. Typically 0.1 or 0.01.
            * separation (int): The separation distance between the 
            substrate and 2D material.
            * nlayers_sub (int): Set the selective dynamics tags for the top
            of the substrate. This sets how many layers of the surface 
            is allowed to relax. By default the bottom is frozen.
            * nlayers_2d (int): Set the selective dynamics tags for the 2D 
            material. This sets how many layers are allowed to relax. 
            NOTE: n_layers also is used to generate_all_configs.
        slab_params (dict): Same parameter format as the TransmuterFW to 
            create a slab.
            transmute (list): A list of transformations to be performed on
            each structure. See valid TransmuterFW inputs.
            transmute_params (list): A list of dictionaries defining the 
            input parameters to transmute the structure. 
            Example: {'transmute':['SlabTransformation'],'transmute_params':
            [{'miller_index':[0,0,1],'lll_reduce':True, 
            'max_normal_search':True,'min_vacuum_size':18,
            'primitive':False,'min_slab_size':12,'center_slab':True}]}
        user_additions (dict): A specification to control the workflow. 
            See firetasks.heteriface_tasks._update_spec for detailed list 
            of parameters. 
    
    Optional:
        dipole (bool): If True dipole corrections will be used for all slab 
            calculations. Defaults to True.
        tags (dict): A dictionary which list the tags for the whole, 2d, 
            3d2d, bulk, transmuter, and iface tags which you want to add to 
            each step in the calculation.
        uis (dict): A dictionary of general user incar settings you wish to 
            apply to every simulation. Defaults to None.
        uis_I (dict): A dictionary of INCAR settings to override the 
            VASPInputSet defaults or set additional commands for the Ith 
            structure where I = 2d, 3d2d, bulk, trans, and iface.
        vis_I (VASPInputSet): A VASP input set to relax the Ith materials 
            where I = 2d, 3d2d, bulk, trans, and iface. Defaults to 
            CMDLInterfaceSet.

    Returns:
        Heterostrcture workflow
    
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
    is_2d_optimized =   spec.get('is_2d_optimized')
    is_3d2d_optimized = spec.get('is_3d2d_optimized')
    is_sub_optimized =  spec.get('is_sub_optimized')
    
    # Vasp Input and wf input from spec
    wf_name = spec["wf_name"]
    if dipole == None:
        dipole = True
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
        name = "Bulk Structure Optimization: {}".format(
            unique_id)

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
        name = "Slab Structure Optimization-{}: {}".format(
            spec['orient'], unique_id)
        
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
        links_dict.update({'3d2d':{
            'fws': input_fws[len(input_fws)-1]}}) 
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
    [tags_iface.update(tags.get(i,{})) 
         for i in ['iface','general']]
  
    # make iface spec
    spec_iface = deepcopy(spec)
    spec_iface['tags'] = tags_iface
    spec['struct_2d'] = struct_2d

    # Create heterointerface structure firework
    #    FW name is assigned in the FW below
    input_fws.append(GenHeteroStructuresFW(spec=spec_iface,
                                           structure=struct_2d, heterotransformation_params=
        h_params, vasp_input_set = vis_iface,
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
    print('Workflow Name:',name)
    print(wf.links)

    return wf
