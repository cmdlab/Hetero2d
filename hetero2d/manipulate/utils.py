# coding: utf-8
# Copyright (c) CMD Lab Development Team.
# Distributed under the terms of the GNU License.

"""
Useful utilities to view, analyze, and change structures. Some functions are not intended for 
the user. 
"""

import gridfs
import gzip
import json
import math
import numpy as np
import os
import pymongo
import re
import sys
import zlib
from copy import deepcopy

from ase.calculators.vasp import VaspChargeDensity
from ase.visualize import view
from atomate.vasp.powerups import get_fws_and_tasks
from bson import ObjectId
from monty.serialization import loadfn, dumpfn
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure, Lattice, Element
from pymatgen.core.surface import Slab
from pymatgen.electronic_structure.dos import Dos, CompleteDos
from pymatgen.io.ase import AseAtomsAdaptor

from hetero2d.manipulate.layersolver import LayerSolver

__author__ = "Tara M. Boland, Arunima Singh"
__copyright__ = "Copyright 2020, CMD Lab"
__maintainer__ = "Tara M. Boland"
__email__ = "tboland1@asu.edu"
__date__ = "June 5, 2020"


############################
### Connection utilities ###
def get_mongo_client(db_file, db_type=None):
    """
    Connect to the database using mongoclient. Has multiple
    connection modes for URI (the url string) and ATLAS credentials.

    Args:
      db_file (str): The db.json file location.
      db_type (str): String describing the type of connection you are
        trying to make. Options - ATLAS or URI. For a private MongoDB
        server, ignore this tag. 
        
    Returns: 
      client, database_name
    """
    db = loadfn(db_file)
    host_name = db['host']
    database_name = db['database']
    user_name = db['admin_user']
    password = db['admin_password']

    if db_type == None:
        port = db.get('port', 27017)
        client = pymongo.MongoClient(host=host_name, port=port,
                                     authsource='admin', username=user_name, password=password)
    elif db_type == 'ATLAS':
        connection_url = "mongodb+srv://" + user_name + ":" + password + "@" + host_name + "/" \
                         + database_name + "?retryWrites=true&w=majority"
        client = pymongo.MongoClient(connection_url)
    elif db_type == 'URI':
        port = str(db.get('port', 27017))
        connection_url = "mongodb://" + user_name + ":" + password + "@" + host_name + ":" + port + "/" \
                         + database_name + "?authSource=admin" + "&retryWrites=true" + "&w=majority" \
                         + "&wtimeout=300000"
        client = pymongo.MongoClient(connection_url)
    return client, database_name


def get_dos(client, taskdoc):
    '''
    Uses the object_id from the task document to search for the dos in gridfs
    and loads the DOS object into a PMG DOS object. The key get_dos must be
    true in the task doc or the object does not exist in the gridfs.

    Args:
        client (MongoClient Object): The MongoClient connected to the database
            you are extracting information from to get the dos.
        taskdoc (dict): The task doc you want to get the DOS for.

    Returns:
        DOS obect
    '''
    # TASKDOC LOOKUP: DosBader._id=dos_fs.files._id=dos_fs.chunks.files_id
    fs_id = ObjectId(taskdoc['_id'])

    # connect to dos_fs
    fs = gridfs.GridFS(client, 'dos_fs')
    bs_json = zlib.decompress(fs.get(fs_id).read())
    obj_dict = json.loads(bs_json.decode())

    try:
        dos_obj = CompleteDos.from_dict(obj_dict)
    except:
        dos_obj = {Element(key): Dos.from_dict(value) for key, value in obj_dict.items()}

    return dos_obj


############################################
### Post/Pre processing helper functions ###
def vtotav(chgcar_file, axis='z'):
    '''
    A script which averages a CHGCAR or LOCPOT file in one axis to
    make a 1D curve. User must specify filename and axis on command
    line. Depends on ase (converted to 3.6 from ase's vtotav.py
    python 2.7).

    Args:
        chgcar (str): Path to the VASP CHGCAR file
        axis (str): Axis to project the charge density onto.
    
    Returns:
        { 'distance' (Ang.), 'chg_density' (projected e/A^3) }
    '''
    if not os.path.isfile(chgcar_file):
        print("\n** ERROR: Input file %s was not found." % chgcar_file)
        sys.exit()

    # Specify the axis to make average in. Default is Z.
    allowed = "xyzXYZ"
    if allowed.find(axis) == -1 or len(axis) != 1:
        axis = 'Z' if allowed.find(axis) == -1 or len(axis) != 1 else axis
    axis = axis.upper() if axis.islower() else axis

    # Open geometry and density class objects
    # -----------------------------------------
    vasp_charge = VaspChargeDensity(filename=chgcar_file)
    potl = vasp_charge.chg[-1]
    atoms = vasp_charge.atoms[-1]
    del vasp_charge

    # Read in lattice parameters and scale factor
    # ---------------------------------------------
    cell = atoms.cell

    # Find length of lattice vectors
    # --------------------------------
    latticelength = np.dot(cell, cell.T).diagonal()
    latticelength = latticelength ** 0.5

    # Read in potential data
    # ------------------------
    ngridpts = np.array(potl.shape)
    x_grid, y_grid, z_grid = ngridpts[0], ngridpts[1], ngridpts[2]
    totgridpts = ngridpts.prod()  # Total number of points
    sys.stdout.flush()

    # Perform average
    # -----------------
    if axis == "X":
        idir = 0
        a = 1
        b = 2
    elif axis == "Y":
        a = 0
        idir = 1
        b = 2
    else:
        a = 0
        b = 1
        idir = 2
    a = (idir + 1) % 3
    b = (idir + 2) % 3

    # At each point, sum over other two indices
    average = np.zeros(ngridpts[idir], np.float)
    for ipt in range(ngridpts[idir]):
        if axis == "X":
            average[ipt] = potl[ipt, :, :].sum()
        elif axis == "Y":
            average[ipt] = potl[:, ipt, :].sum()
        else:
            average[ipt] = potl[:, :, ipt].sum()

    # Scale chgcar by size of area element in the plane,
    # gives unit e/Ang. I.e. integrating the resulting
    # CHG_dir file should give the total charge.
    area = np.linalg.det([(cell[a, a], cell[a, b]),
                          (cell[b, a], cell[b, b])])
    dA = area / (ngridpts[a] * ngridpts[b])
    average *= dA

    # Get the average
    # -----------------
    sys.stdout.flush()
    proj_chg = {'distance': [], 'chg_density': []}
    xdiff = latticelength[idir] / float(ngridpts[idir] - 1)
    for i in range(ngridpts[idir]):
        x = i * xdiff
        proj_chg['distance'].append(round(x, 8))
        proj_chg['chg_density'].append(round(average[i], 8))
    return proj_chg


## Structure Analyzers ##
def average_z_sep(structure, iface_idx, initial=None):
    '''
    Get the average z separation distance between 2 
    layers in a structure. If provided an initial
    structure the change in the post-adsorbed z 
    separation distance is calculated. Typically used
    for 2D-substrate heterostructures in AnalysisToDb.

    Args:
        structure (Structure): the final structure to
            analyze.
        iface_idx (dict): Dictionary containing a list
            of atom indices for each layer. Computed 
            using tag_iface.
            
    Other Parameters:
        initial (Structure): the initial structure prior
            to DFT optimization.
            
    Returns:
        z-separation, delta_2d_z
    '''
    # get the 2d sub separation dist (bot & top atoms) 
    # define layers for top and bottom
    nlayers = iface_idx['num_2d_layer']
    z = ['sub_layer_1', '2d_layer_1', '2d_layer_' + str(nlayers)]

    # get the indices from iface that define the interface
    idx_subs, idx_tD_top, idx_tD = [iface_idx[i] for i in z]
    coords_f = structure.cart_coords

    # compute the initial 2d film width
    td_z_avg_f = np.average([coords_f[i][2] for i in idx_tD])
    # get average z coord of the substrate indices
    sub_z_avg_f = np.average([coords_f[i][2] for i in idx_subs])
    # compute the average z coord of the top layer of the 2d film
    td_top_z_avg_f = np.average([coords_f[i][2] for i in idx_tD_top])
    # difference between bottom layer of 2d from top layer of substrate
    zsep_f = abs(sub_z_avg_f - td_z_avg_f)
    # the thickness of the post-adsorbed 2d film
    td_diff_f = abs(td_z_avg_f - td_top_z_avg_f)

    # if given initial structure compute initial distances
    if initial:
        # initial structure
        coords_i = initial.cart_coords
        # z separation distance
        td_z_avg_i = np.average([coords_i[i][2] for i in idx_tD])
        td_top_z_avg_i = np.average([coords_i[i][2] for i in idx_tD_top])
        td_diff_i = abs(td_z_avg_i - td_top_z_avg_i)
        delta_2d_z = td_diff_f - td_diff_i

    # return value to users
    if not initial:
        return zsep_f
    else:
        return zsep_f, delta_2d_z


def get_fu(struct_sub, struct_2d, sub_aligned, td_aligned):
    '''
    Given superlattice structures and original unit cell structures
    return the number of formula units of the aligned 2d and substrate
    based on the original substrate and 2d component structures.
    
    Args:
        struct_sub (Structure): Unit cell for the substrate structure
            to be aligned to another structure.
        struct_2d (Structure): Unit cell for the 2d material to be 
            aligned to the substrate structure.
        aligned_sub (Structure): Superlattice for the substrate aligned
            to the 2d material.
        aligned_2d (Structure): Superlattice for the 2d material aligned
            to the substrate structure.

    Returns:
        fu_2d, fu_sub
    '''
    # ensure the structures are structure objects
    if not isinstance(struct_2d, Structure):
        struct_2d = Structure.from_dict(struct_2d)
    if not isinstance(struct_sub, Structure):
        struct_sub = Structure.from_dict(struct_sub)

    # number of atoms per element in the aligned lattices
    na_aligned_2d = dict(td_aligned.composition.get_el_amt_dict())
    na_aligned_sub = dict(sub_aligned.composition.get_el_amt_dict())

    # number of atoms per element in the supplied structures
    na_unit_2d = dict(struct_2d.composition.get_el_amt_dict())
    na_unit_sub = dict(struct_sub.composition.get_el_amt_dict())

    # final formula units for the dictionary
    fu_sub = {}
    fu_2d = {}

    # count the number of formula units of the 2d material
    # in the heteroiface struct
    for key in na_unit_2d.keys():
        fu_2d[key] = na_aligned_2d[key] / na_unit_2d[key]
    n_2d = fu_2d[key]

    # get the number of formula units of the substrate
    # slab in the heteroiface structure
    for key in na_unit_sub.keys():
        fu_sub[key] = na_aligned_sub[key] / na_unit_sub[key]
    n_sub = fu_sub[key]

    return n_2d, n_sub


def center_slab(structure):
    """
    Centers the atoms in a slab structure around 0.5
    fractional height.

    Args:
        structure (Structure): Structure to center.
        
    Returns:
        Centered Structure object
    """

    center = np.average([s._fcoords[2] for s in structure.sites])
    translation = (0, 0, 0.5 - center)
    structure.translate_sites(range(len(structure.sites)), translation)
    return structure


def tag_iface(structure, nlayers_2d, nlayers_sub=2):
    """
    Find the atom indices in a heterostructure by specifying how many 
    layers there are for the 2D material. Returns a dictionary of atom
    ids for each layer of the 2D material and nlayers_sub of the
    substrate surface.

    Args:
        structure (Structure): The hetero_interface structure.
            Should be an unrelaxed structure.
        nlayers_2d (int): The number of layers contained
            within the 2d material.
        nalyers_sub (int): The number of layers of the substrate
            surface to include in the interface tags.
            
    Returns:
        layer_indices
    """
    # set structure as slab
    interface = slab_from_struct(structure)
    # get z coords of the sites
    z_coords = interface.frac_coords[:, 2]
    # round z coords to avoid precision errors from np
    z_coords_round = [round_decimals_down(i, decimals=3) for i in z_coords]
    # get the z-cords of atoms at interface
    unique_layer_coords = np.unique(z_coords_round)

    # get all unique coords for each layer
    layer_coords_dict = {}
    # get 2d layer indices
    for idx in list(range(1, nlayers_2d + 1)):
        layer_coords = unique_layer_coords[-idx]
        layer_name = '2d_layer_' + str(abs(idx))
        layer_coords_dict[layer_name] = layer_coords
    # get sub layer indices
    for count, idx in enumerate(range(nlayers_2d + 1, nlayers_2d + nlayers_sub + 1)):
        layer_coords = unique_layer_coords[-idx]
        layer_name = 'sub_layer_' + str(count + 1)
        layer_coords_dict[layer_name] = layer_coords

        # get atom ids for each layer
    layer_indices = {key: [] for key in layer_coords_dict.keys()}
    for idx, site in enumerate(structure.sites):
        site_coords = round_decimals_down(site.frac_coords[2], decimals=3)
        if site_coords in layer_coords_dict.values():
            key = get_key(layer_coords_dict, site_coords)
            layer_indices[key].append(idx)

    layer_indices['num_2d_layer'] = nlayers_2d
    layer_indices['num_sub_layer'] = nlayers_sub

    return layer_indices


def iface_layer_locator(structure, cutoff, iface_elements):
    """
    Helper function used to locate the iface layers from
    the layer solver.

    Args:
        structure (Structure): input structure to find the
            interface layers for.
        cutoff (float): the layer separation cutoff distance.
        iface_elements (list): A list of element species
            that compose the top and bottom iface layers.
            
    Returns:
        LayerSolver object, top_layer specie, bottom_layer specie
    """
    sample = LayerSolver(structure, cutoff=cutoff)
    flag = True

    # get the 2d sub separation dist (bot & top atoms) 
    for i in range(0, len(sample) - 2):
        # define layer data
        l1 = sample['Layer' + str(i + 1)]
        l2 = sample['Layer' + str(i)]

        # sort layers by species
        test_layer = sorted([l1['species'][0],
                             l2['species'][0]])

        # make sure test_layer matches the iface composition
        if test_layer == sorted(iface_elements):
            return sample, l1, l2

    if flag:
        print('\t\t## Iface composition not found. Maybe the surface has undergone significant relaxation.')
        return None, None, None


def atomic_distance(structure, dist=2):
    """
    Given a structure and an interatomic separation distance all sites which have a
    spacing less than dist will be printed out.

    Args:
        structure (Structure): The structure to analyze.
        dist (float): A cut-off value which defines the minimum inter_atomic spacing.
    """
    for idx, i in enumerate(structure.sites):
        for idx1, j in enumerate(structure.sites):
            if structure.distance_matrix[idx][idx1] < dist and idx != idx1:
                print(idx, idx1, i, j)


## POSCAR Modifier ##
def set_sd_flags(interface=None, n_layers=2, top=True, bottom=True, lattice_dir=2):
    """
    Set the relaxation flags for top and bottom layers of interface.

    The upper and lower bounds of the z coordinate are determined
    based on the slab. All layers above and below the bounds will
    be relaxed. This means if there is a ligand on top of the slab,
    all of its atoms will also be relaxed.

    Args:
        interface (Structure): input structure file.
        n_layers (int): number of layers to be relaxed.
        top (bool): whether n_layers from top are be relaxed.
        bottom (bool): whether n_layers from bottom are be relaxed.
        lattice_dir (str): whether to search the a, b, or c axis for
            layers.
            
    Returns:
        sd_flags
    """
    sd_flags = np.zeros_like(interface.frac_coords)
    slab = slab_from_struct(interface)

    z_cords = interface.frac_coords[:, lattice_dir]
    z_cords_slab = slab.frac_coords[:, lattice_dir]
    z_lower_bound, z_upper_bound = None, None
    if bottom:
        z_cords_round = [round_decimals_down(i, decimals=4) for i in z_cords_slab]
        z_lower_bound = np.unique(z_cords_round)[n_layers - 1]
        sd_flags[np.where(z_cords <= z_lower_bound)] = np.ones((1, 3))
    if top:
        z_cords_round = [round_decimals_down(i, decimals=4) for i in z_cords_slab]
        z_upper_bound = np.unique(z_cords_round)[-n_layers]
        sd_flags[np.where(z_cords >= z_upper_bound)] = np.ones((1, 3))
        print('sd_flags', sd_flags)
    return sd_flags.tolist()


## Workflow Modifier ##
def change_Tasks(original_wf, mode, fw_name_constraint=None,
                 task_name_constraint='VaspToDb', change_method=None):
    '''
    Change the default behavior for task_name_constraint for all fw_name_constraint
    by changing, removing, or updating the input of the method. Changes must be
    provided using change_method. This setting will completely overwrite the current
    method.
    
    Args:
        original_wf (Workflow): Original workflow to change.
        mode (str): Mode determines how the VaspToDb method is changed for the
            workflow. Options are: replace (with new vasptodb method), update 
            (the current settings for vasptodb method), and remove (vasptodb
            method from workflow).
        fw_name_constraint (str): Only apply changes to FWs where fw.name
            contains this substring. Default set to None.
        task_name_constraint (str): Name of the Firetasks to be tagged, 
            Default is 'VaspToDb'.
        change_method (dict): If update mode, provide a dictionary of 
            valid arguements for the VaspToDb method in key: value format. 
    '''
    idx_list = get_fws_and_tasks(original_wf, fw_name_constraint=fw_name_constraint,
                                 task_name_constraint=task_name_constraint)

    if mode == 'change':
        for idx_fw, idx_t in idx_list:
            original_wf.fws[idx_fw].tasks[idx_t].update(change_method)

            # update all vasptodb methods given the contrainsts provided
    if mode == 'update':
        for idx_fw, idx_t in idx_list:
            original_wf.fws[idx_fw].tasks[idx_t].update(change_method)

    # remove all matching vasptodb methods given the contrainsts provided
    if mode == 'remove':
        for idx_fw, idx_t in idx_list:
            original_wf.fws[idx_fw].tasks.pop(idx_t)
    return original_wf


## Helper Function ##
def slurm_set_npar():
    '''
    Set npar for slurm computing systems. NSLOTS is not valid.
    
    Returns sqrt(ncores) or npar = 2
    '''
    # set npar looking for ntasks from os.environ: nslots and slurm_cpus_per_task
    # fails on all slurm systems
    slurm_keys = list(os.environ.keys())
    ncore_keys = [[key for key in slurm_keys if re.search(match, key)]
                  for match in ['TASKS', 'NSLOTS', 'CPU', 'NODE']]
    ncore_keys = list(np.unique(sum(ncore_keys, [])))

    slurm_dict = {key: os.environ[key] for key in ncore_keys}
    dumpfn(slurm_dict, 'slurm_keys.json')

    for key in ncore_keys:
        if re.search('SLURM_NTASKS', key):
            ncores = int(os.environ[key])
            break
        # elif re.search('SLURM_JOB_CPUS_ON_NODE',key):
        #    
        #    ncores = int(os.environ[key])
        #    break
        elif re.search('NSLOTS', key):
            ncores = int(os.environ[key])
            break
    npars = []
    npar_found = False
    if ncores and ncores != 1:
        for npar in range(int(math.sqrt(ncores)), ncores):
            if ncores % npar == 0:
                npars.append(npar)
                npar_found = True
    if npar_found:
        # the first npar found is the closest to sqrt of ncores
        return npars[0]
    else:
        # returning npar = 2 is a safe default
        return 2


def decompress(infile, tofile):
    '''
    Method to decompress a gzip file. Specify the input file and the output.
    Leaves the original gzipped file compressed.

    Args:
        infile (str): Absolute or relative file path of file to decompress
            using gzip.
        tofile (str): Absolute or relative file path of the decompressed file.
    '''
    with open(infile, 'rb') as inf, open(tofile, 'w', encoding='utf8') as tof:
        decom_str = gzip.decompress(inf.read()).decode('utf-8')
        tof.write(decom_str)


def get_FWjson():
    """
    Helper function which reads the FW.json file in the
    local directory and returns the FW.json file.
    """
    try:
        with gzip.GzipFile('FW.json.gz', 'r') as p:
            fw_json_file = json.loads(p.read().decode('utf-8'))
    except:
        with open('FW.json', 'rb') as p:
            fw_json_file = json.load(p)
    return fw_json_file


def round_decimals_down(number: float, decimals: int = 2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor


def get_key(my_dict, val):
    ''' 
    Function returns the key corresponding to a dictionary value.
    '''
    for key, value in my_dict.items():
        if val == value:
            return key
    return "key doesn't exist"


############################
### Conversion Functions ###
def slab_from_struct(structure, hkl=None):
    """                                                                         
    Reads a pymatgen structure object and returns a Slab object. Useful for reading 
    in 2d/substrate structures from atomate wf's.              
    
    Args:                                                                       
      hkl (list): miller index of the slab in the input file.                       
      structure (Structure): structure file in any format supported by pymatgen.
      
    Returns:                                                                    
      Slab structure object                                                            
    """
    if hkl is None:
        hkl = [0, 0, 1]
    if not isinstance(structure, Structure):
        slab_input = Structure(structure)
    else:
        slab_input = structure

    return Slab(slab_input.lattice,
                slab_input.species_and_occu,
                slab_input.frac_coords,
                hkl,
                Structure.from_sites(slab_input, to_unit_cell=True),
                shift=0,
                scale_factor=np.eye(3, dtype=np.int),
                site_properties=slab_input.site_properties)


def struct_from_str(string):
    """
    Given a string serialized pymatgen structure object, return a structure object.
    Fixes a conversion error when j_sanitizing structure objects not in dictionary
    format. Probably due to my ignorance.

    Args:
      string (str): A string representation of a structure object.

    Returns:
      pymatgen structure object
    """
    string = string.split('\n')
    # specie and coords
    specie = []
    coords = []

    sites = int(string[4].split('(')[1].split(')')[0])
    for i in range(7, 7 + sites):
        tmp = []
        [tmp.append(i) for i in string[i].split(' ') if i != '']
        specie.append(tmp[1])
        coords.append([float(tmp[2]), float(tmp[3]), float(tmp[4])])

    # lattice
    ang = []
    [ang.append(float(i)) for i in string[3].split(':')[1].split(' ') if i != '']

    abc = []
    [abc.append(float(i)) for i in string[2].split(':')[1].split(' ') if i != '']

    lattice = Lattice.from_lengths_and_angles(abc, ang)

    return Structure(lattice, specie, coords)


def show_struct_ase(structure):
    """
    Creates a pop up structure model for a pymatgen structure object using ase's
    viewer. Created for use in jupyter notebooks and might not work over remote
    access.

    Args:
      structure (Structure): Structure to show.
    """
    to_ase = AseAtomsAdaptor()
    structure = to_ase.get_atoms(structure)
    viewer = view(structure)
    return viewer


#########################################
### local structure analysis function ###
# TODO: it would be nice to return bond distance statistics to the user @tboland1
class nn_site_indices(CrystalNN):
    """
    This function returns the nearest neighbor atom id's for a
    given atom id. This function searches from [start,end]
    using the NN variable.

    Args:
        structure (Structure): pymatgen structure object to perform
            analysis on.
        NN (list): A list indicating the radius from [start,end] to
            search for valid anions/cations. This is used to set
            the search_cutoff which will be given by the end
            value.
        target_sites ([int]): A list of atom ids which you want
            to search around for NN cations/anions. Atom id's can be
            visualized using ASE.
        cation_anion (bool): Whether to match anion and cation pairs.
            Defaults to False.
        oxi_states (dict): If anion_cation true, provide a dictionary
            of the oxidation states for each cation/anion species.
        image (bool): Whether to return data from image cells. Defaults
            to True.
        duplicated (bool): Remove duplicated sites and return only the
            unique site_indices for the structure. Defaults to True.
            
    Returns:
        site_data (dict): A dictionary containing the atom id
        which was searched around and the corresponding list
        of NN atom ids of the cation/anions.
    """

    def __init__(self, structure, NN, target_sites, cation_anion=False, oxi_states=None, image=True, duplicate=True,
                 **kwargs):
        super().__init__(cation_anion)
        if oxi_states is None:
            oxi_states = {}
        self.search_cutoff = NN[1] or 7
        self.distance_cutoffs = (NN) or (0.5, 3.0)
        self.cation_anion = cation_anion
        self.oxi_states = oxi_states
        self.struct = deepcopy(structure)
        self.target_sites = target_sites
        self.image = image
        self.duplicate = duplicate

        # **kwargs
        self.fingerprint_length = kwargs.get('fingerprint_length', None)
        self.porous_adjustment = kwargs.get('porous_adjustment', True)
        self.x_diff_weight = kwargs.get('x_diff_weight', 3.0)
        self.weighted_cn = kwargs.get('weighted_cn', False)

        # add oxidation states if provided
        if self.cation_anion:
            self.struct.add_oxidation_state_by_element(oxi_states)
        self.cNN = CrystalNN(self.weighted_cn, self.cation_anion,
                             self.distance_cutoffs, self.x_diff_weight,
                             self.porous_adjustment, self.search_cutoff,
                             self.fingerprint_length)

    def avg_site_bond_dist(self, bonded_elm=None, return_avg_list=True):
        """
        Give the site-dependent bond distance from the target site.
        If return_avg_list is True, an average bond distance is provided
        for each target sites nearest neighbor list entry returned
        by the all_nn_info method. Bonding distance dictionary returned
        with the format {nn_target_site_index: {nn_bonded_atom_index:
        bond_distance_value}. Bond distance is computed from the
        all_nn_info() method.

        Args:
            bonded_elm (list): List of elements that you want
                to determine bond lengths for. If empty finds
                all nearby elements. If empty, all elements are
                considered for analysis.
            return_avg_list (bool): Whether to return the average
                bond distance.

        Returns:
            dictionary of NN bond distances for each target specie.
            If return_avg_list=True, returns the avg bond distance
            of each target specie.
        """
        # set up return data array
        if bonded_elm is None:
            bonded_elm = []
        bond_dist = {}

        # get all_nn_info from CNN
        nn_data = self.all_nn_info()

        # loop over nn_data keys(target_site atom indices)
        for target in nn_data.keys():
            # create bond distance dict entry for each target
            bond_dist[target] = {}

            # analyze each bonded site for each target site
            for bonded_site in nn_data[target]:
                if len(bonded_elm) == 0 or bonded_site['site'].species_string in bonded_elm:
                    # get cords to determine bond dist
                    nn_site = bonded_site['site'].coords
                    t_site = self.struct[target].coords

                    # compute distance between 2 points
                    x2 = (t_site[0] - nn_site[0]) ** 2
                    y2 = (t_site[1] - nn_site[1]) ** 2
                    z2 = (t_site[2] - nn_site[2]) ** 2
                    dist = np.sqrt(x2 + y2 + z2)

                    # append bond distance for each target atom
                    bond_dist[target][bonded_site['site_index']] = dist

            # return the average bond distance for the target site
            if return_avg_list:
                tot = [value for key, value in bond_dist[target].items()]
                bond_dist[target]['avg_bond_dist'] = np.sum(tot) / len(tot)
        return bond_dist

    def image_filter(self, data):
        """
        Filter out all sites that are not bonded within the (0,0,0) unit
        cell. Triggered by image = False.

        Args:
            data (dict): the cNN.all_nninfo data type.
        """

        unique_nn = {}

        # loop over each target sites nn_data info
        for key, value in data.items():
            unit_cell = []
            for nn_site in value:
                # only return (0,0,0) nn
                if nn_site['image'] == (0, 0, 0):
                    unit_cell.append(nn_site)
            unique_nn[key] = unit_cell

        return unique_nn

    def remove_duplicates(self, data):
        """
        Return only the unique site indices from the all_nninfo
        cNN returned for each site. Truggered by duplicates=True.

        Args:
            data (dict): the cNN.all_nninfo data type.
        """

        unique_nn = {}

        # loop over each target sites nn_data info
        for key, value in data.items():
            unique_periodic_sites = []
            unique_sites = []
            for nn_site in value:
                # unique site indices
                if nn_site['site_index'] not in unique_sites:
                    unique_sites.append(nn_site['site_index'])
                    unique_periodic_sites.append(nn_site)
            unique_nn[key] = unique_periodic_sites

        return unique_nn

    def all_nn_info(self):
        """
        Return the get_nn_data attribute from CrystalNN with or
        without image data, with options to remove duplicated
        site_indices. Example: [{'site':'PeriodicSite',
        'image':(int,int,int), 'weight':float, 'site_index':int}, ..]

        Returns:
            nn_data (dict)
        """

        nn_data = {}  # the NN data dict

        # itt over all target sites
        for site in self.target_sites:
            # initialize nn's around each target site
            try:
                nn = self.cNN.get_nn_data(structure=self.struct,
                                          n=site)
                nn_data[site] = nn.all_nninfo  # target data array
            except ValueError:
                print('The NN distance is not long enough to generate bond',
                      'information for each site. This entry contains None values.')
                nn_data[site] = [{'site': None, 'image': None, 'weight': None,
                                  'site_index': None}]
        # return image data if wanted
        nn_data = self.image_filter(nn_data) if not self.image else nn_data  # remove images
        nn_data = self.remove_duplicates(nn_data) if self.duplicate else nn_data  # remove duplicate index sites

        self.nn_data = nn_data
        return nn_data

    def site_indices(self):
        """
        Returns the nearest neighbor site indices subject to the NN search criteria.
        """

        nn_idx = {}  # nn site indices around target site

        for site in self.target_sites:
            nn = self.nn_data[site]

            nn_site_idx = []  # array to store site idx
            # nn site indices for the target site
            for value in nn:
                site_idx = value['site_index']
                nn_site_idx.append(site_idx)
            nn_idx[site] = nn_site_idx  # target data array

        return nn_idx

    def site_elements(self):
        """
        Returns the nearest neighbor elements that correspond to the site indices.
        """

        nn_elms = {}  # nn elements to return

        for t_site in self.target_sites:
            nn = self.nn_data[t_site]

            nn_site_elm = {}  # nn elements to target site
            # get nn elements for target site
            for value in nn:
                sp_str = value['site'].species_string
                nn_site_elm.update({value['site_index']: sp_str})
            nn_elms[t_site] = nn_site_elm

        return nn_elms


def reduced_supercell_vectors(ab, n):
    """
    returns all possible reduced in-plane lattice vectors and
    transition matrices for the given starting unit cell lattice
    vectors(ab) and the supercell size n
    """
    uv_list = []
    tm_list = []
    for r_tm in get_trans_matrices(n):
        for tm in r_tm:
            uv = get_uv(ab, tm)
            uv, tm1 = get_reduced_uv(uv, tm)
            uv_list.append(uv)
            tm_list.append(tm1)
    return uv_list, tm_list


def get_trans_matrices(n):
    """
    yields a list of 2x2 transformation matrices for the
    given supercell
    n: size
    """

    def factors(n0):
        for i in range(1, n0 + 1):
            if n0 % i == 0:
                yield i

    for i in factors(n):
        m = n // i
        yield [[[i, j], [0, m]] for j in range(m)]


def get_uv(ab, t_mat):
    """
    return u and v, the supercell lattice vectors obtained through the
    transformation matrix
    """
    u = np.array(ab[0]) * t_mat[0][0] + np.array(ab[1]) * t_mat[0][1]
    v = np.array(ab[1]) * t_mat[1][1]
    return [u, v]


def get_reduced_uv(uv, tm):
    """
    returns reduced lattice vectors
    """
    is_not_reduced = True
    u = np.array(uv[0])
    v = np.array(uv[1])
    tm1 = np.array(tm)
    u1 = u.copy()
    v1 = v.copy()
    while is_not_reduced:
        if np.dot(u, v) < 0:
            v = -v
            tm1[1] = -tm1[1]
        if np.linalg.norm(u) > np.linalg.norm(v):
            u1 = v.copy()
            v1 = u.copy()
            tm1c = tm1.copy()
            tm1[0], tm1[1] = tm1c[1], tm1c[0]
        elif np.linalg.norm(v) > np.linalg.norm(u + v):
            v1 = v + u
            tm1[1] = tm1[1] + tm1[0]
        elif np.linalg.norm(v) > np.linalg.norm(u - v):
            v1 = v - u
            tm1[1] = tm1[1] - tm1[0]
        else:
            is_not_reduced = False
        u = u1.copy()
        v = v1.copy()
    return [u, v], tm1


def get_r_list(area1, area2, max_area, tol=0.02):
    """
    returns a list of r1 and r2 values that satisfies:
    r1/r2 = area2/area1 with the constraints:
    r1 <= Area_max/area1 and r2 <= Area_max/area2
    r1 and r2 corresponds to the supercell sizes of the 2 interfaces
    that align them
    """
    r_list = []
    rmax1 = int(max_area / area1)
    rmax2 = int(max_area / area2)
    print('rmax1, rmax2: {0}, {1}\n'.format(rmax1, rmax2))
    for r1 in range(1, rmax1 + 1):
        for r2 in range(1, rmax2 + 1):
            if abs(float(r1) * area1 - float(r2) * area2) / max_area <= tol:
                r_list.append([r1, r2])
    return r_list


def get_mismatch(a, b):
    """
    percentage mistmatch between the lattice vectors a and b
    """
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(b) / np.linalg.norm(a) - 1


def get_angle(a, b):
    """
    angle between lattice vectors a and b in degrees
    """
    a = np.array(a)
    b = np.array(b)
    return np.arccos(
        np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)) * 180 / np.pi


def get_area(uv):
    """
    area of the parallelogram
    """
    a = uv[0]
    b = uv[1]
    return np.linalg.norm(np.cross(a, b))
