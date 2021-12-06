# coding: utf-8
# Copyright (c) CMD Lab Development Team.
# Distributed under the terms of the GNU License.

"""
These modules are used to create the heterostructure configurations given a 2d and substrate slab.
The code is adopted from the MPInterfaces to ensure the code is compatible with the FireWorks and 
atomate architecture, to fix minor bugs in the original code, and return interface matching criteria.
"""

from six.moves import range
from copy import deepcopy
import numpy as np, sys

from mpinterfaces.transformations import reduced_supercell_vectors, get_r_list, get_angle, \
    get_mismatch, get_area, get_uniq_layercoords

from pymatgen import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import Slab

from hetero2d.manipulate.utils import slab_from_struct, show_struct_ase, get_fu
from hetero2d.manipulate.layersolver import LayerSolver

__author__ = "Tara M. Boland, Arunima Singh"
__copyright__ = "Copyright 2020, CMD Lab"
__maintainer__ = "Tara M. Boland"
__email__ = "tboland1@asu.edu"
__date__ = "June 5, 2020"


def rotate_to_acute(structure):
    """
    If the angle for a 2D structure is obtuse, reflect the b vector to -b to make it acute.
    
    Args:
        structure (Structure): structure to rotate.
    
    Returns:
        rotated structure
    """
    # the new coordinate system basis vectors
    a_prime = np.array(
        [
            structure.lattice.matrix[0, :],
            -structure.lattice.matrix[1, :],
            structure.lattice.matrix[2, :]
        ])

    # the old coordinate system basis vectors
    a = deepcopy(structure.lattice.matrix.T)

    # find transformation matrix from basis vectors
    a_inv = np.linalg.inv(a)
    p = np.dot(a_inv, a_prime.T)
    p_inv = np.linalg.inv(p)
    # apply transformation matrix to coords
    frac_coords = []
    species = [site.specie for site in structure.sites]
    for site in structure.sites:
        coord = site.frac_coords
        trans = np.dot(p_inv, coord)
        frac_coords.append(trans)

    # create new structure
    new_structure = Structure(lattice=a_prime, species=species,
                              coords=frac_coords, coords_are_cartesian=False,
                              to_unit_cell=True)

    return new_structure


def aligned_hetero_structures(struct_2d, struct_sub, max_mismatch=0.01, max_area=200,
                              nlayers_2d=3, nlayers_sub=2, r1r2_tol=0.02, max_angle_diff=1, 
                              separation=3.4):
    """
    Given the 2 slab structures and the alignment parameters, return
    slab structures with lattices that are aligned with respect to each
    other.

    Args:
        slab_2d (Structure): Two dimensional slab structure object.
        slab_sub (Structure): Substrate slab structure object.
        max_area (float): The maximum area you want to search for
            a matching lattice. Defaults to 200 sqr. Angstroms.
        max_mismatch (float):  The maximum mismatch between the a 
            and b lattice vectors of the aligned 2D and substrate 
            lattice vectors. Defaults to 0.01, multiply by 100 to 
            obtain percent.
        max_angle_diff (float): The angle deviation between the a and
            b lattice vectors between the old lattice vectors and the
            new lattice vectors. Defaults to 1 degree.
        r1r2_tol (float): Allowed area approximation tolerance for the two
            lattices. Defaults to 0.02.
        nlayers_substrate (int): number of substrate layers. Defaults
            to 2 layers.
        nlayers_2d (int): number of 2d material layers. Defaults to
            3 layers.
        separation (float): separation between the substrate and the 2d
            material. Defaults to 3.4 angstroms.

    Returns:
        aligned_sub, aligned_2d, alignment_info  
    """
    # deep copy original input structures to match lats
    struct_2d = deepcopy(struct_2d)
    struct_sub = deepcopy(struct_sub)

    # remove any site properties from input structs
    [struct_2d.remove_site_property(i)
     for i in struct_2d.site_properties.keys()]
    [struct_sub.remove_site_property(i)
     for i in struct_sub.site_properties.keys()]

    # ensure both structures are slabs
    if not isinstance(struct_2d, Slab):
        struct_2d = slab_from_struct(structure=struct_2d)
    if not isinstance(struct_sub, Slab):
        struct_sub = slab_from_struct(structure=struct_sub)

    # copy the substrate and 2d material
    iface1 = deepcopy(struct_sub)
    iface2 = deepcopy(struct_2d)

    # get the surface (ab) area for the sub and 2d mat
    area1 = iface1.surface_area
    area2 = iface2.surface_area

    # a, b vectors that define the surface
    ab1 = [iface1.lattice.matrix[0, :], iface1.lattice.matrix[1, :]]
    ab2 = [iface2.lattice.matrix[0, :], iface2.lattice.matrix[1, :]]

    # get the list of vectors r1 and r2 that span the new lattice
    r_list = get_r_list(area1, area2, max_area, tol=r1r2_tol)
    if not r_list:
        print('r_list is empty.\n')
        sys.exit()
        
    found = []
    uv1_list, tm1_list, uv2_list, tm2_list = [], [], [], []

    # 4 each lattice vector in r_list reduce the super_lattice vectors
    # to find new unit cells
    for r1r2 in r_list:
        x1, y1 = reduced_supercell_vectors(ab1, r1r2[0])
        uv1_list.append(x1)
        tm1_list.append(y1)
        x2, y2 = reduced_supercell_vectors(ab2, r1r2[1])
        uv2_list.append(x2)
        tm2_list.append(y2)
        if not uv1_list and not uv2_list:
            continue

    # remove duplicate r1 r2 vectors
    new_uv1, new_tm1 = remove_duplicates(uv1_list, tm1_list)
    new_uv2, new_tm2 = remove_duplicates(uv2_list, tm2_list)

    # old trans lattice
    idx = range(len(new_uv2))
    for idx, sup_lat_n1, sup_lat_n2 in zip(idx, new_uv1, new_uv2):
        for i, uv1 in enumerate(sup_lat_n1):  # new_uv1):
            for j, uv2 in enumerate(sup_lat_n2):  # new_uv2):
                u_mismatch = get_mismatch(uv1[0], uv2[0])
                v_mismatch = get_mismatch(uv1[1], uv2[1])
                angle1 = get_angle(uv1[0], uv1[1])
                angle2 = get_angle(uv2[0], uv2[1])
                angle_mismatch = abs(angle1 - angle2)
                area1 = get_area(uv1)
                area2 = get_area(uv2)
                if abs(u_mismatch) < max_mismatch and abs(
                        v_mismatch) < max_mismatch:
                    max_angle = max(angle1, angle2)
                    min_angle = min(angle1, angle2)
                    mod_angle = max_angle % min_angle
                    is_angle_factor = False
                    if abs(mod_angle) < 0.001 or abs(
                            mod_angle - min_angle) < 0.001:
                        is_angle_factor = True
                    if angle_mismatch < max_angle_diff or is_angle_factor:
                        # double check if this i cell reduction params the cell reduction does 
                        # not take into account atomic positions just cell vectors
                        #if round(angle_mismatch,2) == round(struct_2d.lattice.gamma,2):
                        #    found.append((uv1, uv2, min(area1, area2), 
                        #                  u_mismatch,v_mismatch,angle_mismatch, 
                        #                  tm1_list[idx][i],tm2_list[idx][j] ))
                        if angle_mismatch > max_angle_diff:
                            continue
                        elif angle_mismatch < max_angle_diff:
                            found.append((uv1, uv2, min(area1, area2), 
                                          u_mismatch,v_mismatch,angle_mismatch, 
                                          tm1_list[idx][i],tm2_list[idx][j] ))

    # sort each super_lattice found by surface area
    if found:
        print('\nMATCH FOUND\n')
        uv_opts = sorted(found, key=lambda x: x[2])
        uv_opt = uv_opts[0]
        alignment_info = {'opt_uv1': uv_opt[0], 'opt_uv2': uv_opt[1], 
                          'u': uv_opt[3],'v': uv_opt[4], 'angle': uv_opt[5], 
                          'tm1': uv_opt[6], 'tm2': uv_opt[7]}
        print('u,v & angle mismatches:\n{0}, {1}, {2}\n'.format(
              uv_opt[3], uv_opt[4], uv_opt[5]))
        uv_substrate, uv_mat2d = uv_opt[0], uv_opt[1]
    else:
        print('\n NO MATCH FOUND\n')
        return None, None, None
        sys.exit()
    
    # SUBSTRATE: map the initial slabs to the newly found matching lattices
    substrate = Structure.from_sites(struct_sub)
    mat2d = Structure.from_sites(struct_2d)

    #### Substrate Lattice Assignment ###
    # map the intial slabs to the newly found matching lattices
    substrate_latt = Lattice(np.array(
        [
            uv_substrate[0][:],
            uv_substrate[1][:],
            substrate.lattice.matrix[2, :]
        ]))

    # try to get a non-singular matrix within 10 attempts
    for res in substrate.lattice.find_all_mappings(substrate_latt, 
                                                   ltol=0.05,
                                                   atol=max_angle_diff):
        scell_sub = res[2]
        scell_sub[2] = np.array([0, 0, 1])
        if np.linalg.det(scell_sub) < 1e-5:
            continue
        else:
            break
    substrate.make_supercell(scell_sub)
    
    ### Mat2d Lattice Assignment ## 
    # map the intial slabs to the newly found matching 
    # lattices to avoid numerical issues with find_mapping
    mat2d_fake_c = mat2d.lattice.matrix[2,:]/np.linalg.norm(
        mat2d.lattice.matrix[2,:])*5.0
    mat2d_latt = Lattice(np.array(
        [
            uv_mat2d[0][:],
            uv_mat2d[1][:],
            mat2d_fake_c
        ]))
    mat2d_latt_fake = Lattice(np.array(
        [
            mat2d.lattice.matrix[0, :],
            mat2d.lattice.matrix[1, :],
            mat2d_fake_c
        ]))

    # try to get a non-singular matrix within 10 attempts
    for res in mat2d_latt_fake.find_all_mappings(mat2d_latt, 
                                                 ltol=0.05,
                                                 atol=max_angle_diff):
        # res = aligned_lat, rot, scell
        scell_2d = res[2]
        scell_2d[2] = np.array([0, 0, 1])
        if np.linalg.det(scell_2d) < 1e-5:
            continue
        else:
            break
    try:
        mat2d = deepcopy(mat2d)
        mat2d.make_supercell(scell_2d)
    except:
        print('Failed making mat2d superlattice')

    det_2d = np.linalg.det(res[0].matrix)
    if det_2d > 0:
        lmap = Lattice(np.array(
            [
                substrate.lattice.matrix[0, :],
                substrate.lattice.matrix[1, :],
                mat2d.lattice.matrix[2, :]
            ]))
    else:
        lmap = Lattice(np.array(
            [
                -substrate.lattice.matrix[0, :],
                -substrate.lattice.matrix[1, :],
                mat2d.lattice.matrix[2, :]
            ]))
    mat2d.modify_lattice(lmap)
    
    ## ensure that the symmetry of the 2d and aligned 2d agree
    sg_mat2d = SpacegroupAnalyzer(mat2d)
    sg_align = SpacegroupAnalyzer(struct_2d)
    
    m2d = {}
    align = {}
    [m2d.update({key:sg_mat2d.get_symmetry_dataset()[key]})
        for key in ['number','hall_number','international','hall','pointgroup']]
    [align.update({key:sg_align.get_symmetry_dataset()[key]})
        for key in ['number','hall_number','international','hall','pointgroup']]

    # compare mat2d and aligned lattice
    is_equal = m2d == align

    if is_equal == True:
        return substrate, mat2d, alignment_info
    else:
        print('SpacegroupAnalyzer failed.\n')
        return None, None, None


def generate_all_configs(mat2d, substrate, nlayers_2d=3, nlayers_substrate=2, separation=3.4):
    """
    For the given lattice matched 2D material and substrate structures, this functions computes all
    unique (Wyckoff) sites of the mat2d and substrate. The unique sites are iteratively matched 
    between the mat2d and substrate stacking the unique sites on top of each other separated by the
    separation distance parameter. This subsequently generates all possible 2d/substrate heterostructure
    configurations stacked over high symmetry points. All unique structures are returned.

    Args:
        mat2d (Structure): Lattice and symmetry-matched 2D material 
            structure.
        substrate (Structure): Lattice and symmetry-matched 2D 
            substrate structure.
        nlayers_substrate (int): number of substrate layers. Defaults 
            to 2 layers.
        nlayers_2d (int): number of 2d material layers. Defaults to 
            3 layers.
        separation (float): separation between the substrate and the 2d
            material. Defaults to 3.4 angstroms.
    
    Returns:
        list of hetero_interface configurations 
    """
    if not (mat2d and substrate):
        sys.exit()

    # unique site coordinates in the substrate top layers
    coords_uniq_sub = get_uniq_layercoords(substrate, nlayers_substrate, top=True)

    # unique site coordinates in the 2D material bottom layers
    coords_uniq_2d = get_uniq_layercoords(mat2d, nlayers_2d, top=False)

    # set separation distance betwn 2d and substrate
    substrate_top_z = np.max(np.array([site.coords for site in substrate])[:, 2])
    mat_2d_bottom = np.min(np.array([site.coords for site in mat2d])[:, 2])

    # shift normal to the surface by 'separation'
    surface_normal = substrate.lattice.matrix[2, :]
    origin = np.array([0, 0, substrate_top_z])
    shift_normal = surface_normal / np.linalg.norm(surface_normal) * separation

    # generate all possible interfaces, one for each combination of
    # unique substrate and unique 2d materials site in the layers .i.e
    # an interface structure for each parallel shift
    # interface = 2D material + substrate    
    hetero_interfaces = []
    shifted_mat2ds = []
    for coord_i in coords_uniq_sub:
        for coord_j in coords_uniq_2d:
            interface = substrate.copy()  # in new code
            new_2d = deepcopy(mat2d)  # shfited 2d w/o substrate
            new_2d.remove_sites(range(0, new_2d.num_sites))

            # set the x,y coors for 2d
            shift_parallel = coord_i - coord_j
            shift_parallel[2] = 0
            # x,y coords shfited by shift_parallel, z by separation
            shift_net = shift_normal - shift_parallel
            # add 2d mat to substrate with shifted coords
            for idx, site in enumerate(mat2d):
                new_coords = site.coords  # xyz coords of 2d site
                # place 2d above substrate (z-dir change only)
                new_coords[2] = site.coords[2] - mat_2d_bottom
                new_coords = new_coords + origin + shift_net
                interface.append(site.specie, new_coords,
                                 coords_are_cartesian=True)
                new_2d.append(site.specie, new_coords,
                              coords_are_cartesian=True)  # add for name
            hetero_interfaces.append(interface)
            shifted_mat2ds.append(new_2d)  # shifted pure 2d 

    # get the names for the hetero_interfaces
    for td, iface in zip(shifted_mat2ds, hetero_interfaces):
        name = iface_name(td, substrate)
        iface.add_site_property('name', [name] * iface.num_sites)

    return hetero_interfaces


def hetero_interfaces(struct_2d, struct_sub, max_mismatch=0.01, max_area=200,
                      nlayers_2d=3, nlayers_sub=2, r1r2_tol=0.02, max_angle_diff=1, separation=3.4):
    """
    The given the 2D material and the substrate slab, the 2 slabs are
    combined to generate all possible unique structures for the
    2D material on the substrate surface generating all unique
    hetero_structures.

    Args:
        struct_2d: The 2-dimensions slab structure to combine with
            the substrate slab.
        struct_sub: The substrate slab structure which the 2D
            structure will be placed on top of.
        max_mismatch (float): The allowed misfit strain between struct_2d
            and struct_sub. Defaults to 0.01, multiply by 100 to obtain 
            percent strain.
        max_area (float): The maximum surface area that the hetero-structure
            is allowed to have.
        max_angle_diff (float): The maximum allowed deviation 
	        between the new superlattice and the old lattice a and b
	        vectors. Angle between a and b vectors: arccos[a.b/(|a||b|)].
	        Default value 1 degree.
        r1r2_tol (float): The tolerance in ratio of the r1 over r2
            ratio. Default 0.02.
        nlayers_2d (int): The number of layers of the 2D materials
            which you want to relax during the relaxation. Defaults
            to 3 layers.
        nlayers_sub (int): The number of layers of the substrate surface
            which you want to relax during the relaxation. Defaults
            to 2 layers.
        separation (float): Separation distance between struct_sub
            and struct_2d. Default 3.4 Angstroms.

    Returns:
        Unique hetero_structures list, last entry contains lattice alignment
        information. Site properties contain iface name.
    """
    struct_2d = deepcopy(struct_2d)
    struct_sub = deepcopy(struct_sub)

    # get aligned hetero_structures
    sub_aligned, td_aligned, alignment_info = aligned_hetero_structures(struct_2d, 
                                                                        struct_sub, 
                                                                        max_area=max_area, 
                                                                        max_mismatch=max_mismatch,
                                                                        max_angle_diff=max_angle_diff, 
                                                                        r1r2_tol=r1r2_tol)

    # exit if the aligned_hetero_structures returns None due to bad symmetry
    if sub_aligned is None:
        print('Aligned lattices failed: Rotating struct_2d')
        
        # rotate struct_2d to acute angle
        gamma = struct_2d.lattice.gamma
        if 180-gamma < gamma:
            struct_2d = rotate_to_acute(struct_2d)

        # get aligned heterostructures
        sub_aligned, td_aligned, alignment_info = aligned_hetero_structures(struct_2d, 
                                                                            struct_sub,
                                                                            max_area=max_area, 
                                                                            max_mismatch=max_mismatch,
                                                                            max_angle_diff=max_angle_diff, 
                                                                            r1r2_tol=r1r2_tol)
        
        # exit if the aligned_hetero_structures returns None due to bad symmetry
        if sub_aligned is None:
            print('Rotated aligned lattices failed\n')
            sys.exit()

    # merge substrate and mat2d in all possible ways
    # these are all the possible structures
    hetero_interfaces = []
    h_iface = generate_all_configs(td_aligned, sub_aligned, nlayers_2d=nlayers_2d, nlayers_substrate=nlayers_sub,
                                   separation=separation)

    # Return only unique structures - remove duplicates
    matcher = StructureMatcher(stol=0.001, ltol=0.001)
    matches = matcher.group_structures(h_iface)
    unique_structs = [matches[i][0] for i in range(len(matches))]
    unique_count = len(unique_structs)
    hetero_interfaces.extend(unique_structs)
    
    # get the formula units for the aligned 2d and substrate
    fu_2d, fu_sub = get_fu(struct_sub, struct_2d, sub_aligned, td_aligned)
    alignment_info.update({'fu_2d': fu_2d, 'fu_sub': fu_sub})

    # append the uv and angle mismatch to the interfaces
    hetero_interfaces.append(alignment_info)
    return hetero_interfaces


def iface_name(mat2d, substrate):
    """
    Helper function used to generate a unique interface name for a set of interfaces. The substrate's name
    will always be the same but the 2d materials indices shit and these are changed from one configuration to
    the next. 
    """
    # Gather Layer Data from LayerSolver
    sub_data = LayerSolver(structure=substrate)
    td_data = LayerSolver(structure=mat2d)

    # number of 2d layers to get the bottom layer
    numl_2d = td_data['num_layers']

    # top layer of the substrate
    top_layer = sub_data['Layer0']
    # bottom layer of the mat2d
    bottom_layer = td_data['Layer' + str(numl_2d - 1)]
    
    # Generate Substrate Name
    wyc_s = np.array(top_layer['wyckoffs']) # get unique wyckoffs 
    wyc_lyr_s, idx_s, multi_s = np.unique(wyc_s, return_index=True,
                                          return_counts=True) # get unique layer info
    
    # pull the unique atom species strings 
    s_s = [top_layer['sites'][idx].species_string
           for idx in idx_s]

    # create the substrates name
    sub_name = ','.join([str(s) + '(' + str(w) + ')'
                         for s, w in zip(s_s, wyc_lyr_s)])

    # Generate Two-D Name
    wyc_t = np.array(bottom_layer['wyckoffs'])
    wyc_lyr_t, idx_t, multi_t = np.unique(wyc_t, return_index=True,
                                          return_counts=True)
    s_t = [bottom_layer['sites'][idx].species_string
           for idx in idx_t]
    td_name = ';'.join([str(s) + '(' + str(w) + ')'
                        for s, w in zip(s_t, wyc_lyr_t)])

    name = sub_name + td_name
    return name


def remove_duplicates(uv_list, tm_list):
    """
    Remove duplicates based on a, b, alpha matching. Helper function.
    
    Args:
    	uv_list (list): the a and b lattice vectors to transform.
	tm_list (list): a list of transformation matrices.
	
    Returns: 
    	new_uv_list, new_tm_list
    """
    new_uv_list = []
    new_tm_list = []
    for sup_lat_n1, tm_n1 in zip(uv_list, tm_list):
        a1 = [np.linalg.norm(i[0]) for i in sup_lat_n1]
        b1 = [np.linalg.norm(i[1]) for i in sup_lat_n1]
        angles = [get_angle(i[0], i[1]) for i in sup_lat_n1]
        n1_lattices = [(a, b, alpha) for a, b, alpha in zip(a1, b1, angles)]
        for lat in n1_lattices:
            zround = np.array(n1_lattices).round(1)
            zlist = zround.tolist()
            zstr = np.array([str(j) for j in zlist])
            zu, zind = np.unique(zstr, return_index=True)
            unq_sup_lat = [sup_lat_n1[i] for i in zind]
            unq_tm = [tm_n1[i] for i in zind]
        new_uv_list.append(unq_sup_lat)
        new_tm_list.append(unq_tm)
    return new_uv_list, new_tm_list
