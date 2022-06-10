# coding: utf-8
# Copyright (c) CMD Lab Development Team.
# Distributed under the terms of the GNU License.

"""
These modules are used to create the heterostructure configurations given a 2d and substrate slab. The code is adopted
from the MPInterfaces to ensure the code is compatible with the FireWorks and atomate architecture, to fix minor bugs in
the original code, and return interface matching criteria.
"""

import numpy as np, sys
from six.moves import range
from copy import deepcopy

from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import Slab

from hetero2d.manipulate.utils import slab_from_struct, get_fu

__author__ = "Tara M. Boland, Arunima Singh"
__copyright__ = "Copyright 2022, CMD Lab"
__maintainer__ = "Tara M. Boland"
__email__ = "tboland1@asu.edu"
__date__ = "June 10, 2022"


def hetero_interfaces(struct_2d, struct_sub, max_mismatch = 0.01, max_area = 200, nlayers_2d = 3, nlayers_sub = 2,
                      r1r2_tol = 0.02, max_angle_diff = 1, separation = 3.4, symprec = False):
    """
    The given the 2D material and the substrate slab, the 2 slabs are combined to generate all possible unique
    structures for the 2D material on the substrate surface generating all unique_structs hetero_structures.

    Args:
        struct_2d: The 2-dimensions slab structure to combine with the substrate slab.
        struct_sub: The substrate slab structure which the 2D structure will be placed on top of.
        max_mismatch (float): The maximum allowed lattice strain applied to the struct_2d. Defaults to 0.01, multiply by
            100 to obtain percent strain.
        max_area (float): The maximum surface area of the supercell to search for potential lattice matches between
            struct_2d and struct_sub. Typical values 30-200 Angstroms.
        max_angle_diff (float): The maximum allowed deviation between the new superlattice and the old lattice a and b
	        vectors. Angle between a and b vectors: arccos[a.b/(|a||b|)]. Default value 1 degree.
        r1r2_tol (float): The maximum allowed deviation between the scaled surface area of the 2d and substrate. Typical
            values range from 0.01 to 0.1.
        nlayers_2d (int): The number of layers of the 2D materials which you want to relax during the relaxation.
            Defaults to 3 layers.
        nlayers_sub (int): The number of layers of the substrate surface which you want to relax during the relaxation.
            Defaults to 2 layers with bottom layer frozen.
        separation (float): Separation distance between struct_sub and struct_2d. Default 3.4 Angstroms.
	    symprec (bool/float): Perform symmetry matching to the specified tolerance for symmetry finding between the
	        aligned and original 2D structure. Enable if you notice the 2D lattice is disformed. Defaults to False.

    Returns:
        Unique hetero_structures list, last entry contains lattice alignment information. Site properties contain iface
        name.
    """
    struct_2d = deepcopy(struct_2d)
    struct_sub = deepcopy(struct_sub)

    # get aligned hetero_structures
    sub_aligned, td_aligned, alignment_info = aligned_hetero_structures(struct_2d,
                                                                        struct_sub,
                                                                        max_area=max_area,
                                                                        max_mismatch=max_mismatch,
                                                                        max_angle_diff=max_angle_diff,
                                                                        r1r2_tol=r1r2_tol,
                                                                        symprec=symprec)

    # exit if the aligned_hetero_structures returns None due to bad symmetry
    if sub_aligned is None:
        print('Aligned lattices failed: Rotating struct_2d')

        # rotate struct_2d to acute angle
        gamma = struct_2d.lattice.gamma
        if 180 - gamma < gamma:
            struct_2d = rotate_to_acute(struct_2d)

        # get aligned heterostructures
        sub_aligned, td_aligned, alignment_info = aligned_hetero_structures(struct_2d,
                                                                            struct_sub,
                                                                            max_area=max_area,
                                                                            max_mismatch=max_mismatch,
                                                                            max_angle_diff=max_angle_diff,
                                                                            r1r2_tol=r1r2_tol,
                                                                            symprec=symprec)

        # exit if the aligned_hetero_structures returns None due to bad symmetry
        if sub_aligned is None:
            print('Rotated aligned lattices failed\n')
            sys.exit()

    # merge substrate and mat2d in all possible ways these are all the possible structures
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


def aligned_hetero_structures(struct_2d, struct_sub, max_mismatch=0.01, max_area=200, nlayers_2d=3, nlayers_sub=2,
                              r1r2_tol=0.02, max_angle_diff=1, separation=3.4, symprec=False):
    """
    Given the 2 slab structures and the alignment parameters, return slab structures with lattices that are aligned with
    respect to each other.

    Args:
        slab_2d (Structure): Two dimensional slab structure object.
        slab_sub (Structure): Substrate slab structure object.
        max_area (float): The maximum area you want to search for a matching lattice. Defaults to 200 sqr. Angstroms.
        max_mismatch (float):  The maximum mismatch between the a and b lattice vectors of the aligned 2D and substrate
            lattice vectors. Defaults to 0.01, multiply by 100 to obtain percent.
        max_angle_diff (float): The angle deviation between the a and b lattice vectors between the old lattice vectors
            and the new lattice vectors. Defaults to 1 degree.
        r1r2_tol (float): Allowed area approximation tolerance for the two lattices. Defaults to 0.02.
        nlayers_substrate (int): number of substrate layers. Defaults to 2 layers.
        nlayers_2d (int): number of 2d material layers. Defaults to 3 layers.
        separation (float): separation between the substrate and the 2d material. Defaults to 3.4 angstroms.
        symprec (bool/float): Perform symmetry matching to the specified tolerance for symmetry finding between the
            aligned and original 2D structure. Enable if you notice the 2D lattice is disformed. Defaults to False.
        
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

    # 4 each lattice vector in r_list reduce the super_lattice vectors to find new unit cells
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
                        # double check if this i cell reduction params the cell reduction does not take into account
                        # atomic positions just cell vectors
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
    # map the intial slabs to the newly found matching  lattices to avoid numerical issues with find_mapping
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
    mat2d.lattice = lmap

    ## ensure that the symmetry of the 2d and aligned 2d agree
    if symprec:
        sg_mat2d = SpacegroupAnalyzer(mat2d, symprec=symprec)
        sg_align = SpacegroupAnalyzer(struct_2d, symprec=symprec)
    
        m2d, align = {}, {}
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
       
    else:
        return substrate, mat2d, alignment_info

def generate_all_configs(mat2d, substrate, nlayers_2d=3, nlayers_substrate=2, separation=3.4):
    """
    For the given lattice matched 2D material and substrate structures, this functions computes all unique (Wyckoff)
    sites of the mat2d and substrate. The unique sites are iteratively matched between the mat2d and substrate stacking
    the unique sites on top of each other separated by the separation distance parameter. This subsequently generates
    all possible 2d/substrate heterostructure configurations stacked over high symmetry points. All unique structures
    are returned. To identify the wyckoff sites stacked on top of each other the name is attached to each structure
    under site_properties.

    Args:
        mat2d (Structure): Lattice and symmetry-matched 2D material structure.
        substrate (Structure): Lattice and symmetry-matched 2D substrate structure.
        nlayers_substrate (int): number of substrate layers. Defaults to 2 layers.
        nlayers_2d (int): number of 2d material layers. Defaults to 3 layers.
        separation (float): separation between the substrate and the 2d material. Defaults to 3.4 angstroms.
    
    Returns:
        list of hetero_interface configurations 
    """
    if not (mat2d and substrate):
        sys.exit()

    # unique site coordinates in the substrate top layers
    coords_uniq_sub, uniq_sub_idx = get_uniq_layercoords(substrate, nlayers_substrate, top=True)
    # add the wyckoff site to each site in substrate
    spg = SpacegroupAnalyzer(substrate)
    substrate.add_site_property('wyckoffs', spg.get_symmetry_dataset()['wyckoffs'])

    # unique site coordinates in the 2D material bottom layers
    coords_uniq_2d, uniq_2d_idx = get_uniq_layercoords(mat2d, nlayers_2d, top=False)
    # add the wyckoff site to each site in mat2d
    spg = SpacegroupAnalyzer(mat2d)
    mat2d.add_site_property('wyckoffs', spg.get_symmetry_dataset()['wyckoffs'])

    # set separation distance betwn 2d and substrate
    substrate_top_z = np.max(np.array([site.coords for site in substrate])[:, 2])
    mat_2d_bottom = np.min(np.array([site.coords for site in mat2d])[:, 2])

    # shift normal to the surface by 'separation'
    surface_normal = substrate.lattice.matrix[2, :]
    origin = np.array([0, 0, substrate_top_z])
    shift_normal = surface_normal / np.linalg.norm(surface_normal) * separation

    # generate all possible interfaces, one for each combination of unique substrate and unique 2d materials site in the
    # layers .i.e an interface structure for each parallel shift interface = 2D material + substrate
    hetero_interfaces = []
    for coord_i, sub_idx in zip(coords_uniq_sub, uniq_sub_idx):
        for coord_j, td_idx in zip(coords_uniq_2d, uniq_2d_idx):
            # create interface structure to append the shifted 2d material
            interface = substrate.copy()
            interface.remove_site_property('wyckoffs') # remove wyckoffs
            
            # get the wyckoff site of the 2 sites being aligned on top of each other
            td_wyckoff = mat2d[td_idx].properties['wyckoffs']
            sub_wyckoff = substrate[sub_idx].properties['wyckoffs']
            name='2D-'+td_wyckoff+' '+'Sub-'+sub_wyckoff # stacking of 2d on sub wyckoff

            # set the x,y coors for 2d
            shift_parallel = coord_i - coord_j
            shift_parallel[2] = 0
            # x,y coords shfited by shift_parallel, z by separation
            shift_net = shift_normal - shift_parallel

            # add 2d mat to substrate with shifted coords
            mat2d2=deepcopy(mat2d)
            for idx, site in enumerate(mat2d2):
                new_coords = site.coords  # xyz coords of 2d site
                # place 2d above substrate (z-dir change only)
                new_coords[2] = site.coords[2] - mat_2d_bottom
                new_coords = new_coords + origin + shift_net
                interface.append(site.specie, new_coords,
                                 coords_are_cartesian=True)
            interface.add_site_property('name', [name] * interface.num_sites)
            hetero_interfaces.append(interface)

    return hetero_interfaces


def get_uniq_layercoords(struct, nlayers, top=True):
    """
    Returns the coordinates and indices of unique sites in the top or bottom nlayers of the given structure.

    Args:
        struct (Structure): Input structure.
        nlayers (int): Number of layers.
        top (bool): Top or bottom layers, default is top layer.

    Returns:
        numpy array of unique coordinates, corresponding list of atom indices
    """
    coords = np.array([site.coords for site in struct])
    z = coords[:, 2]
    z = np.around(z, decimals=4)
    zu, zuind = np.unique(z, return_index=True)
    z_nthlayer = z[zuind[-nlayers]]
    zfilter = (z >= z_nthlayer)
    if not top:
        z_nthlayer = z[zuind[nlayers - 1]]
        zfilter = (z <= z_nthlayer)
    # site indices in the layers
    indices_layers = np.argwhere(zfilter).ravel()
    sa = SpacegroupAnalyzer(struct)
    symm_data = sa.get_symmetry_dataset()
    # equivalency mapping for the structure i'th site in the struct equivalent to eq_struct[i]'th site
    eq_struct = symm_data["equivalent_atoms"]
    # equivalency mapping for the layers
    eq_layers = eq_struct[indices_layers]
    # site indices of unique atoms in the layers
    __, ueq_layers_indices = np.unique(eq_layers, return_index=True)
    # print(ueq_layers_indices)
    indices_uniq = indices_layers[ueq_layers_indices]
    # coordinates of the unique atoms in the layers
    return coords[indices_uniq], indices_uniq


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
    new_structure = Structure(lattice=a_prime, species=species, coords=frac_coords, coords_are_cartesian=False,
                              to_unit_cell=True)

    return new_structure


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


def reduced_supercell_vectors(ab, n):
    """
    Obtain all possible reduced in-plane lattice vectors and transition matrices for the given starting unit cell
    lattice vectors (ab) and the supercell size n.

    Args:
        ab ():
        n (int):

    Returns:
        uv_list (3xn list) in-plane lattice vectors, tm_list
        ([2x2]xn list) transition matrices
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
    Yields a list of 2x2 transformation matrices for the given supercell.

    Args:
        n (list): size.

    Returns:
        2x2 transformation matrics
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
    Apply the transformation matrix to lattice vectors to obtain the new u and v supercell lattice vectors.

    Args:
        ab (list): The original lattice vectors to apply the transformation matrix.
        t_mat (list): The transformation matrices returned by get_trans_matrices.

    Returns:
        [u,v] - the supercell lattice vectors
    """
    u = np.array(ab[0]) * t_mat[0][0] + np.array(ab[1]) * t_mat[0][1]
    v = np.array(ab[1]) * t_mat[1][1]
    return [u, v]


def get_reduced_uv(uv, tm):
    """
    Reduce the lattice vectors to remove equivalent lattices.

    Args:
        uv ():
        tm ():

    Returns:
         [u, v], tm1 - reduced lattice vectors and associated transformation matrix
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


def get_r_list(area1, area2, max_area, tol = 0.02):
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
    Compute the percent mistmatch between the lattice vectors a and b.

    Args:
        a (list): A 3x1 list representing the 'a' lattice vector.
        b (list): A 3x1 list representing the 'b' lattice vector.

    Returns:
        mismatch (float) - percent mismatch between lattice vector a and b
    """
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(b) / np.linalg.norm(a) - 1


def get_angle(a, b):
    """
    Compute the angle between lattice vectors a and b in degrees.

    Args:
        a (list): A 3x1 list representing the 'a' lattice vector.
        b (list): A 3x1 list representing the 'b' lattice vector.

    Returns:
        angle (float) - angle between lattice vector a and b

    """
    a = np.array(a)
    b = np.array(b)
    return np.arccos(
        np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)) * 180 / np.pi


def get_area(uv):
    """
    Compute the area enclosed by the uv vectors using the formula of a parallelogram.

    Args:
        uv (list): A 3x2 list representing the 'u' and 'v' lattice vectors.

    Returns:
        area (float) - area in units of the lattice vectors.
    """
    a = uv[0]
    b = uv[1]
    return np.linalg.norm(np.cross(a, b))
