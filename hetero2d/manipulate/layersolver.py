# coding: utf-8
# Copyright (c) CMD Lab Development Team.
# Distributed under the terms of the GNU License.

"""
Analyze an existing structure and parse it based on atomic layers. The cutoff can be
changed but defaults to 0.5 Angstroms. Returns a LayeredStructure dictionary.
"""

from copy import deepcopy
from operator import itemgetter

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

__author__ = "Sydney Olson"
__copyright__ = "Copyright 2020, CMD Lab"
__maintainer__ = "Sydney Olson, Tara M. Boland"
__email__ = "snolson1@asu.edu, tboland1@asu.edu"
__date__ = "June 5, 2020"


def LayerSolver(structure, cutoff=0.5):
    """
    Groups sites in a structure into layers according to z-coordinate. Returning layer specific data such as the
    z-coordinate of layer, a list of sites, the species of elements in the layer and the atom indices of that layer.

    Args:
        structure (Structure): Input structure to be analyzed.
        cutoff (float): The inter-layer separation distance that the algorithm search for.

    Returns:
        Layers dictionary returning layer specific information: each layer information is encoded with dictionary key
        num_layers (int): number of layers in the structure and Layer{i} with sub-keys
        avg_z (float): average z-coordinate of the layer
        sites (list of PeriodicSite): PeriodicSites for each site within the layer
        species (list): elemental species of each site within the layer
        site_idx (list): atom indices for species within the layer
        wyckoffs (list): wyckoff symbols for each site in the layer
    """
    # ensure structure is pymatgen structure obj
    if not isinstance(structure, Structure):
        # return print('Structure must be a pymatgen structure object.')
        print('Structure must be a pymatgen structure object.')

    # add wyckoff to site_property
    sga = SpacegroupAnalyzer(structure)
    sym_data = sga.get_symmetry_dataset()
    wyckoffs = sym_data['wyckoffs']
    structure.add_site_property('wyckoffs', values=wyckoffs)

    # add site indices & z_value to sort structure & track sites
    dict_structure = structure.as_dict()  # dictionary rep of struct
    [site.update({'index': idx, 'z_value': site['xyz'][2]})
     for idx, site in enumerate(dict_structure['sites'])]

    # sort sites based on z value
    dict_structure['sites'] = sorted(dict_structure['sites'], key=itemgetter('z_value'))

    # get lower & upper z value searching bounds
    z0 = dict_structure['sites'][0]['z_value']
    z1 = dict_structure['sites'][-1]['z_value']

    # initialized return data and layer counter
    n = 0  # number of layers to track
    z = deepcopy(z0)  # starting z value
    layered_structure = {}  # layer dependent dict

    while z <= z1:
        layer_indices = []  # layer atom indices
        layer_sites = []  # site dictionary object
        layer_wyc = []  # site wyckoff labels for each atom

        # search for all sites within [z,z+cutoff]
        # all sites found within this range compose a layer
        for site in dict_structure['sites']:
            if z <= site['z_value'] <= z + cutoff:
                layer_indices.append(site['index'])
                layer_sites.append(structure[site['index']])
                layer_wyc.append(site['properties']['wyckoffs'])

        # only analyze data if it exists
        if layer_indices:
            # compute: average z-coordinate for all layer sites & 
            # return layer species

            z_cords = [layer_site.z for layer_site in layer_sites]
            layer_species = [layer_site.species_string for layer_site in layer_sites]
            average_z = sum(z_cords) / len(z_cords)

            # dictionary of layer information computed
            layered_structure["Layer{}".format(n)] = {'avg_z': average_z,
                                                      'sites': layer_sites, 'species': list(np.unique(layer_species)),
                                                      'site_idx': layer_indices, 'wyckoffs': layer_wyc}
            n += 1  # increment the layer number

        # add cutoff to z to search next z area
        z += cutoff
    layered_structure['num_layers'] = n

    return layered_structure
