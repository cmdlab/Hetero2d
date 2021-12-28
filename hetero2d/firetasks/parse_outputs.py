# coding: utf-8
# Copyright (c) CMD Lab Development Team.
# Distributed under the terms of the GNU License.

"""
These modules parse output from VASP runs, calculate various energetic and structural 
parameters and update a mongoDB with the relevant information.
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import glob, os, re, gridfs, json, zlib
import numpy as np
from bson import ObjectId

from monty.os.path import which
from monty.json import MontyEncoder, jsanitize
from monty.serialization import dumpfn

from pymatgen import Structure
from pymatgen.io.vasp import Potcar
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.sets import get_vasprun_outcar
from pymatgen.command_line.bader_caller import bader_analysis_from_path

from fireworks.core.firework import FiretaskBase, FWAction
from fireworks.utilities.fw_serializers import DATETIME_HANDLER
from fireworks.utilities.fw_utilities import explicit_serialize

from atomate.common.firetasks.glue_tasks import get_calc_loc 
from atomate.utils.utils import get_logger, env_chk, get_meta_from_structure
from atomate.vasp.drones import VaspDrone

from hetero2d.manipulate.utils import tag_iface, get_mongo_client, get_FWjson
from hetero2d.manipulate.utils import vtotav

bader_exe_exists = which("bader") or which("bader.exe")


__author__ = 'Tara M. Boland'
__copyright__ = "Copyright 2020, CMD Lab"
__maintainer__ = "Tara M. Boland"
__email__ = 'tboland1@asu.edu'

logger = get_logger(__name__)

# Analysis
@explicit_serialize
class HeteroAnalysisToDb(FiretaskBase):
    """
    Enter heterostructure workflow analysis into the database. Determines 
    what data to enter into the database based on what calculation was
    performed.

    Args: 
        db_file (str): Path to file containing the database credentials.
            Supports env_chk. Default: write data to JSON file.
        task_label (str): The task_label (firework name) automatically generated
            by the FireWorks in Hetero2d. Used to determine what type of calculation
            is being parsed.
        
    Optional params:
        calc_dir (str): Path to dir (on current filesystem) that contains VASP
            output files. Default: use current working directory.
        calc_loc (str/bool): If True will set most recent calc_loc. If str
            will search calc_locs for the matching name (most recent).
        dos (bool/str): If True, parses the density of states assuming uniform
            mode. Set to line if you are parsing bandstructures. Data stored 
            in GridFS. Default set to False.
        bader (bool): If True, bader analysis is performed for the current
            directory. The bader.exe must exist on the path. Default set to 
            False.
        cdd (bool): If True, the charge density difference is performed. Default 
            set to False.
        Adsorption_Energy (bool): If True, the adsorption formation energy
            analysis is calculated for the run the results are stored with the
            structure in the database under the Adsorption_Energy key.
        Binding_Energy (bool): If True, the binding energy will be calculated 
            and stored in the database under the Binding Energy key.
        Formation_Energy (bool): If True, the formation energy will be calculated
            and stored with structure in the database under the Formation_Energy
            key.
        additional_fields (dict): Dict of additional fields to add to the database.
    """
    required_params = ["db_file", "task_label"]
    optional_params = ["dos", "bader", "cdd", "Adsorption_Energy", "Binding_Energy", 
            "Formation_Energy", "additional_fields"]

    def run_task(self, fw_spec):
        logger.info("Starting HeteroAnalysisToDb: collecting information.")
        db_file = env_chk('>>db_file<<', fw_spec)
        task_label = self.get("task_label", None) # get task_label

        # determine what data to insert into the database
        additional_fields = self.get("additional_fields", None) # update additional_fields

        # define input to push to mod_spec and stored data
        if re.search("Optimization", task_label):
            logger.info("Cleaning up TaskDoc")
            stored_data, mod_spec = {'analysis_info': {}}, [{'_push': {'analysis_info': {}}}]
            info = {}
            [info.update(element) for element in fw_spec.get('analysis_info', [{}])]
        else:
            stored_data, mod_spec = None, None

        print('### Parsing Calculation', task_label)
        #      Bulk Substrate Optimization Analysis     #
        if re.search("Bulk Structure Optimization", task_label):
            logger.info("PASSING PARAMETERS TO TASKDOC: Bulk")
            struct_bulk, N_bulk, E_bulk = HeteroTaskDoc(self, fw_spec,
                                                        task_label, 'Bulk', additional_fields,
                                                        db_file)

        # Oriented Substrate Slab Optimization Analysis #
        if re.search("Slab Structure Optimization", task_label):
            logger.info("PASSING PARAMETERS TO TASKDOC: Slab")
            struct_sub, N_sub, E_sub = HeteroTaskDoc(self, fw_spec,
                                                     task_label, 'Substrate', additional_fields,
                                                     db_file)
            info.update({'E_sub': E_sub, 'N_sub': N_sub})
            stored_data["analysis_info"].update(info)
            mod_spec[0]['_push']['analysis_info'].update(info)

        #      2D Structure Optimization Analysis       #
        if re.search("[^3D]2D Structure Optimization", task_label):
            logger.info("PASSING PARAMETERS TO TASKDOC: 2D")
            struct_2D, N_2D, E_2D = HeteroTaskDoc(self, fw_spec, task_label, '2D', 
                                                  additional_fields, db_file)
            stored_data["analysis_info"].update({'N_2d': struct_2D.num_sites,
                                                 'E_2d': E_2D})
            mod_spec[0]['_push']['analysis_info'].update({'N_2d': struct_2D.num_sites,
                                                          'E_2d': E_2D})

        #     3D2D Structure Optimization Analysis      #
        if re.search("3D2D Structure Optimization", task_label):
            logger.info("PASSING PARAMETERS TO TASKDOC: 3D2D")
            struct_3D2D, N_3D2D, E_3D2D = HeteroTaskDoc(self, fw_spec, task_label, 
                                                        '3D2D', additional_fields, db_file)
            stored_data["analysis_info"].update({'N_3d2d': N_3D2D,
                                                 'E_3d2d': E_3D2D})
            mod_spec[0]['_push']['analysis_info'].update({'N_3d2d': N_3D2D,
                                                          'E_3d2d': E_3D2D})

        #     2d on substrate Optimization Analysis     #
        if re.search("Heterostructure Optimization:", task_label):
            logger.info("PASSING PARAMETERS TO TASKDOC: 2D_on_Substrate")
            struct_2Dsub, N_2Dsub, E_2Dsub = HeteroTaskDoc(self, fw_spec, task_label, 
                                                           "2D_on_Substrate", 
                                                           additional_fields,
                                                           db_file)

        ##################################################        
        #      Density of States and Bader Analysis      #
        dos, bader, cdd = [self.get(i, False) for i in ['dos','bader','cdd']]
        if True in [dos, bader, cdd]:
            logger.info("PASSING PARAMETERS TO TASKDOC: Dos and Bader")
            DosBaderTaskDoc(self, fw_spec, task_label, "DosBader", 
                            dos, bader, cdd, additional_fields, db_file)
        return FWAction(stored_data=stored_data, mod_spec=mod_spec)


def HeteroTaskDoc(self, fw_spec, task_name, task_collection, 
                  additional_fields=None, db_file=None):
    """
    Insert a new doc for the 2d, 3d2d, bulk, substrate slab, 2d-subs
    generator, or 2d-subs configuration into the database.

    Args:
        self (self): The self parameter for HeteroAnalysisToDb.
        fw_spec (dict): A dictionary containing all the information
            linked to this firework.
        task_name (str): The name of the firework being analyzed.
        task_collection (str): The name of the task collection you
            want to push the data to.
        additional_fields (dict): a dictionary containing additional
            information you want added to the database.
        db_file (str): a string representation for the location of
            the database file.

    Returns:
        Analyzed structure, number of sites, and energy
    """
    dumpfn(fw_spec, 'fw_spec.json')

    # get directory info
    calc_dir = os.getcwd()
    if "calc_dir" in self:
        calc_dir = self["calc_dir"]
    elif self.get("calc_loc"):
        calc_dir = get_calc_loc(self["calc_loc"], fw_spec["calc_locs"])["path"]
    logger.info("PARSING DIRECTORY: {}".format(calc_dir))

    ##########################
    # Analyze VASP Directory #
    drone = VaspDrone(
        additional_fields=self.get("additional_fields"),
        parse_dos=self.get("parse_dos", False),
        bandstructure_mode=self.get("bandstructure_mode", False),
        parse_chgcar=self.get("parse_chgcar", False),
        parse_aeccar=self.get("parse_aeccar", False))
    task_doc = drone.assimilate(calc_dir)

    # collect analysis info
    E = task_doc["calcs_reversed"][0]["output"]['energy']
    final_struct = Structure.from_dict(task_doc["calcs_reversed"][0]["output"]['structure'])
    # get the actual initial POSCAR for database
    f = glob.glob("POSCAR.orig*")[0]
    # this takes the poscar and preps it for database doc 
    init_struct = Structure.from_file(f, False)
    cif = final_struct.to(fmt='cif')
    N = final_struct.num_sites
    fw_id = get_FWjson()['fw_id']  # get fw_id
    info = {}
    [info.update(element) for element in fw_spec.get('analysis_info', [{}])]

    # standard database information 
    heterostructure_dict = {'compound': task_doc['formula_pretty'],
                            'dir_name': task_doc['dir_name'], 'fw_id': fw_id, 
                            'task_label': task_name,
                            'final_energy': E, 'initial_structure': init_struct.as_dict(),
                            'final_structure': final_struct.as_dict(), 'cif': cif,
                            'analysis_data': info, 
                            "metadata": get_meta_from_structure(structure=final_struct)}

    ##########################################
    if task_collection == "2D_on_Substrate":
        # get iface_idx
        nlayers_2d = additional_fields.get('heterotransformation')['nlayers_2d']
        layer_indices = tag_iface(init_struct, nlayers_2d)
        iface_idx = {'iface_idx': layer_indices}
        heterostructure_dict.update(iface_idx)
    ##########################################

    # additional appended information
    if additional_fields:
        for key, value in additional_fields.items():
            heterostructure_dict[key] = value

    # additional energetic information
    # NOTE: never pull the *_Energy tags from fw_spec only
    # pull them from self. 
    Formation_Energy = self.get("Formation_Energy", None)
    Binding_Energy = self.get("Binding_Energy", None)
    Adsorption_Energy = self.get("Adsorption_Energy", None)

    if Formation_Energy:
        logger.info("TaskDoc: Computing Formation Energy")
        E_form = (info['E_2d'] / info['N_2d']) - (info['E_3d2d'] / info['N_3d2d'])
        energetics = {"E_form": E_form}
        heterostructure_dict['Formation_Energy'] = energetics

    if Binding_Energy:
        logger.info("TaskDoc: Computing Binding Energy")
        struct_2d = additional_fields.get('heterotransformation')['struct_2d']
        if not isinstance(struct_2d, Structure):
            struct_2d = Structure.from_dict(struct_2d)
        struct_sub = additional_fields.get('heterotransformation')['struct_sub']
        if not isinstance(struct_sub, Structure):
            struct_sub = Structure.from_dict(struct_sub)
        N_iface = final_struct.num_sites
        # calc E_ads 
        align_info = heterostructure_dict['tags']['alignment_info']
        n_2d, n_sub = align_info['fu_2d'], align_info['fu_sub']
        E_bind = (n_2d * info['E_2d'] + n_sub * info['E_sub'] - E) / (n_2d * info['N_2d'])
        energetics = {"E_bind": E_bind, 'fu_2d': n_2d, 'fu_sub': n_sub}
        heterostructure_dict['Binding_Energy'] = energetics

    if Adsorption_Energy:
        logger.info("TaskDoc: Computing Adsorption Formation Energy")
        E_form = (info['E_2d'] / info['N_2d']) - (info['E_3d2d'] / info['N_3d2d'])

        # account for not 1:1 scaling of the heterointerface structures
        struct_2d = additional_fields.get('heterotransformation')['struct_2d']
        if not isinstance(struct_2d, Structure):
            struct_2d = Structure.from_dict(struct_2d)
        struct_sub = additional_fields.get('heterotransformation')['struct_sub']
        if not isinstance(struct_sub, Structure):
            struct_sub = Structure.from_dict(struct_sub)
        N_iface = final_struct.num_sites
        # calc E_ads 
        align_info = heterostructure_dict['tags']['alignment_info']
        n_2d, n_sub = align_info['fu_2d'], align_info['fu_sub']
        E_ads = E_form - (n_2d * info['E_2d'] + n_sub * info['E_sub'] - E) / (n_2d * info['N_2d'])
        film_elements = list(struct_2d.composition.get_el_amt_dict().keys())
        substrate_elements = list(struct_sub.composition.get_el_amt_dict().keys())
        if 'tags' in heterostructure_dict.keys():
            heterostructure_dict['tags'].update({'film_elements': film_elements,
                                                 'substrate_elements': substrate_elements})
        energetics = {"E_ads": E_ads, 'fu_2d': n_2d, 'fu_sub': n_sub}
        heterostructure_dict['Adsorption_Energy'] = energetics

    # connect to database & insert info
    h_dict = {**task_doc, **heterostructure_dict}
    heterostructure_dict = jsanitize(h_dict)
    if not db_file:
        with open('task.json', 'w') as f:
            f.write(json.dumps(heterostructure_dict, default=DATETIME_HANDLER))
    else:
        db_type = fw_spec.get('db_type', None)
        conn, database = get_mongo_client(db_file, db_type=db_type)
        db = conn[database]
        col = db[task_collection]
        col.insert_one(heterostructure_dict)
        conn.close()
    dumpfn(heterostructure_dict, 'heterostructure_doc.json')

    return final_struct, N, E

def DosBaderTaskDoc(self, fw_spec, task_name, task_collection, dos,
                    bader, cdd, additional_fields=None, db_file=None):
    """
    Insert a new task document into task_collection specified by db_file
    that contains dos, bader, and charge density difference for a set of
    calculations.

    Args:
        fw_spec (dict): A dictionary containing all the information
            linked to this firework.
        task_name (str): The name of the firework being analyzed.
        task_collection (str): The name of the task collection you
            want to push the data to.
        dos (bool/str): If True, parses the density of states assuming uniform
            mode. Set to line if you are parsing bandstructures. Data stored 
            in GridFS. Default set to False.
        bader (bool): If True, bader analysis is performed for the current
            directory. The bader.exe must exist on the path. Default set to 
            False.
        cdd (bool): If True, the charge density difference is performed. Default 
            set to False.
        additional_fields (dict): A dictionary containing additional
            information you want added to the database.
        db_file (str): a string representation for the location of
            the database file.
    """
    ####TODO: make sure the task doc has access to the iface_idx for combined systems
    #### name for the DOS

    # get directory info
    calc_dir = os.getcwd()
    if "calc_dir" in self: # passed calc dir
        calc_dir = self["calc_dir"]
    elif self.get("calc_loc"): # find the calc_loc in fw_spec
        calc_dir = get_calc_loc(self["calc_loc"], fw_spec["calc_locs"])["path"]
    logger.info("PARSING DIRECTORY: {}".format(calc_dir))

    drone = VaspDrone()
    task_doc = drone.assimilate(calc_dir)

    # TASKDOC: DOS processing
    vrun, outcar = get_vasprun_outcar('.')
    if dos:
        try:
            dos_dict = vrun.complete_dos.as_dict()
        except Exception:
            raise ValueError("No valid dos data exist")
    else:
        dos_dict = None

    # TASKDOC: Bader processing
    if bader:
        ba = bader_analysis_from_path(path=calc_dir)
        structure=Structure.from_dict(task_doc["calcs_reversed"][0]["output"]['structure'])
        potcar=Potcar.from_file(filename=os.path.join(calc_dir, 'POTCAR.gz'))   
        nelectrons={p.element: p.nelectrons for p in potcar}                    
        ba['species']=[s.species_string for s in structure.sites]               
        ba['nelectrons']=[nelectrons[s] for s in ba['species']]                 
        ba['zcoords']=[s.coords[2] for s in structure.sites]  
    else:
        ba = {}

    # TASKDOC: Charge Density Difference processing
    if cdd:
        calc_locs = fw_spec['calc_locs']
        iso1_path = [loc['path'] for loc in calc_locs   
                                    if re.search('ISO 1 NSCF: DOS and Bader', loc['name'])]
        iso2_path = [loc['path'] for loc in calc_locs 
                                    if re.search('ISO 2 NSCF: DOS and Bader', loc['name'])]
        comb_path = [loc['path'] for loc in calc_locs 
                                    if re.search('Combined NSCF: DOS and Bader', loc['name'])]
        chg1, chg1, chg_comb = [vtotav(glob.glob(os.path.join(fullpath, "CHGCAR*")))
                                        for fullpath in [iso1_path, iso2_path, comb_path]]
        chg_cdd = list( np.array(chg_comb['chg_density']) - \
                                (np.array(chg1['chg_density']) + np.array(chg2['chg_density'])) )
        cdd_dict['cdd'] = {'energies': chg1['energies'], 'chg_density': chg_cdd}
    else:
        cdd_dict = {}

    # connect to database & insert info
    doc_keys = ['dir_name', 'run_stats', 'chemsys', 'formula_reduced_abc', 'completed_at', 'nsites', 
        'composition_unit_cell', 'composition_reduced', 'formula_pretty', 'elements', 'nelements', 
        'input', 'last_updated', 'custodian', 'orig_inputs']
    store_doc = { key: task_doc[key] for key in doc_keys }
    e_dict = {**ba, **cdd_dict, **store_doc}
    electronic_dict = jsanitize(e_dict) # export analysis in case update fails

    # additional appended information
    if additional_fields:
        for key, value in additional_fields.items():
            electronic_dict[key] = value
    
    # dump electronic properties to directory
    with open('electronic_property.json', 'w') as f:
        f.write(json.dumps(electronic_dict, default=DATETIME_HANDLER))
    
    # insert task doc without the dos b/c its to large
    conn, database = get_mongo_client(db_file, db_type=fw_spec.get('db_type', None))
    db = conn[database]
    col = db[task_collection]
    t_id = col.insert(electronic_dict)
    
    # insert the dos document into gridfs
    if dos_dict:
        # get object id
        oid = oid or ObjectId()

        # Putting task id in the metadata subdocument as per mongo specs
        m_data = {"compression": "zlib", "task_id": t_id}
        # always perform the string conversion when inserting directly to gridfs
        d = json.dumps(dos_dict, cls=MontyEncoder)
        d = zlib.compress(d.encode(), True)
        # connect to gridFS
        fs = gridfs.GridFS(database, f"{task_collection}_fs")
        fs_id = fs.put(d, _id=oid, metadata=m_data)

        # insert into gridfs
        col.update_one({"task_id": t_id}, {"$set": {f"{dos}_compression": "zlib"}})
        col.update_one({"task_id": t_id}, {"$set": {f"{dos}_fs_id": fs_id}})

        conn.close()
