# Hetero2d
The Hetero2d package leverages well known computational tools: pymatgen, MPInterfaces, atomate, fireworks, and custodian to perform high-throughput *ab-initio* calculations. Hetero2d is tailored to addressing scientific questions regarding the stability of 2D-substrate hetero-structured materials using an all-in-one workflow approach to model the hetero-structures formed by arbitrary 2D materials and substrates. The workflow creates, computes, analyzes and stores all relevant simulation parameters and results in a queryable MongoDB database that can be accessed through our API.

Hetero2d provides automated routines for the generation of low-lattice mismatched hetero-structures for arbitrary 2D materials and substrate surfaces, the creation of van der Waals corrected density-functional theory (DFT) input files, the submission and monitoring of simulations on computing resources, the post-processing of the key parameters to compute (a) the interface interaction energy of 2D-substrate hetero-structures, (b) the identification of substrate-induced changes in interfacial structure, and (c) charge doping of the 2D material.

### IMPORTANT NOTE
Atomate and FireWorks do not run on Windows OS. You need a unix-based OS (Mac or Linux) in order for these packages to run. As such, all setup instructions are given for Unix systems. 

## Installing Hetero2d
1. Download the repo from the green code icon or via github's commandline tool gh. 
- ``gh repo clone cmdlab/Hetero2d`` (gh must be installed)
- ``git clone https://github.com/cmdlab/Hetero2d.git``
2. Install Hetero2d in a clean enviromnent using python=3.6. I suggest using Anaconda3 to manange environments. 
- ``conda create --name hetero2d python=3.6``
3. Activate the Hetero2d environment and run the line below in the Hetero2d directory to install:
- ``pip install -r requirements.txt``
- ``python setup.py develop`` or ``python setup.py install`` to add Hetero2d to your python path.
4. After installation, add Hetero2d to your python path by *.bashrc* file. Only necessary if python cannot find the package.
- `export PYTHONPATH="$HOME/path_to_package/Hetero2d:$PYTHONPATH"`
5. If this is your first time installing the package dependencies listed below, please ensure you have followed the respective setup instructions:
- [atomate](https://atomate.org/)  
- [FireWorks](https://materialsproject.github.io/fireworks/installation.html)
- [pymatgen](https://pymatgen.org/installation.html)
- [MPInterfaces](https://github.com/henniggroup/MPInterfaces)
6. To run jupyter notebooks on various resources make sure you add the kernel to the list of environments.
- First activate your environment `conda activate Hetero2d`
- Then run `python -m ipykernel install --user --name Hetero2d`

The Hetero2d package dependancies have a lot of documentation to look over. I will highlight the essential documentation to get started as quickly as possible.
1. *atomate* requires the most set up. Mainly, creating a directory scaffold and writing the 5 required files to connect to the database and run jobs. (MongoDB or free Atlas MongoDB is required) 
2. *pymatgen* has a command line tool installed to set up default directory paths called pmg. There are 2 essential commands you have to run to use Hetero2d on any system. 
- Reference directory for the VASP POTCARs. You need to have the POTCARs from VASP yourself.
  - `pmg config -p <EXTRACTED_VASP_POTCAR> <MY_PSP>` 
- Default pseudopotential files from VASP 
  - `pmg config --add PMG_DEFAULT_FUNCTIONAL PBE_54`
3. *MPInterfaces* a config file similar to pymatgen. You need to set at least 2 parameters in the mpint_config.yaml file. An example config file can be found on the github website.
  - mp_api: the_key_obtained_from_materialsproject
  - potentials: path_to_vasp_potcar_files

## Package Description
The 2D-substrate hetero-structure workflow takes a given 2D material, 3D phase (bulk) of the 2D material, and a substrate, relaxes the structures using vdW-corrected DFT and creates hetero-structures subject to user constraints. The workflow analyzes and stores energetic stability information of various 2D hetero-structured materials to predict the feasibility of a substrate stabilizing a meta-stable 2D material.
