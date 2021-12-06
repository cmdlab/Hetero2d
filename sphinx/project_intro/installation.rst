=====================
Installing Hetero2d
=====================

IMPORTANT NOTE: Atomate and FireWorks do not run on Windows OS. You need a unix-based OS (Mac or Linux) in order for these packages to run. As such, all setup instructions are given for Unix systems. 

1. Download the repo from the green code icon or via github's commandline tool gh. 

  * ``gh repo clone cmdlab/Hetero2d`` (gh must be installed)
  * ``git clone https://github.com/cmdlab/Hetero2d.git``

2. Install Hetero2d in a clean enviromnent using python=3.6. I suggest using Anaconda3 to manange environments.

  * ``conda create --name hetero2d python=3.6``

3. Activate the *hetero2d* environment and run the line below to install Hetero2d. Must be in the directory where Hetero2d was downloaded to.

  * ``pip install -r requirements.txt``

4. After installation, Hetero2d needs to be added to your python path. This can be done by running the first line below **OR** by adding the 2nd line to your *.bashrc* file. This is only necessary if python cannot find the package or the setup.py failed for some reason.

  * ``python setup.py develop`` or ``python setup.py install``
  * ``export PYTHONPATH="$HOME/path_to_package/Hetero2d:$PYTHONPATH"``

5. If this is your first time installing the package dependencies listed below, please ensure you have followed the respective setup instructions:
  * `atomate <https://atomate.org/>`_
  * `FireWorks <https://materialsproject.github.io/fireworks/installation.html>`_
  * `pymatgen <https://pymatgen.org/installation.html>`_
  * `MPInterfaces <https://github.com/henniggroup/MPInterfaces>`_

6. To run jupyter notebooks on various resources the ipykernel has to be installed. Sometimes this isn't enough and you need explicitly add the kernel to the list of environments.

  * Activate your environment ``conda activate hetero2d``
  * ``python -m ipykernel install --user --name hetero2d``


Setting up dependancies
========================
The Hetero2d package dependancies have a lot of documentation to look over. I will highlight the essential documentation to get started as quickly as possible.

1. *atomate* requires the most set up. Mainly, creating a directory scaffold and writing the 5 required files to connect to the database and run jobs. (MongoDB or free Atlas MongoDB is required) 

2. *pymatgen* has a command line tool installed to set up default directory paths for psuedopotentials called *pmg*. There are 2 essential commands you have to run to use Hetero2d on any system. 

  * VASP POTCARs directory: ``pmg config -p <EXTRACTED_VASP_POTCAR> <MY_PSP>`` 
  * Default pseudopotential: ``pmg config --add PMG_DEFAULT_FUNCTIONAL PBE_54``

3. *MPInterfaces* has a config file similar to pymatgen. You need to set at least 2 parameters in the mpint_config.yaml file to remove the warning message. An example config file can be found on MPInterfaces github website.

  * mp_api: the_key_obtained_from_materialsproject
  * potentials: path_to_vasp_potcar_files


