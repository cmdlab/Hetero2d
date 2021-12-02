============
Introduction
============

The Hetero2d package leverages well known computational tools: pymatgen, MPInterfaces, atomate, fireworks, and custodian to perform high-throughput *ab-initio* calculations. Hetero2d is tailored to addressing scientific questions regarding the stability of 2D-substrate hetero-structured materials using an all-in-one workflow approach to model the hetero-structures formed by arbitrary 2D materials and substrates. The workflow creates, computes, analyzes and stores all relevant simulation parameters and results in a queryable MongoDB database that can be accessed through our API.


Package Description
===================

The 2D-substrate hetero-structure workflow takes a given 2D material, 3D phase (bulk) of the 2D material, and a substrate, relaxes the structures using vdW-corrected DFT and creates hetero-structures subject to user constraints. The workflow analyzes and stores energetic stability information of various 2D hetero-structured materials to predict the feasibility of a substrate stabilizing a meta-stable 2D material.

Hetero2d provides automated routines for the generation of low-lattice mismatched hetero-structures for arbitrary 2D materials and substrate surfaces, the creation of van der Waals corrected density-functional theory (DFT) input files, the submission and monitoring of simulations on computing resources, the post-processing of the key parameters to compute (a) the interface interaction energy of 2D-substrate hetero-structures, (b) the identification of substrate-induced changes in interfacial structure, and (c) charge doping of the 2D material.

Examples
========

To get started using Hetero2d, various tutorials and examples have been created using Jupyter Notebooks. These notebooks demonstrate the basic functionality of Hetero2d to enable users to quickly learn how to use the various modules within this package. These can be found under Hetero2d/examples.

How to cite Hetero2d
===================

If you use Hetero2d in your research, please consider citing the following work:

**add later**

License
=======

Hetero2d is released under the GNU General Public Version 3 License. Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>. The terms of the license can be found in the main directory in this software package under the LICENSE file.

About the Team
==============

Arunima Singh (P.I.) of the Computational Materials Design Lab started Hetero2d in 2020, and is project lead. 

Currently, the main developer for this package is graduate student Tara M. Boland who has developed much of the existing Hetero2d package under the guidance of Arunima Singh. 

See more team details in the Development Team section.

Copyright Policy
================

Hetero2d uses a shared copyright model. Each contributor maintains 
copyright over their contributions to pymatgen. But, it is important 
to note that these contributions are typically only changes to the 
repositories. Thus, the Hetero2d source code, in its entirety is not 
the copyright of any single person or institution. Instead, it is the 
collective copyright of the entire Hetero2d Development Team. If 
individual contributors want to maintain a record of what 
changes/contributions they have specific copyright on, they should 
indicate their copyright in the commit message of the change, when 
they commit the change to one of the pymatgen repositories.

With this in mind, the following banner should be used in any source 
code file to indicate the copyright and license terms::

  # Copyright (c) CMD Lab Development Team.
  # Distributed under the terms of the GNU License.
