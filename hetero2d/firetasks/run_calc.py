# coding: utf-8
# Copyright (c) CMD Lab Development Team.
# Distributed under the terms of the GNU License.

"""
This module defines tasks that support running vasp jobs for a specific purpose.
Under active development for Hetero2d. Simple modifications to atomate functions
to modify the default behaviour. 
"""

import os, re, shlex, numpy as np

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, VaspInput

from custodian import Custodian
from custodian.vasp.jobs import VaspJob
from custodian.vasp.validators import VasprunXMLValidator, VaspFilesValidator
from custodian.vasp.handlers import VaspErrorHandler, AliasingErrorHandler, \
    MeshSymmetryErrorHandler, UnconvergedErrorHandler, MaxForceErrorHandler, \
    FrozenJobErrorHandler, NonConvergingErrorHandler, PositiveEnergyErrorHandler, \
    WalltimeHandler, StdErrHandler, DriftErrorHandler

from fireworks import explicit_serialize, FiretaskBase, FWAction

from atomate.utils.utils import env_chk, get_logger

from monty.os.path import zpath
from monty.serialization import loadfn


__author__ = 'Tara M. Boland <tboland1@asu.edu>'
__credits__ = 'Shyue Ping Ong <ong.sp>'
__copyright__ = "Copyright 2020, CMD Lab"
__maintainer__ = "Tara M. Boland"
__email__ = 'tboland1@asu.edu'

logger = get_logger(__name__)


class ElectronicJob(VaspJob):
    """
    A subclass of VaspJob created to run a double kpoint run for calculating bader
    and DOS. Just runs whatever is in the directory. But conceivably can be a complex 
    processing of inputs etc. with initialization.
    """

    def __init__(self, vasp_cmd, output_file="vasp.out", stderr_file="std_err.txt",
        suffix="", final=True, backup=True, auto_npar=False, auto_gamma=True,
        settings_override=None, gamma_vasp_cmd=None, copy_magmom=False, auto_continue=False):
        """
        This constructor is necessarily complex due to the need for
        flexibility. For standard kinds of runs, it's often better to use one
        of the static constructors. The defaults are usually fine too.
        Args:
            vasp_cmd (str): Command to run vasp as a list of args. For example,
                if you are using mpirun, it can be something like
                ["mpirun", "pvasp.5.2.11"]
            output_file (str): Name of file to direct standard out to.
                Defaults to "vasp.out".
            stderr_file (str): Name of file to direct standard error to.
                Defaults to "std_err.txt".
            suffix (str): A suffix to be appended to the final output. E.g.,
                to rename all VASP output from say vasp.out to
                vasp.out.relax1, provide ".relax1" as the suffix.
            final (bool): Indicating whether this is the final vasp job in a
                series. Defaults to True.
            backup (bool): Whether to backup the initial input files. If True,
                the INCAR, KPOINTS, POSCAR and POTCAR will be copied with a
                ".orig" appended. Defaults to True.
            auto_npar (bool): Whether to automatically tune NPAR to be sqrt(
                number of cores) as recommended by VASP for DFT calculations.
                Generally, this results in significant speedups. Defaults to
                True. Set to False for HF, GW and RPA calculations.
            auto_gamma (bool): Whether to automatically check if run is a
                Gamma 1x1x1 run, and whether a Gamma optimized version of
                VASP exists with ".gamma" appended to the name of the VASP
                executable (typical setup in many systems). If so, run the
                gamma optimized version of VASP instead of regular VASP. You
                can also specify the gamma vasp command using the
                gamma_vasp_cmd argument if the command is named differently.
            settings_override ([dict]): An ansible style list of dict to
                override changes. For example, to set ISTART=1 for subsequent
                runs and to copy the CONTCAR to the POSCAR, you will provide::
                    [{"dict": "INCAR", "action": {"_set": {"ISTART": 1}}},
                     {"file": "CONTCAR",
                      "action": {"_file_copy": {"dest": "POSCAR"}}}]
            gamma_vasp_cmd (str): Command for gamma vasp version when
                auto_gamma is True. Should follow the list style of
                subprocess. Defaults to None, which means ".gamma" is added
                to the last argument of the standard vasp_cmd.
            copy_magmom (bool): Whether to copy the final magmom from the
                OUTCAR to the next INCAR. Useful for multi-relaxation runs
                where the CHGCAR and WAVECAR are sometimes deleted (due to
                changes in fft grid, etc.). Only applies to non-final runs.
            auto_continue (bool): Whether to automatically continue a run
                if a STOPCAR is present. This is very useful if using the
                wall-time handler which will write a read-only STOPCAR to
                prevent VASP from deleting it once it finishes
        """
        self.vasp_cmd = vasp_cmd
        self.output_file = output_file
        self.stderr_file = stderr_file
        self.final = final
        self.backup = backup
        self.suffix = suffix
        self.settings_override = settings_override
        self.auto_npar = auto_npar
        self.auto_gamma = auto_gamma
        self.gamma_vasp_cmd = gamma_vasp_cmd
        self.copy_magmom = copy_magmom
        self.auto_continue = auto_continue
        super().__init__(vasp_cmd, output_file, stderr_file, final, backup, suffix, settings_override,
                auto_npar, auto_gamma, gamma_vasp_cmd, copy_magmom, auto_continue)

    @classmethod
    def double_kpoints_run(cls, vasp_cmd, auto_npar=True, half_kpts_first=True,
        auto_continue=False):
        """
        Returns a list of two jobs the first is to obtain the CHGCAR and WAVECAR
        with a low kp number using ICHARG = 1 with increased bands, charge density 
        grid, and nedos and the second calcaluation with ICHARG = 11 with increased
        kp grid.

        Args:
            vasp_cmd (str): Command to run vasp as a list of args. For example,
                if you are using mpirun, it can be something like
                ["mpirun", "pvasp.5.2.11"]
            auto_npar (bool): Whether to automatically tune NPAR to be sqrt(
                number of cores) as recommended by VASP for DFT calculations.
                Generally, this results in significant speedups. Defaults to
                True. 
            half_kpts_first (bool): Whether to halve the kpoint grid for the 
                first run to obtain convergence chgcar file. Speeds up calculation 
                time considerably for bader analysis. Defaults to True.

        Returns:
            List of two jobs corresponding to an AFLOW style run.
        """
        incar_orig = Incar.from_file('INCAR')
        incar1 = {"ICHARG": 1, "ISYM": 2, "LAECHG": False, "NEDOS": 301} 
        incar2 = {key: incar_orig.get(key) for key in incar1.keys() if key in incar_orig.keys()}
        # set npar looking for ntasks from os.environ: nslots and slurm_cpus_per_task
        # fails on all slurm systems
        slurm_keys = list(os.environ.keys())
        ncore_keys = [ [key for key in slurm_keys if re.search(match,key) ] 
                                for match in ['TASKS','NSLOTS']]
        ncore_keys = list(np.unique(sum(ncore_keys, [])))
        for key in ncore_keys:
            if re.search('NTASKS',key):
                ncores = os.environ[key]
                break
            elif re.search('NSLOTS',key):
                ncores = os.environ[key] 
                break
        if ncores:
            for npar in range(int(math.sqrt(ncores)),ncores):
                if ncores % npar == 0:
                    incar1['NPAR'] = npar
                    incar2['NPAR'] = npar
        # set the override settings
        settings_overide_1 = [{"dict": "INCAR", "action": {"_set": incar1}}]
        settings_overide_2 = [{"dict": "INCAR", "action": {"_set": incar2}}]
        if half_kpts_first and os.path.exists("KPOINTS") and os.path.exists("POSCAR"):
            kpts = Kpoints.from_file("KPOINTS")
            orig_kpts_dict = kpts.as_dict()
            # lattice vectors with length < 8 will get >1 KPOINT
            kpts.kpts = np.round(np.maximum(np.array(kpts.kpts) / 2, 1)).astype(int).tolist()
            low_kpts_dict = kpts.as_dict()
            settings_overide_1.append({"dict": "KPOINTS", "action": {"_set": low_kpts_dict}})
            settings_overide_2.append({"dict": "KPOINTS", "action": {"_set": orig_kpts_dict}})

        return [
            VaspJob(
                vasp_cmd,
                final=False,
                suffix=".kpoints1",
                auto_npar=auto_npar,
                auto_continue=auto_continue,
                settings_override=settings_overide_1,
            ),
            VaspJob(
                vasp_cmd,
                final=True,
                backup=False,
                suffix="",
                auto_npar=auto_npar,
                auto_continue=auto_continue,
                settings_override=settings_overide_2,
            ),
        ]

@explicit_serialize
class RunElectronicCustodian(FiretaskBase):
    """
    Modified version of RunVaspCustodian that runs VASP using custodian "on rails" but
    taylored to run double runs for computing electronic structure properties. Convergences
    the chgcar using small kp grid with bader level ngif grid density then an increased grid
    density.

    Args:
        vasp_cmd (str): the name of the full executable for running VASP. Supports env_chk.

    Other Parameters:
        job_type: (str) - "double_kpoints_run" (two consecutive jobs) - default.
        handler_group: (str or [ErrorHandler]) - group of handlers to use. See handler_groups dict 
            in the code for the groups and complete list of handlers in each group. Alternatively, 
            you can specify a list of ErrorHandler objects.
        scratch_dir: (str) - if specified, uses this directory as the root scratch dir.
            Supports env_chk.
        gzip_output: (bool) - gzip output (default=T)
        max_errors: (int) - maximum # of errors to fix before giving up (default=5)
        auto_npar: (bool) - use auto_npar (default=F). Recommended set to T
            for single-node jobs only. Supports env_chk.
        gamma_vasp_cmd: (str) - cmd for Gamma-optimized VASP compilation.
            Supports env_chk.
        wall_time (int): Total wall time in seconds. Activates WallTimeHandler if set.
        half_kpts_first (bool): Use half the k-points first to converge charge density.
    """
    required_params = ["vasp_cmd"]
    optional_params = ["job_type", "handler_group", "scratch_dir", "gzip_output", 
                       "max_errors", "auto_npar", "gamma_vasp_cmd",
                       "wall_time", "half_kpts_first"]

    def run_task(self, fw_spec):

        handler_groups = {
            "default": [VaspErrorHandler(), MeshSymmetryErrorHandler(), UnconvergedErrorHandler(),
                        NonConvergingErrorHandler(),
                        PositiveEnergyErrorHandler(), FrozenJobErrorHandler(), StdErrHandler()],
            "strict": [VaspErrorHandler(), MeshSymmetryErrorHandler(), UnconvergedErrorHandler(),
                       NonConvergingErrorHandler(),
                       PositiveEnergyErrorHandler(), FrozenJobErrorHandler(),
                       StdErrHandler(), AliasingErrorHandler(), DriftErrorHandler()],
            "md": [VaspErrorHandler(), NonConvergingErrorHandler()],
            "td": [VaspErrorHandler(), MeshSymmetryErrorHandler(), UnconvergedErrorHandler(),
                   NonConvergingErrorHandler(),
                   PositiveEnergyErrorHandler(), FrozenJobErrorHandler(), StdErrHandler()],
            "no_handler": []
        }

        vasp_cmd = env_chk(self["vasp_cmd"], fw_spec)

        if isinstance(vasp_cmd, str):
            vasp_cmd = os.path.expandvars(vasp_cmd)
            vasp_cmd = shlex.split(vasp_cmd)

        # initialize variables
        job_type = self.get("job_type", "double_kpoints_run")
        scratch_dir = env_chk(self.get("scratch_dir"), fw_spec)
        gzip_output = self.get("gzip_output", True)
        max_errors = self.get("max_errors", 5)
        auto_npar = env_chk(self.get("auto_npar"), fw_spec, strict=False, default=False)
        gamma_vasp_cmd = env_chk(self.get("gamma_vasp_cmd"), fw_spec, strict=False, default=None)
        if gamma_vasp_cmd:
            gamma_vasp_cmd = shlex.split(gamma_vasp_cmd)

        # construct jobs
        if job_type == "double_kpoints_run":
            jobs = ElectronicJob.double_kpoints_run(vasp_cmd, auto_npar=auto_npar,
                                              half_kpts_first=self.get("half_kpts_first", True))
        else:
            raise ValueError("Unsupported job type: {}".format(job_type))

        # construct handlers
        handler_group = self.get("handler_group", "default")
        if isinstance(handler_group, str):
            handlers = handler_groups[handler_group]
        else:
            handlers = handler_group

        if self.get("wall_time"):
            handlers.append(WalltimeHandler(wall_time=self["wall_time"]))
        validators = [VasprunXMLValidator(), VaspFilesValidator()]
        c = Custodian(handlers, jobs, validators=validators, max_errors=max_errors,
                      scratch_dir=scratch_dir, gzipped_output=gzip_output)
        c.run()

        if os.path.exists(zpath("custodian.json")):
            stored_custodian_data = {"custodian": loadfn(zpath("custodian.json"))}
            return FWAction(stored_data=stored_custodian_data)

