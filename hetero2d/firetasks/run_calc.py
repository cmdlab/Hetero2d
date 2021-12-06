# coding: utf-8
# Copyright (c) CMD Lab Development Team.
# Distributed under the terms of the GNU License.

"""
This module defines tasks that support running vasp in various ways.
Under active development for Hetero2d. Currently not in use as the default
behaviour from atomate is sufficient. 
"""

import os
import shlex

from atomate.utils.utils import env_chk, get_logger
from atomate.vasp.config import HALF_KPOINTS_FIRST_RELAX
from custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, AliasingErrorHandler, \
    MeshSymmetryErrorHandler, UnconvergedErrorHandler, MaxForceErrorHandler, \
    FrozenJobErrorHandler, NonConvergingErrorHandler, PositiveEnergyErrorHandler, \
    WalltimeHandler, StdErrHandler, DriftErrorHandler
from custodian.vasp.jobs import VaspJob, VaspNEBJob
from custodian.vasp.validators import VasprunXMLValidator, VaspFilesValidator
from fireworks import explicit_serialize, FiretaskBase, FWAction
from monty.os.path import zpath
from monty.serialization import loadfn

__author__ = 'Tara M. Boland <tboland1@asu.edu>'
__credits__ = 'Shyue Ping Ong <ong.sp>'
__copyright__ = "Copyright 2020, CMD Lab"
__maintainer__ = "Tara M. Boland"
__email__ = 'tboland1@asu.edu'

logger = get_logger(__name__)


@explicit_serialize
class RunVaspCustodian(FiretaskBase):
    """
    Run VASP using custodian "on rails", i.e. in a simple way that supports most common options.

    Args:
        vasp_cmd (str): the name of the full executable for running VASP. Supports env_chk.

    Other Parameters:
        job_type: (str) - choose from "normal" (default), "double_relaxation_run" (two consecutive
            jobs), "full_opt_run" (multiple optimizations), and "neb"
        handler_group: (str or [ErrorHandler]) - group of handlers to use. See handler_groups dict in the code for
            the groups and complete list of handlers in each group. Alternatively, you can
            specify a list of ErrorHandler objects.
        max_force_threshold: (float) - if >0, adds MaxForceErrorHandler. Not recommended for
            nscf runs.
        scratch_dir: (str) - if specified, uses this directory as the root scratch dir.
            Supports env_chk.
        gzip_output: (bool) - gzip output (default=T)
        max_errors: (int) - maximum # of errors to fix before giving up (default=5)
        ediffg: (float) shortcut for setting EDIFFG in special custodian jobs
        auto_npar: (bool) - use auto_npar (default=F). Recommended set to T
            for single-node jobs only. Supports env_chk.
        gamma_vasp_cmd: (str) - cmd for Gamma-optimized VASP compilation.
            Supports env_chk.
        wall_time (int): Total wall time in seconds. Activates WallTimeHandler if set.
        half_kpts_first_relax (bool): Use half the k-points for the first relaxation
    """
    required_params = ["vasp_cmd"]
    optional_params = ["job_type", "handler_group", "max_force_threshold", "scratch_dir",
                       "gzip_output", "max_errors", "ediffg", "auto_npar", "gamma_vasp_cmd",
                       "wall_time", "half_kpts_first_relax"]

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
        job_type = self.get("job_type", "normal")
        scratch_dir = env_chk(self.get("scratch_dir"), fw_spec)
        gzip_output = self.get("gzip_output", True)
        max_errors = self.get("max_errors", CUSTODIAN_MAX_ERRORS)
        auto_npar = env_chk(self.get("auto_npar"), fw_spec, strict=False, default=False)
        gamma_vasp_cmd = env_chk(self.get("gamma_vasp_cmd"), fw_spec, strict=False, default=None)
        if gamma_vasp_cmd:
            gamma_vasp_cmd = shlex.split(gamma_vasp_cmd)

        # construct jobs
        if job_type == "normal":
            jobs = [VaspJob(vasp_cmd, auto_npar=auto_npar, gamma_vasp_cmd=gamma_vasp_cmd)]
        elif job_type == "double_relaxation_run":
            jobs = VaspJob.double_relaxation_run(vasp_cmd, auto_npar=auto_npar,
                                                 ediffg=self.get("ediffg"),
                                                 half_kpts_first_relax=self.get("half_kpts_first_relax",
                                                                                HALF_KPOINTS_FIRST_RELAX))
        elif job_type == "metagga_opt_run":
            jobs = VaspJob.metagga_opt_run(vasp_cmd, auto_npar=auto_npar,
                                           ediffg=self.get("ediffg"),
                                           half_kpts_first_relax=self.get("half_kpts_first_relax",
                                                                          HALF_KPOINTS_FIRST_RELAX))

        elif job_type == "full_opt_run":
            jobs = VaspJob.full_opt_run(vasp_cmd, auto_npar=auto_npar,
                                        ediffg=self.get("ediffg"),
                                        max_steps=9,
                                        half_kpts_first_relax=self.get("half_kpts_first_relax",
                                                                       HALF_KPOINTS_FIRST_RELAX))
        elif job_type == "neb":
            # TODO: @shyuep @HanmeiTang This means that NEB can only be run (i) in reservation mode
            # and (ii) when the queueadapter parameter is overridden and (iii) the queue adapter
            # has a convention for nnodes (with that name). Can't the number of nodes be made a
            # parameter that the user sets differently? e.g., fw_spec["neb_nnodes"] must be set
            # when setting job_type=NEB? Then someone can use this feature in non-reservation
            # mode and without this complication. -computron
            nnodes = int(fw_spec["_queueadapter"]["nnodes"])

            # TODO: @shyuep @HanmeiTang - I am not sure what the code below is doing. It looks like
            # it is trying to override the number of processors. But I tried running the code
            # below after setting "vasp_cmd = 'mpirun -n 16 vasp'" and the code fails.
            # (i) Is this expecting an array vasp_cmd rather than String? If so, that's opposite to
            # the rest of this task's convention and documentation
            # (ii) can we get rid of this hacking in the first place? e.g., allowing the user to
            # separately set the NEB_VASP_CMD as an env_variable and not rewriting the command
            # inside this.
            # -computron

            # Index the tag "-n" or "-np"
            index = [i for i, s in enumerate(vasp_cmd) if '-n' in s]
            ppn = int(vasp_cmd[index[0] + 1])
            vasp_cmd[index[0] + 1] = str(nnodes * ppn)

            # Do the same for gamma_vasp_cmd
            if gamma_vasp_cmd:
                index = [i for i, s in enumerate(gamma_vasp_cmd) if '-n' in s]
                ppn = int(gamma_vasp_cmd[index[0] + 1])
                gamma_vasp_cmd[index[0] + 1] = str(nnodes * ppn)

            jobs = [VaspNEBJob(vasp_cmd, final=False, auto_npar=auto_npar,
                               gamma_vasp_cmd=gamma_vasp_cmd)]
        else:
            raise ValueError("Unsupported job type: {}".format(job_type))

        # construct handlers

        handler_group = self.get("handler_group", "default")
        if isinstance(handler_group, str):
            handlers = handler_groups[handler_group]
        else:
            handlers = handler_group

        if self.get("max_force_threshold"):
            handlers.append(MaxForceErrorHandler(max_force_threshold=self["max_force_threshold"]))

        if self.get("wall_time"):
            handlers.append(WalltimeHandler(wall_time=self["wall_time"]))

        if job_type == "neb":
            validators = []  # CINEB vasprun.xml sometimes incomplete, file structure different
        else:
            validators = [VasprunXMLValidator(), VaspFilesValidator()]

        c = Custodian(handlers, jobs, validators=validators, max_errors=max_errors,
                      scratch_dir=scratch_dir, gzipped_output=gzip_output)

        c.run()

        if os.path.exists(zpath("custodian.json")):
            stored_custodian_data = {"custodian": loadfn(zpath("custodian.json"))}
            return FWAction(stored_data=stored_custodian_data)
