import logging
import os
import submitit

log = logging.getLogger(__name__)

# ========================================================================= #
# slurm                                                                     #
# ========================================================================= #


def slurm_run(
        func,
        join=True,
        logs_dir=os.path.join(os.getcwd(), 'logs/submitit'),
        print_info=True,
        **slurm_kwargs
):
    slurm_kwargs.setdefault('timeout_min', 3600)
    slurm_kwargs.setdefault('slurm_partition', 'batch')
    slurm_kwargs.setdefault('slurm_mem', None)
    slurm_kwargs.setdefault('slurm_job_name', f'submitit-{func.__name__}')

    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder=logs_dir)
    executor.update_parameters(**slurm_kwargs)

    # run job
    job = executor.submit(func)

    # print job info
    if print_info:
        gry, red, grn, rst = '\033[90m', '\033[91m', '\033[92m', '\033[0m'
        log.info(f'Started Job: {job.job_id}')
        log.info(f'Watch progress with: {gry}$ {grn}tail -f {os.path.join(logs_dir, f"{job.job_id}_{job.task_id}_log.out")}{rst}')
        log.info(f'Watch errors with: {gry}$ {red}tail -f {os.path.join(logs_dir, f"{job.job_id}_{job.task_id}_log.err")}{rst}')

    if join:
        return job.result()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
