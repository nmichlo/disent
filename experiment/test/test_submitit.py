
# ========================================================================= #
# main                                                                   #
# ========================================================================= #


import submitit
import time

def add(a, b):
    job_env = submitit.JobEnvironment()
    print(f"There are {job_env.num_tasks} in this job")
    print(f"I'm the task #{job_env.local_rank} on the node {job_env.node}")
    print(f"I'm the task #{job_env.global_rank} in the job")
    return a + b

# executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder="log_test")

# set timeout in min, and partition for running the job
executor.update_parameters(
    timeout_min=45,
    slurm_partition="batch",
    slurm_mem=None,  # 64GB by default... um...
    slurm_job_name='test',
)

job = executor.submit(add, 5, 7)  # will compute add(5, 7)
print(job.job_id)  # ID of your job

# you do not need to wait for the jobs to finish
output = job.result()  # waits for completion and returns output
assert output == 12  # 5 + 7 = 12...  your addition was computed in the cluster


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
