#!/bin/bash
#
#SBATCH --job-name=p_vals
#
#SBATCH --ntasks=1
#SBATCH --time=23:59:59
#SBATCH --mem=10000
#SBATCH --cpus-per-task 4
#
#
#SBATCH --array=0-564

source $HOME/workspace/structured-temporal-convolution/venv/bin/activate
module load Python/3.8.3
srun -p dbc maggotuba mmd find_hits_slurm --job_id $SLURM_ARRAY_TASK_ID --total_jobs $SLURM_ARRAY_TASK_COUNT --root $HOME/workspace/maggotuba_scale_20

deactivate
