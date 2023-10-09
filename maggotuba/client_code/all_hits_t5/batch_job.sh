#!/bin/bash
#
#SBATCH --job-name=p_vals
#
#SBATCH --time=23:59:59
#SBATCH --mem=10000
#SBATCH -p dbc_pmo
#SBATCH --qos dbc
#
#
#SBATCH --array=0-566

module load Python/3.8.3
source $HOME/workspace/structured-temporal-convolution/venv/bin/activate
srun python3.8 -u $HOME/workspace/structured-temporal-convolution/client_code/all_hits_t5/slurmmmd.py \
                       --job_id $SLURM_ARRAY_TASK_ID --total_jobs $SLURM_ARRAY_TASK_COUNT --root $HOME/workspace/maggotuba_scale_20 \
                       -ref $1 -refp $2 --nboot 100000

deactivate
