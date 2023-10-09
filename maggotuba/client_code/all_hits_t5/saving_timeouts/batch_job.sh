#!/bin/bash
#
#SBATCH --job-name=p_vals
#
#SBATCH --time=47:59:59
#SBATCH --mem=10000
#SBATCH -p dbc_pmo
#SBATCH --qos dbc
#SBATCH --array 0-99



module load Python/3.8.3
source $HOME/workspace/structured-temporal-convolution/venv/bin/activate
srun python3.8  -u $HOME/workspace/structured-temporal-convolution/client_code/all_hits_t5/saving_timeouts/slurmmmd_timeout.py \
                       $1 $2 $SLURM_ARRAY_TASK_COUNT $SLURM_ARRAY_TASK_ID --root $HOME/workspace/maggotuba_scale_20 --nboot 100000

deactivate
