#!/bin/bash
# list protocols
# module load Python/3.8.3
source $HOME/workspace/structured-temporal-convolution/venv/bin/activate

line_list="$HOME/workspace/structured-temporal-convolution/client_code/all_hits_t5/saving_timeouts/lines.txt"
larva_dataset="/pasteur/zeus/projets/p02/hecatonchire/alexandre/larva_dataset/point_dynamics_5/t5"

cat $line_list

# go through the protocol list and launch the subprocesses
jobids=""
while read p; do
   jobids+=:$(sbatch -p dbc --parsable $HOME/workspace/structured-temporal-convolution/client_code/all_hits_t5/saving_timeouts/batch_job.sh $(echo $p))
done <$line_list

jobfile="aggregating_job.sh"
echo "#!/bin/sh
#SBATCH --job-name=aggregate
module load Python/3.8.3
source $HOME/workspace/structured-temporal-convolution/venv/bin/activate
python3.8 $HOME/workspace/structured-temporal-convolution/client_code/all_hits_t5/aggregate_bootstraps.py $HOME/workspace/maggotuba_scale_20
" > $jobfile

sbatch -p dbc --dependency afterany$jobids $jobfile

# remove protocol list and jobfile
rm $protocol_list $jobfile
