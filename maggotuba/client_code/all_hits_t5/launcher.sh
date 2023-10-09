#!/bin/bash
# list protocols
# module load Python/3.8.3
source $HOME/workspace/structured-temporal-convolution/venv/bin/activate

protocol_list="$HOME/workspace/protocol_list.tmp"
larva_dataset="/pasteur/zeus/projets/p02/hecatonchire/alexandre/larva_dataset/point_dynamics_5/t5"

python3.8 $HOME/workspace/structured-temporal-convolution/client_code/all_hits_t5/list_protocols.py --dest $protocol_list

cat $protocol_list

# go through the protocol list and launch the subprocesses
jobids=""
while read p; do
   jobids+=:$(sbatch -p dbc --parsable $HOME/workspace/structured-temporal-convolution/client_code/all_hits_t5/batch_job.sh $(echo $p))
done <$protocol_list

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
