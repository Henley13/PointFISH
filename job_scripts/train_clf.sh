#!/bin/bash

#SBATCH --export=ALL
#SBATCH -J features
#SBATCH -o /cluster/CBIO/home/aimbert/logs/log-%A_%a.log
#SBATCH -e /cluster/CBIO/home/aimbert/logs/log-%A_%a.err
#SBATCH --array=1-4%4          # Number of job arrays to launch
#SBATCH --gres=gpu:P100:1       # Number of GPUs per node
#SBATCH -t 0-100:00             # Time (DD-HH:MM)
#SBATCH --mem 28000             # Memory per node in MB (0 allocates all the memory)
#SBATCH --ntasks=1              # Number of processes to run (default is 1)
#SBATCH --cpus-per-task=6       # CPU cores per process (default 1)
#SBATCH -p cbio-gpu             # Name of the partition to use

echo 'Running train_clf.sh...'

echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_JOBID: " $SLURM_JOBID
echo

echo $CUDA_VISIBLE_DEVICES
echo
nvcc --version
echo
nvidia-smi
echo

# directories
data_directory='/cluster/CBIO/home/aimbert/data_ssd/2021_features'
output_directory='/cluster/CBIO/data1/data3/aimbert/output/2021_features'

# python script
script='/cluster/CBIO/home/aimbert/2021_features/src/train_clf.py'

# parse parameters
file="$(pwd)/training_clf.txt"
add_cluster=$(grep "^$SLURM_ARRAY_TASK_ID " "$file" | cut -d ' ' -f 2)
add_morphology=$(grep "^$SLURM_ARRAY_TASK_ID " "$file" | cut -d ' ' -f 3)
add_distance=$(grep "^$SLURM_ARRAY_TASK_ID " "$file" | cut -d ' ' -f 4)
input_dimension=$(grep "^$SLURM_ARRAY_TASK_ID " "$file" | cut -d ' ' -f 5)
base_model_name=$(grep "^$SLURM_ARRAY_TASK_ID " "$file" | cut -d ' ' -f 6)
inputs_alignment=$(grep "^$SLURM_ARRAY_TASK_ID " "$file" | cut -d ' ' -f 7)
features_alignment=$(grep "^$SLURM_ARRAY_TASK_ID " "$file" | cut -d ' ' -f 8)
k=$(grep "^$SLURM_ARRAY_TASK_ID " "$file" | cut -d ' ' -f 9)
nb_head=$(grep "^$SLURM_ARRAY_TASK_ID " "$file" | cut -d ' ' -f 10)
latent_dimension=$(grep "^$SLURM_ARRAY_TASK_ID " "$file" | cut -d ' ' -f 11)

python "$script" "$data_directory" "$output_directory" \
       "$add_cluster" "$add_morphology" "$add_distance" "$input_dimension" \
       "$base_model_name" "$inputs_alignment" "$features_alignment" \
       "$k" "$nb_head" "$latent_dimension" "$SLURM_ARRAY_TASK_ID"