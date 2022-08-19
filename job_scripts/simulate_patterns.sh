#!/bin/bash

#SBATCH --export=ALL
#SBATCH -J features
#SBATCH -o /cluster/CBIO/home/aimbert/logs/log-%A_%a.log
#SBATCH -e /cluster/CBIO/home/aimbert/logs/log-%A_%a.err
#SBATCH --array=1-8%8           # Number of job arrays to launch
#SBATCH -t 0-100:00             # Time (DD-HH:MM)
#SBATCH --mem 10000             # Memory per node in MB (0 allocates all the memory)
#SBATCH --ntasks=1              # Number of processes to run (default is 1)
#SBATCH --cpus-per-task=6       # CPU cores per process (default 1)
#SBATCH -p cbio-cpu             # Name of the partition to use

echo 'Running simulate_patterns.sh...'

echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_JOBID: " $SLURM_JOBID

# directories
data_directory='/cluster/CBIO/data1/data3/aimbert/data/2021_features'
log_directory="/cluster/CBIO/data1/data3/aimbert/output/2021_features/log"

# python script
script='/cluster/CBIO/home/aimbert/2021_features/src/simulate_patterns.py'

# parse parameters
file="$(pwd)/patterns.txt"
pattern=$(grep "^$SLURM_ARRAY_TASK_ID " "$file" | cut -d ' ' -f 2)

python "$script" "$data_directory" "$pattern" "$log_directory"