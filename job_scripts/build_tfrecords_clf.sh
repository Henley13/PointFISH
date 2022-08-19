#!/bin/bash

#SBATCH --export=ALL
#SBATCH -J features
#SBATCH -o /cluster/CBIO/home/aimbert/logs/log-%A.log
#SBATCH -e /cluster/CBIO/home/aimbert/logs/log-%A.err
#SBATCH -t 0-100:00             # Time (DD-HH:MM)
#SBATCH --mem 32000             # Memory per node in MB (0 allocates all the memory)
#SBATCH --ntasks=1              # Number of processes to run (default is 1)
#SBATCH --cpus-per-task=24      # CPU cores per process (default 1)
#SBATCH -p cbio-cpu             # Name of the partition to use

echo 'Running build_tfrecords_clf.sh...'

# directories
data_directory='/cluster/CBIO/data1/data3/aimbert/data/2021_features'
log_directory="/cluster/CBIO/data1/data3/aimbert/output/2021_features/log"

# python script
script='/cluster/CBIO/home/aimbert/2021_features/src/build_tfrecords_clf.py'

python "$script" "$data_directory" "$log_directory"