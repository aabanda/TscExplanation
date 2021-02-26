#!/bin/bash

#SBATCH --job-name=rank # Cores
#SBATCH --time=02:00:00 # Partition
#SBATCH --ntasks=4 # Cores
#SBATCH --mem=8G  # Memory
#SBATCH -p medium # Partition
# mail alert at start, end and abortion of execution; All possible options are START, END, ALL
#SBATCH --mail-type=END
# send mail to this address
#SBATCH --mail-user=aabanda@bcamath.org


srun echo "SCRATCH directory: "$SCRATCH_JOB
srun echo "USER directory: "$SCRATCH_USER
srun echo "TMP directory: "$TMP_JOB
srun echo "Node: "$HOSTNAME

source ../anaconda2/etc/profile.d/conda.sh
conda activate py36
module load GCC/8.3.0
module load Python
python warp_intervals_accuracy.py ${1} ${2}

