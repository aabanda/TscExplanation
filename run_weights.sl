#!/bin/bash

#SBATCH --job-name=EvCbfShSt# Cores
#SBATCH --time=720:00:00 # Partition
#SBATCH --ntasks=1 # Cores
#SBATCH --mem=16G  # Memory
#SBATCH -p xlarge # Partition
# mail alert at start, end and abortion of execution; All possible options are START, END, ALL
#SBATCH --mail-type=END
# send mail to this address
#SBATCH --mail-user=aabanda@bcamath.org


srun echo "SCRATCH directory: "$SCRATCH_JOB
srun echo "USER directory: "$SCRATCH_USER
srun echo "TMP directory: "$TMP_JOB
srun echo "Node: "$HOSTNAME

source ../anaconda2/etc/profile.d/conda.sh
#conda activate py36
conda activate sktime-dev
#module load GCC/8.3.0
#module load Python
#python compute_neig_labels_st.py ${1} st CBF scale
#python evaluation_hipatia.py CBF noise 9 st
python evaluation_shift_hipatia.py CBF shif 1 st
