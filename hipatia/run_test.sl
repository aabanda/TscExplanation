#!/bin/bash
#SBATCH --job-name=loop # Cores
#SBATCH --time=00:10:00 # Partition
#SBATCH --ntasks=4 # Cores
#SBATCH --mem=1G  # Memory
#SBATCH -p short # Partition
# mail alert at start, end and abortion of execution; All possible options are START, END, ALL
#SBATCH --mail-type=END
# send mail to this address
#SBATCH --mail-user=aabanda@bcamath.org



declare -a test_ind=(1 2 3 4 5)


# get length of an array
testlength=${#test_ind[@]}


# use for loop to read all values and indexes

for (( i=1; i<${testlength}+1; i++ ));
do	
	sbatch -J t${test_ind[$i-1]} run_weights.sl ${test_ind[$i-1]}
done




