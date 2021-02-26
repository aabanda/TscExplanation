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


#declare -a threshold=(5 10 20 30)
declare -a threshold=(5)

declare -a cutoff=(0.5)
#declare -a cutoff=(0.5 0.6 0.7 0.8 0.9)

# get length of an array
threslength=${#threshold[@]}
cutlength=${#cutoff[@]}

# use for loop to read all values and indexes
for (( i=1; i<${threslength}+1; i++ ));
do
	for (( j=1; j<${cutlength}+1; j++ ));
	do	
  		sbatch -J ${threshold[$i-1]}${cutoff[$j-1]} run_python.sl ${threshold[$i-1]} ${cutoff[$j-1]} 
	done
done



