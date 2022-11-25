#!/bin/bash --login
###
# job name
#SBATCH --job-name=PipeFlow_%a
# job stderr file
#SBATCH --error=PipeFlow.err.%a
# maximum job time in D-HH:MM
#SBATCH --time=0-00:30
# maximum memory megabytes
#SBATCH --mem-per-cpu=1
# run a two tasks
#SBATCH --ntasks=1
# run the tasks across two nodes; i.e. one per node
#SBATCH --nodes=1
# specify our current project
#SBATCH --account=scw1706
#SBATCH --mail-user=2115589@swansea.ac.uk
#SBATCH --mail-type=END
###


module load anaconda/2020.07
source activate py38wlMacro
python3 worklineMainV1HPC.py 1 1 0.001 4
# python3 ./worklineMacro/worklineMainV1HPC.py 1 1 0.001 4
