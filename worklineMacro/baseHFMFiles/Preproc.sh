#!/bin/bash --login
###
# job name
#SBATCH --job-name=PipeFlow_%a
# job stderr file
#SBATCH --error=PipeFlow.err.%a
# maximum job time in D-HH:MM
#SBATCH --time=0-00:01
# maximum memory megabytes
#SBATCH --mem-per-cpu=4000
# run a two tasks
#SBATCH --ntasks=32
# run the tasks across two nodes; i.e. one per node
#SBATCH --nodes=1
# specify our current project
#SBATCH --account=scw1706
#SBATCH --mail-user=2115589@swansea.ac.uk
#SBATCH --mail-type=END
###

cd /lustrehome/home/s.2115589/caseDatabase/
module load compiler/intel/2018/4\ mpi/intel/2018/4\ boost/1.69.0\ cmake/3.14.3\ fftw/3.3.8
source /apps/local/materials/OpenFOAM/v2106/el7/AVX512/intel-2018/intel-2018/OpenFOAM-v2106/etc/bashrc
./Preproc
