#!/bin/bash
#SBATCH -J asclepius_fake
#SBATCH --account=ml4science
#SBATCH --partition=a100_normal_q 
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 
#SBATCH --time=0-2:00:00  
#SBATCH --gres=gpu:1
#SBATCH -o ./SLURM/slurm-%j.out

module reset
module load Anaconda3/2020.11
module load GCC/11.2.0
source activate med
module reset
source activate med

python /home/amartya/medhalt/bio_llm.py -m asclepius -b 12

exit;