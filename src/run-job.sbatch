#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
#SBATCH --output=result.txt

#access to HPC-managed libraries
module purge
module load python3/intel/3.5.3
module load torchvision/python3.5/0.1.9
module load pytorch/python3.6/0.3.0_4
module load numpy/python3.5/intel/1.13.1
module load torch/intel/20170104
module load scipy/intel/0.19.1
#run 
python3 train.py
exit