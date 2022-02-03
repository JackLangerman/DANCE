#!/bin/bash

#SBATCH --job-name=jupyterTest2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:v100:2
#SBATCH --output=./slurm_info/match2real_cut3.out
module purge

#singularity exec //open3d-0.10.0.sif /bin/bash
singularity exec /scratch/zq415/env/open3d-0.10.0.sif /bin/bash
source /scratch/zq415/env/miniconda3/etc/profile.d/conda.sh
conda activate data_collection

python train.py --dataroot ./datasets/match2real_new/ \
			    --name match2real_cut3 \
			    --CUT_mode CUT \
			    --n_epochs 2 \
			    --n_epochs_decay 4\
			    --save_epoch_freq 1 \
                            --gpu 0,1 \
                            --batch_size 10

