#!/bin/bash
#SBATCH --array=8,9
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=pdr3_pnp_512_10_3bandinpaint_try
#SBATCH --output=./output/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/env/astro_env/bin/activate
cd ../


python main.py --config configs/inpaint.ini --sample_ratio_cho $SLURM_ARRAY_TASK_ID
