#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=pdr3_5_grizy_pnp_inpaint
#SBATCH --output=./output/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/env/astro_env/bin/activate
cd ../


#var1=(2)
#var2=(1 1.5 2)
#var3=(1200 1500 2000 2500 3500 5000)

#for ((i=0; i < ${#var1[@]}; i++)) ; do
#    for ((j=0; j < ${#var2[@]}; j++)) ; do
#	for ((k=0; k < ${#var3[@]}; k++)) ; do 
#	    python main.py --config configs/ipe.ini --mc_cho ${var1[$i]} --ipe_default_sigma ${var2[$j]} --pe_max_deg ${var3[$k]} --gt_spectra_cho 0 --start_r 90 --start_c 190
#	done
#    done
#done

for ratio in 0.01 0.1 ; do
    python main.py --config inpaint0.ini --sample_ratio $ratio
done
