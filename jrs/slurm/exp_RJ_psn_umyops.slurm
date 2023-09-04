#!/bin/bash
#SBATCH -J  labreg
#SBATCH -p gpu2
#SBATCH -w node01.chess
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 240:00:00
export CUDA_VISIBLE_DEVICES=0
nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES
hostname

export CUDA_VISIBLE_DEVICES=1
nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES
hostname


#~/anaconda3/envs/torch1.7/bin/python ./RJ_psn_myo_3_train.py 2d nnUNetTrainerPSNV8 Task833_ZS_MS all --epochs 100
#testing
~/anaconda3/envs/torch1.7/bin/python ./RJ_psn_myo_4_test.py -tr nnUNetTrainerPSNV8 -i ../outputs/nnunet/raw/nnUNet_raw_data/Task833_ZS_MS/imagesTs -o ../outputs/nnunet/raw/nnUNet_raw_data/Task833_ZS_MS/labelsTs_pre/ -t 833 -m 2d --chk model_best --overwrite_existing --fold all --whichsubnet all
~/anaconda3/envs/torch1.7/bin/python ./RJ_psn_myo_5_test.py -tr nnUNetTrainerPSNV8 -i ../outputs/nnunet/raw/nnUNet_raw_data/Task833_ZS_MS/imagesTs -o ../outputs/nnunet/raw/nnUNet_raw_data/Task833_ZS_MS/labelsTs_pre/ -t 833 -m 2d --chk model_best --overwrite_existing --fold all --whichsubnet all

