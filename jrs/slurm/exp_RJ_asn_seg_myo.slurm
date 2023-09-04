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

#~/anaconda3/envs/torch1.7/bin/python ./prepare_data.py
#~/anaconda3/envs/torch1.7/bin/python ./RJ_asn_enc_fusion_ComDe_Myo.py --phase train --weight 1 --save_freq 100 --data_source ZS_unaligned --epochs 2000 --ccfeat
#~/anaconda3/envs/torch1.7/bin/python ./RJ_asn_enc_fusion_ComDe_Myo.py --phase test  --weight 1 --save_freq 100 --data_source ZS_unaligned --epochs 2000 --ccfeat --ckpt 2001
#~/anaconda3/envs/torch1.7/bin/python ./RJ_asn_enc_fusion_ComDe_Myo.py --phase metric  --weight 1 --save_freq 100 --data_source ZS_unaligned --epochs 2000 --ccfeat --ckpt 2001

