#!/bin/bash
if [ -z "$1" ]
then
    blr=2e-4
else
    blr=$1
fi


data_path="dataset/physionet/30s_splited"

python submitit_pretrain.py \
--nodes 4 \
--batch_size 128 \
--norm_pix_loss True \
--model maeeg_vit_base_patch16 \
--mask_ratio 0.8 \
--epochs 33 \
--warmup_epochs 3 \
--save_every_epoch 8 \
--blr $blr --weight_decay 0.0001 \
--data_path $data_path \
--nfft 128 \
--hop_length 64 \
--roll_mag_aug True \
--decoder_mode 1 \

