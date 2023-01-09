#!/bin/bash

fold=('A' 'B' 'C' 'D' 'E')
#bands=('ms' 'merge' )
#ckpts=('' '' '' '' '')

for f in "${fold[@]}"; do
    #for f in "${fold[@]}"; do
    init="/network/scratch/a/amna.elmustafa/rural-urban-Modelbias/outputs/dhs_ooc/ablation_study/DHS_OOC_build_resnet_water_qnt_b128_fce-05_conve-05_lr001_crop224_fold"
    fo=$f
    con="${init}${fo}"
    #con+="_ms_no_attn_355P100_b32_fc01_conv01_lr0001_crop100/best.ckpt"
    #con+="_NL_no_attn_355P100_b32_fce-05_conve-05_lr0001_crop100/best.ckpt"
    con+="/best.ckpt"
    #con+="_building_no_attn_355P224_b32_fce-05_conve-05_lr0001_crop224/best.ckpt"
    #con+="_NL_no_attn_355P224_b32_fce-05_conve-05_lr0001_crop224/best.ckpt"
    sbatch ./train_.sh --fold=$f --init_ckpt_dir=$con
    #done
done


