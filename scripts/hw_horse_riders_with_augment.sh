#!/bin/bash

rm -r cache_files
export FIX_TORCH_ERROR=1
# export JT_SYNC=1
# export trace_py_var=3
# export use_parallel_op_compiler=0
export MKL_NUM_THREADS=1 # slurm 不兼容
# export cc_path="g++"
python train.py \
--name hw_horse_riders_with_augment \
--dataroot_sketch ./data/sketch/photosketch/horse_riders \
--dataroot_image ./data/image/horse \
--l_image 0.7 \
--eval_dir ./eval/horse_riders \
--max_iter 100000 --diffaug_policy translation \
--g_pretrained ./pretrained/stylegan2-horse/netG.pth \
# --d_pretrained ./pretrained/stylegan2-horse/netD.pth \
