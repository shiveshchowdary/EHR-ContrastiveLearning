# #!/bin/bash


mkdir -p outputs/linear_eval_synthetic_both

    
CUDA_VISIBLE_DEVICES=6 python main_downstream.py --method Combined --load_epoch 399 --data_dir /ssd-shared/MIMIC_EICU_DATA/data_17/phenotyping --K 1 --epochs 20 --head_lr 0.03 --dataset phenotype --patience 20 --batch_size 16 --linear_eval --seed 42 > outputs/linear_eval_synthetic_both/phenotyping_linear_eval_seed_42_epoch_24.txt &
    
CUDA_VISIBLE_DEVICES=6 python main_downstream.py --method Combined --load_epoch 399 --data_dir /ssd-shared/MIMIC_EICU_DATA/data_17/phenotyping --K 1 --epochs 20 --head_lr 0.03 --dataset phenotype --patience 20 --batch_size 16 --linear_eval --seed 50 > outputs/linear_eval_synthetic_both/phenotyping_linear_eval_seed_50_epoch_24.txt &
    
CUDA_VISIBLE_DEVICES=6 python main_downstream.py --method Combined --load_epoch 399 --data_dir /ssd-shared/MIMIC_EICU_DATA/data_17/phenotyping --K 1 --epochs 20 --head_lr 0.03 --dataset phenotype --patience 20 --batch_size 16 --linear_eval --seed 60 > outputs/linear_eval_synthetic_both/phenotyping_linear_eval_seed_60_epoch_24.txt &
    
CUDA_VISIBLE_DEVICES=6 python main_downstream.py --method Combined --load_epoch 399 --data_dir /ssd-shared/MIMIC_EICU_DATA/data_17/phenotyping --K 1 --epochs 20 --head_lr 0.03 --dataset phenotype --patience 20 --batch_size 16 --linear_eval --seed 70 > outputs/linear_eval_synthetic_both/phenotyping_linear_eval_seed_70_epoch_24.txt &
    
CUDA_VISIBLE_DEVICES=6 python main_downstream.py --method Combined --load_epoch 399 --data_dir /ssd-shared/MIMIC_EICU_DATA/data_17/phenotyping --K 1 --epochs 20 --head_lr 0.03 --dataset phenotype --patience 20 --batch_size 16 --linear_eval --seed 80 > outputs/linear_eval_synthetic_both/phenotyping_linear_eval_seed_80_epoch_24.txt &

CUDA_VISIBLE_DEVICES=5 python main_downstream.py --method Combined --load_epoch 399 --data_dir /ssd-shared/MIMIC_EICU_DATA/data_17/in-hospital-mortality --K 1 --epochs 20 --head_lr 0.01 --dataset ihm --patience 20 --batch_size 16 --linear_eval --seed 42 > outputs/linear_eval_synthetic_both/ihm_linear_eval_seed_42_epoch_24.txt &

CUDA_VISIBLE_DEVICES=5 python main_downstream.py --method Combined --load_epoch 399 --data_dir /ssd-shared/MIMIC_EICU_DATA/data_17/in-hospital-mortality --K 1 --epochs 20 --head_lr 0.01 --dataset ihm --patience 20 --batch_size 16 --linear_eval --seed 50 > outputs/linear_eval_synthetic_both/ihm_linear_eval_seed_50_epoch_24.txt &

CUDA_VISIBLE_DEVICES=5 python main_downstream.py --method Combined --load_epoch 399 --data_dir /ssd-shared/MIMIC_EICU_DATA/data_17/in-hospital-mortality --K 1 --epochs 20 --head_lr 0.01 --dataset ihm --patience 20 --batch_size 16 --linear_eval --seed 60 > outputs/linear_eval_synthetic_both/ihm_linear_eval_seed_60_epoch_24.txt &

CUDA_VISIBLE_DEVICES=5 python main_downstream.py --method Combined --load_epoch 399 --data_dir /ssd-shared/MIMIC_EICU_DATA/data_17/in-hospital-mortality --K 1 --epochs 20 --head_lr 0.01 --dataset ihm --patience 20 --batch_size 16 --linear_eval --seed 70 > outputs/linear_eval_synthetic_both/ihm_linear_eval_seed_70_epoch_24.txt &

CUDA_VISIBLE_DEVICES=5 python main_downstream.py --method Combined --load_epoch 399 --data_dir /ssd-shared/MIMIC_EICU_DATA/data_17/in-hospital-mortality --K 1 --epochs 20 --head_lr 0.01 --dataset ihm --patience 20 --batch_size 16 --linear_eval --seed 80 > outputs/linear_eval_synthetic_both/ihm_linear_eval_seed_80_epoch_24.txt
