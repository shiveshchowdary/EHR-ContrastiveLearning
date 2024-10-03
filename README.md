# code_base_mimic

**drive link for saved models and results**

link : https://drive.google.com/drive/folders/1cpthrg0DPceVPsbGb2949EE6Ru2XA6sX?usp=sharing

# Run Experiments

## Pretraining

### Masked Pretraining

`python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --data_dir /ssd-shared/MIMIC_EICU_DATA/data/root --pretraining_method MaskedValuePrediction --batch_size 512 --temperature 0.03 --base_lr 1e-3 --weight_decay 0.00001 --gamma 0.1`

### Contrastive Pretraining

`python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --data_dir /ssd-shared/MIMIC_EICU_DATA/data/root --pretraining_method MaskedValuePrediction --batch_size 512 --temperature 0.03 --base_lr 1e-3 --weight_decay 0.00001 --gamma 0.1`

### Combined Pretraining

`python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --data_dir /ssd-shared/MIMIC_EICU_DATA/data/root --pretraining_method MaskedValuePrediction --batch_size 512 --temperature 0.03 --base_lr 1e-3 --weight_decay 0.00001 --gamma 0.1`