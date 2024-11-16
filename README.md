# Contrastive Pretraining with Masked Imputation for Long Clinical Timeseries Data

This is the official code for "An Efficient Contrastive Unimodal Pretraining Method for EHR Time Series Data" at IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI’24). The paper can be found here: https://arxiv.org/abs/2410.09199

## Model Description
This model was developed to handle large time series data from Electronic Health Records (EHR) with a combined pretraining objective that includes both sequence-level and token-level tasks. The model was pretrained using data from the MIMIC-III dataset and externally validated on the eICU dataset.

### Key Features:
- **Triplet-based Embedding:** The model uses a modified triplet embedding, allowing time series data to be treated similarly to tokens in Natural Language Processing (NLP), facilitating efficient handling of long sequences.
- **Combined Objective:** Pretraining is performed using a combination of:
  - **Masked Imputation Task:** The model predicts masked measurement values, enhancing its ability to handle missing data.
  - **Contrastive Learning Task:** A contrastive objective is used for sequence-level pretraining, utilizing a gradient estimator to handle the contrastive term with smaller batch sizes.

## Training Data
The model was pretrained using the **MIMIC-III dataset**, a large-scale, publicly available critical care database. To assess its robustness, the model was externally validated using the **eICU dataset**.

### Transfer Learning:
The model showed robustness across different clinics, successfully transferring learned representations from (MIMIC-III) to (eICU) with diverse patient populations.

## Limitations
- The model’s performance may depend on the quality and completeness of the input time series data.
- External validation on datasets apart from the eICU dataset has not been performed, so its generalizability beyond eICU remains untested.

# Run Experiments

**Pretrained Models** : https://huggingface.co/Shivesh2001/EHR-CombinedModel-MIMIC

## Pretraining

### Masked Pretraining

`python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --data_dir /ssd-shared/MIMIC_EICU_DATA/data/root --pretraining_method MaskedValuePrediction --batch_size 512 --temperature 0.03 --base_lr 1e-3 --weight_decay 0.00001 --gamma 0.1`

### Contrastive Pretraining

`python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --data_dir /ssd-shared/MIMIC_EICU_DATA/data/root --pretraining_method MaskedValuePrediction --batch_size 512 --temperature 0.03 --base_lr 1e-3 --weight_decay 0.00001 --gamma 0.1`

### Combined Pretraining

`python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --data_dir /ssd-shared/MIMIC_EICU_DATA/data/root --pretraining_method MaskedValuePrediction --batch_size 512 --temperature 0.03 --base_lr 1e-3 --weight_decay 0.00001 --gamma 0.1`

## Loading the Model


```python
from models.model import *

bottleneck_model = BottleNeckModel(embed_dim, num_heads, d_ff, num_variables, N).to(DEVICE)

state_dict = torch.load(file_path, map_location="cpu")

state_dict = {
    key: value for key, value in state_dict.items() if "backbone" in key
}
state_dict = {
    key.replace("backbone.", ""): value for key, value in state_dict.items()
}
bottleneck_model.load_state_dict(state_dict)
return bottleneck_model
```
