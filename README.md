# Contrastive Pretraining with Masked Imputation for Long Clinical Timeseries Data

## Model Description
This model was developed to handle large time series data from Electronic Health Records (EHR) with a combined pretraining objective that includes both sequence-level and token-level tasks. The model was pretrained using data from the MIMIC-III dataset and externally validated on the eICU dataset.

### Key Features:
- **Triplet-based Embedding:** The model uses a modified triplet embedding, allowing time series data to be treated similarly to tokens in Natural Language Processing (NLP), facilitating efficient handling of long sequences.
- **Combined Objective:** Pretraining is performed using a combination of:
  - **Masked Imputation Task:** The model predicts masked measurement values, enhancing its ability to handle missing data.
  - **Contrastive Learning Task:** A contrastive objective is used for sequence-level pretraining, utilizing a gradient estimator to handle the contrastive term with smaller batch sizes.

## Training Data
The model was pretrained using the **MIMIC-III dataset**, a large-scale, publicly available critical care database. To assess its robustness, the model was externally validated using the **eICU dataset**, simulating its performance in smaller, more diverse clinics.

### Transfer Learning:
The model showed robustness across different clinics, successfully transferring learned representations from a large clinic (MIMIC-III) to smaller clinics (eICU) with diverse patient populations.

## Limitations
- The model’s performance may depend on the quality and completeness of the input time series data.
- External validation on datasets outside of the healthcare domain has not been performed, so its generalizability beyond clinical data remains untested.
