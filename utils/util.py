import numpy as np
import pandas as pd
import json
def calculate_test_scores(file_path, task):
    if task in ["ihm", "decompensation"]:
        
        with open(f"{file_path}", 'r') as f:
            res_dict = json.load(f)
            
        auc_roc_median = np.mean(res_dict['auc_roc'])
        auc_pr_median = np.mean(res_dict['auc_pr'])
        
        print("AUC ROC Median:", auc_roc_median)
        print("AUC ROC 95% Confidence Intervals:", np.percentile(res_dict['auc_roc'], 5), np.percentile(res_dict['auc_roc'], 95))

        print("AUC PR Median:", auc_pr_median)
        print("AUC PR 95% Confidence Intervals:", np.percentile(res_dict['auc_pr'], 5), np.percentile(res_dict['auc_pr'], 95))
        print("............................")
        
        return
        
            
    elif task == "phenotype":
        with open(f"{file_path}", 'r') as f:
            res_dict = json.load(f)
        auc_roc_median = np.mean(res_dict['macro'])

        auc_pr_median = np.mean(res_dict['micro'])

        print("AUC ROC Macro Median:", auc_roc_median)
        print("AUC ROC Macro 95% Confidence Intervals:", np.percentile(res_dict['macro'], 5), np.percentile(res_dict['macro'], 95))

        print("AUC ROC Micro Median:", auc_pr_median)
        print("AUC ROC Micro 95% Confidence Intervals:", np.percentile(res_dict['micro'], 5), np.percentile(res_dict['micro'], 95))
        print("............................")
        return
    
    elif task == "lengthofstay":
        with open(f"{file_path}", 'r') as f:
            res_dict = json.load(f)
        auc_roc_median = np.mean(res_dict['kappa'])

        auc_pr_median = np.mean(res_dict['mad'])

        print("kappa Median:", auc_roc_median)
        print("kappa 95% Confidence Intervals:", np.percentile(res_dict['kappa'], 5), np.percentile(res_dict['kappa'], 95))

        print("MAD Median:", auc_pr_median)
        print("MAD 95% Confidence Intervals:", np.percentile(res_dict['mad'], 5), np.percentile(res_dict['mad'], 95))
        print("............................")
        return
