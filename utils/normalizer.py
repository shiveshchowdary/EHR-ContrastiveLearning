import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
from tqdm import tqdm
import pickle

# pd.set_option('future.no_silent_downcasting',True)


class Normalizer:
    def __init__(self, data_dir_list):
        self.data_list = data_dir_list
        self.categorical_variables = [
            "Glascow coma scale eye opening",
            "Glascow coma scale motor response",
            "Glascow coma scale verbal response",
        ]
        self.mean_var_dict = self.get_mean_var()

    def get_mean_var(self):
        var_dict = {}
        for sample_path in tqdm(self.data_list):
            df = pd.read_csv(sample_path)
            df.replace(
                [
                    "ERROR",
                    "no data",
                    ".",
                    "-",
                    "/",
                    "VERIFIED",
                    "CLOTTED",
                    "*",
                    "ERROR DISREGARD PREVIOUS RESULT OF 32",
                    "DISREGARD PREVIOUSLY REPORTED 33",
                    "DISREGARD",
                    "+",
                ],
                np.nan,
                inplace=True,
            )
            df.dropna(inplace=True)
            values = df.values
            for variable, value in zip(values[:, 1], values[:, 2]):
                if variable not in var_dict:
                    var_dict[variable] = []
                    var_dict[variable].append(float(value))
                else:
                    var_dict[variable].append(float(value))

        result_dict = {}
        for feature, values in var_dict.items():
            mean_value = np.mean(values)
            variance_value = np.var(values)
            result_dict[feature] = {"mean": mean_value, "variance": variance_value}
        return result_dict


# save = False
# if save:
#     dir_list = os.listdir("../mimic3-benchmarks/data/root/train/")

#     episodes_list = []
#     for d in tqdm(dir_list):
#         d = "../mimic3-benchmarks/data/root/train/" + d
#         dirs = os.listdir(d)
#         for ep in dirs:
#             if "_timeseries" in ep:
#                 episodes_list.append(d + "/" + ep)

#     normalizer = Normalizer(episodes_list)
#     normalizer.mean_var_dict['Blood culture'] = {
#     'mean' : 0,
#     'variance' : 1
#     }

#     normalizer.mean_var_dict['Calcium Ionized'] = {
#         'mean' : 0,
#         'variance' : 1
#     }

#     normalizer.mean_var_dict['Peak inspiratory pressure'] = {
#         'mean' : 0,
#         'variance' : 1
#     }
#     with open('normalizer.pkl', 'wb') as file:
#         pickle.dump(normalizer, file)

#     print("Completed Saving Normalizer........")
