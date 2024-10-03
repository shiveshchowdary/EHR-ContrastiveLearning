import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import math
import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F

# pd.set_option('future.no_silent_downcasting',True)
import json

import os

file_path = os.getcwd() + "/resources/IdNameDict.json"


with open(file_path, "r") as f:
    var_dict = json.load(f)


class MaskedMimicDataSetInHospitalMortality(Dataset):
    def __init__(
        self,
        episodes_list,
        mean_variance,
        mode,
        seq_len,
        var_dict=var_dict,
        pad_value=0,
        measurement_impute_idx=6,
    ):
        super().__init__()
        self.episodes_list = episodes_list
        self.seq_len = seq_len
        self.mode = mode
        self.mean_variance = mean_variance
        self.pad_value = pad_value
        self.device = DEVICE
        self.id_name_dict = {v: k for k, v in (var_dict.items())}

        self.measurement_impute_idx = measurement_impute_idx

        print(
            "Imputation test on ", list(var_dict.values())[self.measurement_impute_idx]
        )

    def __len__(self):
        return len(self.episodes_list)

    def __getitem__(self, idx):
        path = self.episodes_list[idx]
        data = pd.read_csv(path)

        id_name_dict = {}
        data.replace(
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
        data.dropna(inplace=True)
        values = data.values

        sample = self.extract(values, self.id_name_dict)

        if len(sample[0]) >= self.seq_len:
            selected_indices = random.sample(range(len(sample[0])), self.seq_len)
            selected_indices.sort()

            sample[0] = [sample[0][i] for i in selected_indices]
            sample[1] = [sample[1][i] for i in selected_indices]
            sample[2] = [sample[2][i] for i in selected_indices]
            sample[3] = [sample[3][i] for i in selected_indices]

        num_padd_tokens = self.seq_len - len(sample[0])

        mask_len = len(sample[0])

        mask = torch.rand(mask_len) <= 0.15
        val_input = torch.tensor(sample[1], dtype=torch.float)
        val_input[mask] = 0
        variable_input = torch.cat(
            [
                torch.tensor(sample[2], dtype=torch.int64),
                torch.tensor([self.pad_value] * num_padd_tokens, dtype=torch.int64),
            ]
        )
        value_input = torch.cat(
            [
                val_input,
                torch.tensor([self.pad_value] * num_padd_tokens, dtype=torch.float),
            ]
        )
        val = torch.tensor(sample[0], dtype=torch.float)
        time_input = torch.cat(
            [
                val - val.min(),
                torch.tensor([self.pad_value] * num_padd_tokens, dtype=torch.float),
            ]
        )
        variables = sample[3] + ["pad token"] * num_padd_tokens

        assert variable_input.size(0) == self.seq_len
        assert value_input.size(0) == self.seq_len
        assert time_input.size(0) == self.seq_len

        return {
            "encoder_input": [
                time_input,
                variable_input,
                value_input,
            ],
            "encoder_mask": (variable_input != self.pad_value).unsqueeze(0).int(),
            "pretraining_mask": torch.cat(
                [mask, torch.tensor([-1] * num_padd_tokens)]
            ).int(),
            "labels": torch.cat(
                [
                    torch.tensor(sample[1], dtype=torch.float),
                    torch.tensor([self.pad_value] * num_padd_tokens, dtype=torch.float),
                ]
            ),
            "variables": variables,
        }, idx

    def extract(self, values, id_name_dict):
        sample = []
        time = list(values[:, 0])
        variable = list(values[:, 1])
        value = list(values[:, 2])
        count_dict = {}

        value = [
            (float(i) - self.mean_variance[var]["mean"])
            / math.sqrt(self.mean_variance[var]["variance"])
            for i, var in zip(value, variable)
        ]

        varibale_id = [int(id_name_dict[i]) for i in variable]
        sample.append(time)
        sample.append(value)
        sample.append(varibale_id)
        sample.append(variable)

        return sample
