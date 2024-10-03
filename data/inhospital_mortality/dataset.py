import torch
import json
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import math
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset


file_path = os.getcwd() + "/resources/IdNameDict.json"


with open(file_path, "r") as f:
    var_dict = json.load(f)


class MimicDataSetInHospitalMortality(Dataset):
    def __init__(
        self,
        data_dir,
        csv_file,
        mean_variance,
        mode,
        seq_len,
        data_usage=1,
        var_dict=var_dict,
        pad_value=0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.seq_len = seq_len
        self.mode = mode
        self.data_df = csv_file
        self.data_df = self.data_df.sample(frac=data_usage)
        self.data_df = self.data_df.reset_index(drop=True)
        self.data_usage = data_usage
        self.mean_variance = mean_variance
        self.pad_value = pad_value
        self.id_name_dict = {v: k for k, v in (var_dict.items())}
        self.device = DEVICE

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        path = self.data_dir + self.data_df["stay"][idx]
        data = pd.read_csv(path)

        id_name_dict = {}

        data.replace(
            ["ERROR", "no data", ".", "-", "/", "VERIFIED"], np.nan, inplace=True
        )
        data.dropna(inplace=True)
        values = data.values

        sample = self.extract(values, self.id_name_dict)
        if len(sample[0]) >= self.seq_len and self.mode in ["validation", "testing"]:
            sample[0] = sample[0][: self.seq_len]
            sample[1] = sample[1][: self.seq_len]
            sample[2] = sample[2][: self.seq_len]
            sample[3] = sample[3][: self.seq_len]
        if len(sample[0]) >= self.seq_len and self.mode not in [
            "validation",
            "testing",
        ]:
            selected_indices = random.sample(range(len(sample[0])), self.seq_len)
            selected_indices.sort()

            sample[0] = [sample[0][i] for i in selected_indices]
            sample[1] = [sample[1][i] for i in selected_indices]
            sample[2] = [sample[2][i] for i in selected_indices]
            sample[3] = [sample[3][i] for i in selected_indices]

        num_padd_tokens = self.seq_len - len(sample[0])
        variable_input = torch.cat(
            [
                torch.tensor(sample[2], dtype=torch.int64),
                torch.tensor([self.pad_value] * num_padd_tokens, dtype=torch.int64),
            ]
        )
        value_input = torch.cat(
            [
                torch.tensor(sample[1], dtype=torch.float),
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
            "encoder_mask": (variable_input == self.pad_value).unsqueeze(0),
            "variables": variables,
            "label": torch.tensor([self.data_df["y_true"][idx]], dtype=torch.int64),
        }

    def extract(self, values, id_name_dict):
        sample = []
        time = list(values[:, 0])
        variable = list(values[:, 1])
        value = list(values[:, 2])
        count_dict = {}

        # # variable[variable == "admissionheight"] == "Height"
        # for v_idx in range(len(variable)):
        #     if variable[v_idx] == "admissionheight":
        #         variable[v_idx] = "Height"

        #     elif variable[v_idx] == "admissionweight":
        #         variable[v_idx] = "Weight"

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


file_path = os.getcwd() + "/resources/IdNameDict_eICU.json"


with open(file_path, "r") as f:
    var_dict = json.load(f)


class eICUDataSetInHospitalMortality(Dataset):
    def __init__(
        self,
        data_dir,
        csv_file,
        mean_variance,
        mode,
        seq_len,
        data_usage=1,
        var_dict=var_dict,
        pad_value=0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.seq_len = seq_len
        self.mode = mode
        self.data_df = csv_file
        self.data_df = self.data_df.sample(frac=data_usage)
        self.data_df = self.data_df.reset_index(drop=True)
        self.data_usage = data_usage
        self.mean_variance = mean_variance
        self.pad_value = pad_value
        self.id_name_dict = {v: k for k, v in (var_dict.items())}
        self.device = DEVICE

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        path = self.data_dir + self.data_df["stay"][idx]
        data = pd.read_csv(path)

        id_name_dict = {}

        data.replace(
            ["ERROR", "no data", ".", "-", "/", "VERIFIED"], np.nan, inplace=True
        )
        data.dropna(inplace=True)
        h_min = data["Hours"].min()
        h_max = 48 + h_min
        data = data[data["Hours"] <= h_max]
        values = data.values

        sample = self.extract(values, self.id_name_dict)
        if len(sample[0]) >= self.seq_len and self.mode in ["validation", "testing"]:
            sample[0] = sample[0][: self.seq_len]
            sample[1] = sample[1][: self.seq_len]
            sample[2] = sample[2][: self.seq_len]
            sample[3] = sample[3][: self.seq_len]
        if len(sample[0]) >= self.seq_len and self.mode not in [
            "validation",
            "testing",
        ]:
            selected_indices = random.sample(range(len(sample[0])), self.seq_len)

            sample[0] = [sample[0][i] for i in selected_indices]
            sample[1] = [sample[1][i] for i in selected_indices]
            sample[2] = [sample[2][i] for i in selected_indices]
            sample[3] = [sample[3][i] for i in selected_indices]

        num_padd_tokens = self.seq_len - len(sample[0])
        variable_input = torch.cat(
            [
                torch.tensor(sample[2], dtype=torch.int64),
                torch.tensor([self.pad_value] * num_padd_tokens, dtype=torch.int64),
            ]
        )
        value_input = torch.cat(
            [
                torch.tensor(sample[1], dtype=torch.float),
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
            "variables": variables,
            "label": torch.tensor([self.data_df["y_true"][idx]], dtype=torch.int64),
        }

    def extract(self, values, id_name_dict):
        sample = []
        time = list(values[:, 0])
        variable = list(values[:, 1])
        value = list(values[:, 2])
        count_dict = {}

        for idx, var in enumerate(variable):
            if var == "FiO2":
                value[idx] /= 100

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
