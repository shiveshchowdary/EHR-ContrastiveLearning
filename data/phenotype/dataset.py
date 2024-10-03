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


class MimicDataSetPhenotyping(Dataset):
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
        self.data_df.replace(["21607_episode1_timeseries.csv"], np.nan, inplace=True)
        self.data_df.dropna(inplace=True)
        self.data_df.reset_index(drop=True, inplace=True)
        self.data_df = self.data_df.sample(frac=data_usage)
        self.data_df = self.data_df.reset_index(drop=True)
        self.mean_variance = mean_variance
        self.pad_value = pad_value
        self.device = DEVICE
        self.id_name_dict = {v: k for k, v in (var_dict.items())}

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        path = self.data_dir + self.data_df["stay"][idx]
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
        period = self.data_df["period_length"][idx]
        data["drop"] = data["Hours"] < period
        data.replace(False, np.nan, inplace=True)
        data.dropna(inplace=True)
        data.drop(["drop"], axis=1, inplace=True)
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
        cols = self.data_df.columns[2:]
        y_true = list(self.data_df.iloc[idx][cols].values)
        return {
            "encoder_input": [
                time_input,
                variable_input,
                value_input,
            ],
            "encoder_mask": (variable_input == self.pad_value).unsqueeze(0),
            "variables": variables,
            "label": torch.tensor(y_true, dtype=torch.int64),
        }

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


file_path = os.getcwd() + "/resources/IdNameDict_eICU.json"


with open(file_path, "r") as f:
    var_dict = json.load(f)


class eICUDataSetPhenotyping(Dataset):
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
        self.data_df.replace(
            [
                "21607_episode1_timeseries.csv",
                "02713901_episode1_timeseries.csv",
                "00245035_episode2_timeseries.csv",
            ],
            np.nan,
            inplace=True,
        )
        self.data_df.dropna(inplace=True)
        self.data_df.reset_index(drop=True, inplace=True)
        self.data_df = self.data_df.sample(frac=data_usage)
        self.data_df = self.data_df.reset_index(drop=True)
        self.mean_variance = mean_variance
        self.pad_value = pad_value
        self.device = DEVICE
        self.id_name_dict = {v: k for k, v in (var_dict.items())}

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        path = self.data_dir + self.data_df["stay"][idx]
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
        period = self.data_df["period_length"][idx]
        data["drop"] = data["Hours"] < period
        data.replace(False, np.nan, inplace=True)
        data.dropna(inplace=True)
        data.drop(["drop"], axis=1, inplace=True)
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

        assert len(sample[0]) > 0, path
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
        assert time_input.size(0) == self.seq_len, time_input.size()
        cols = self.data_df.columns[2:]
        y_true = list(self.data_df.iloc[idx][cols].values)
        return {
            "encoder_input": [
                time_input,
                variable_input,
                value_input,
            ],
            "encoder_mask": (variable_input != self.pad_value).unsqueeze(0).int(),
            "variables": variables,
            "label": torch.tensor(y_true, dtype=torch.int64),
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

        for i, var in zip(value, variable):
            if var == "FiO2":
                assert i <= 1.2

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
