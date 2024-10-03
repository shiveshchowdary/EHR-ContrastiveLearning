import torch
import json
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import math
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset

# pd.set_option('future.no_silent_downcasting',True)


file_path = os.getcwd() + "/resources/IdNameDict.json"


with open(file_path, "r") as f:
    var_dict = json.load(f)


class MimicDataSetContrastiveLearning(Dataset):
    def __init__(
        self,
        episodes_list,
        mean_variance,
        mode,
        seq_len,
        crop_size,
        var_dict=var_dict,
        pad_value=0,
    ):
        super().__init__()
        self.episodes_list = episodes_list
        self.seq_len = seq_len
        self.mode = mode
        self.mean_variance = mean_variance
        self.pad_value = pad_value
        self.id_name_dict = {v: k for k, v in (var_dict.items())}
        self.crop_size = crop_size
        self.device = DEVICE

    def __len__(self):
        return len(self.episodes_list)

    def __getitem__(self, idx):
        CROP_SIZE = self.crop_size
        path = self.episodes_list[idx]
        data = pd.read_csv(path)
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

        if CROP_SIZE >= len(sample[0]):
            crop_size = int(len(sample[0]) // 2)
            if crop_size == 0:
                crop_size = 1
        else:
            crop_size = CROP_SIZE

        methods = [1, 2, 3]
        method_index = random.sample(methods, 1)
        method = method_index[0]

        if method == 1:
            start_index = random.randint(0, len(sample[0]) - crop_size)
            subset2 = []
            subset2.append(sample[0][start_index : start_index + crop_size])
            subset2.append(sample[1][start_index : start_index + crop_size])
            subset2.append(sample[2][start_index : start_index + crop_size])
            subset2.append(sample[3][start_index : start_index + crop_size])
        if method == 2:
            selected_indices = random.sample(range(len(sample[0])), crop_size)
            selected_indices.sort()
            subset2 = []
            subset2.append([sample[0][i] for i in selected_indices])
            subset2.append([sample[1][i] for i in selected_indices])
            subset2.append([sample[2][i] for i in selected_indices])
            subset2.append([sample[3][i] for i in selected_indices])
        if method == 3:
            subset2 = []
            subset2.append(sample[0])
            subset2.append(
                [
                    val + np.random.normal(0, np.random.uniform(0.1, 0.2))
                    for val in sample[1]
                ]
            )
            subset2.append(sample[2])
            subset2.append(sample[3])
        subset1 = sample

        subset1_num_padd_tokens = self.seq_len - len(subset1[0])
        subset2_num_padd_tokens = self.seq_len - len(subset2[0])
        time1 = torch.tensor(subset1[0], dtype=torch.float)
        time2 = torch.tensor(subset2[0], dtype=torch.float)

        time_subset1 = torch.cat(
            [
                time1 - time1.min(),
                torch.tensor(
                    [self.pad_value] * subset1_num_padd_tokens, dtype=torch.float
                ),
            ]
        )
        time_subset2 = torch.cat(
            [
                time2 - time2.min(),
                torch.tensor(
                    [self.pad_value] * subset2_num_padd_tokens, dtype=torch.float
                ),
            ]
        )

        val_subset1 = torch.cat(
            [
                torch.tensor(subset1[1], dtype=torch.float),
                torch.tensor(
                    [self.pad_value] * subset1_num_padd_tokens, dtype=torch.float
                ),
            ]
        )
        val_subset2 = torch.cat(
            [
                torch.tensor(subset2[1], dtype=torch.float),
                torch.tensor(
                    [self.pad_value] * subset2_num_padd_tokens, dtype=torch.float
                ),
            ]
        )

        var_subset1 = torch.cat(
            [
                torch.tensor(subset1[2], dtype=torch.int64),
                torch.tensor(
                    [self.pad_value] * subset1_num_padd_tokens, dtype=torch.int64
                ),
            ]
        )
        var_subset2 = torch.cat(
            [
                torch.tensor(subset2[2], dtype=torch.int64),
                torch.tensor(
                    [self.pad_value] * subset2_num_padd_tokens, dtype=torch.int64
                ),
            ]
        )

        assert var_subset1.size(0) == self.seq_len
        assert val_subset2.size(0) == self.seq_len
        assert time_subset1.size(0) == self.seq_len

        return {
            "encoder_input1": [
                time_subset1,
                var_subset1,
                val_subset1,
            ],
            "encoder_input2": [
                time_subset2,
                var_subset2,
                val_subset2,
            ],
            "encoder1_mask": (var_subset1 == self.pad_value).unsqueeze(0),
            "encoder2_mask": (var_subset2 == self.pad_value).unsqueeze(0),
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
