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

measurement_impute_idx = 2


class MimicDataSetCombined(Dataset):
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

        # ep_del = [
        #     "../MIMIC_EICU_DATA/data/root/train/21164/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/907/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/15128/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/2153/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/91465/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/10304/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/15093/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/32453/episode6_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/19097/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/18168/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/89140/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/32156/episode3_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/26620/episode2_timeseries.csv"
        #     "../MIMIC_EICU_DATA/data/root/train/22667/episode1_timeseries.csv"
        #     "../MIMIC_EICU_DATA/data/root/train/4392/episode4_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/7805/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/31602/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/14469/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/64550/episode3_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/90152/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/27793/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/68709/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/69398/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/20124/episode3_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/26620/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/7589/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/6121/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/57713/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/72930/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/16680/episode3_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/91855/episode5_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/77815/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/12523/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/52577/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/12229/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/11043/episode3_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/20913/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/235/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/15186/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/51847/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/6824/episode4_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/68089/episode3_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/23064/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/18168/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/79050/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/3917/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/26406/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/2571/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/26118/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/48426/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/99433/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/27163/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/31320/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/73139/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/72083/episode3_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/21607/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/24481/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/30675/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/77771/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/13941/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/32453/episode7_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/13316/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/90354/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/4392/episode4_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/21366/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/30552/episode5_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/31069/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/7808/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/45985/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/53632/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/14755/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/23651/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/8257/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/81810/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/14953/episode5_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/72456/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/5393/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/70147/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/45723/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/9403/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/29806/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/40837/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/32167/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/71459/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/27197/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/913/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/10358/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/68228/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/28115/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/14498/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/88500/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/89992/episode4_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/26582/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/55973/episode11_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/30991/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/8072/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/18350/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/904/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/22667/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/42447/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/49535/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/91539/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/11745/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/449/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/52978/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/4857/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/87687/episode3_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/4108/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/2087/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/9363/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/18693/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/8267/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/21817/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/13144/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/3974/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/train/24975/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/test/75320/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/test/51203/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/test/24729/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/test/16798/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/test/9686/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/test/14898/episode3_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/test/54353/episode3_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/test/26135/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/test/23647/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/test/28869/episode1_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/test/22206/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/test/29129/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/test/29819/episode2_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/test/63637/episode4_timeseries.csv",
        #     "../MIMIC_EICU_DATA/data/root/test/63637/episode3_timeseries.csv",
        # ]

        # for del_ep in ep_del:
        #     try:
        #         self.episodes_list.remove(del_ep)
        #     except Exception as e:
        #         continue

        # self.data_df.replace(ep_del, np.nan, inplace=True)
        # self.data_df.dropna(inplace=True)
        # self.data_df.reset_index(drop=True, inplace=True)

        self.seq_len = seq_len
        self.mode = mode
        self.mean_variance = mean_variance
        self.pad_value = pad_value
        self.crop_size = crop_size
        self.device = DEVICE
        self.id_name_dict = {v: k for k, v in (var_dict.items())}

        # print("Imputation test on ", list(var_dict.values())[measurement_impute_idx])

        # for ep in self.episodes_list:
        #     data = pd.read_csv(ep)
        #     data.replace(
        #         [
        #             "ERROR",
        #             "no data",
        #             ".",
        #             "-",
        #             "/",
        #             "VERIFIED",
        #             "CLOTTED",
        #             "*",
        #             "ERROR DISREGARD PREVIOUS RESULT OF 32",
        #             "DISREGARD PREVIOUSLY REPORTED 33",
        #             "DISREGARD",
        #             "+",
        #         ],
        #         np.nan,
        #         inplace=True,
        #     )
        #     data.dropna(inplace=True)
        #     values = data.values
        #     if data.empty:
        #         print(path)

        #     sample = self.extract(values, self.id_name_dict)

        #     if len(sample[0]) == 0:
        #         print(ep)

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
        if data.empty:
            print(path)

        sample = self.extract(values, self.id_name_dict)

        assert len(sample[0]) > 0, path

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
                    val + val + np.random.normal(0, np.random.uniform(0.1, 0.2))
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

        mask_len = len(subset1[0])
        mask = torch.rand(mask_len) <= 0.10

        # mask = torch.BoolTensor(
        #     [
        #         c_var == int(list(var_dict.keys())[measurement_impute_idx])
        #         for c_var in subset1[2]
        #     ]
        # )
        val_input = torch.tensor(subset1[1], dtype=torch.float)
        val_input[mask] = 0

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
                val_input,
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
            "pretraining_mask": torch.cat(
                [mask, torch.tensor([-1] * subset1_num_padd_tokens)]
            ).int(),
            "labels": torch.cat(
                [
                    torch.tensor(subset1[1], dtype=torch.float),
                    torch.tensor(
                        [self.pad_value] * subset1_num_padd_tokens, dtype=torch.float
                    ),
                ]
            ),
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
        # acceted_features = [
        #     "itemoffset",
        #     "Capillary Refill",
        #     "Invasive BP Diastolic",
        #     "FiO2",
        #     "Eyes",
        #     "Motor",
        #     "GCS Total",
        #     "Verbal",
        #     "glucose",
        #     "Heart Rate",
        #     "admissionheight",
        #     "MAP (mmHg)",
        #     "O2 Saturation",
        #     "Respiratory Rate",
        #     "Invasive BP Systolic",
        #     "Temperature (C)",
        #     "admissionweight",
        #     "pH",
        # ]
        # tmp_time = []
        # tmp_var = []
        # tmp_value = []
        # for time, var, value in zip(values[:, 0], values[:, 1], values[:, 2]):
        #     if var in acceted_features:
        #         tmp_time.append(time)
        #         tmp_var.append(var)
        #         tmp_value.append(value)

        # time = tmp_time
        # variable = tmp_var
        # value = tmp_value

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
