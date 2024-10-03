import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def remove_outliers_for_variable(sorted_df, val_range):
    for variable in sorted_df["Variable"].unique():
        if variable not in var_range["LEVEL2"].values:
            continue

        idx = sorted_df.Variable == variable
        v = sorted_df.Value[idx].copy()

        # Convert to numeric, handle errors with 'coerce' to replace non-numeric values with NaN
        v = pd.to_numeric(v, errors="coerce")

        # print(v)

        # Get the range values for the variable
        outlier_low = var_range.loc[
            var_range["LEVEL2"] == variable, "OUTLIER LOW"
        ].values[0]
        outlier_high = var_range.loc[
            var_range["LEVEL2"] == variable, "OUTLIER HIGH"
        ].values[0]
        valid_low = var_range.loc[var_range["LEVEL2"] == variable, "VALID LOW"].values[
            0
        ]
        valid_high = var_range.loc[
            var_range["LEVEL2"] == variable, "VALID HIGH"
        ].values[0]
        # print(outlier_low)
        # print(sorted_df.loc[idx, 'Value'])

        # Replace values below 'OUTLIER LOW' with NaN
        v.loc[v < outlier_low] = np.nan
        v.loc[v > outlier_high] = np.nan
        v.loc[v < valid_low] = valid_low
        v.loc[v > valid_high] = valid_high

        # Update the sorted_df DataFrame
        sorted_df.loc[idx, "Value"] = v
        sorted_df = sorted_df.dropna(subset=["Value"])

    return sorted_df


category_config = {
    "Glascow coma scale verbal response": {
        "No Response-ETT": 1,
        "No Response": 1,
        "1 No Response": 1,
        "1.0 ET/Trach": 1,
        "2 Incomp sounds": 2,
        "Incomprehensible sounds": 2,
        "3 Inapprop words": 3,
        "Inappropriate Words": 3,
        "4 Confused": 4,
        "Confused": 4,
        "5 Oriented": 5,
        "Oriented": 5,
    },
    "Glascow coma scale eye opening": {
        "None": 0,
        "1 No Response": 1,
        "2 To pain": 2,
        "To Pain": 2,
        "3 To speech": 3,
        "To Speech": 3,
        "4 Spontaneously": 4,
        "Spontaneously": 4,
    },
    "Glascow coma scale motor response": {
        "1 No Response": 1,
        "No response": 1,
        "2 Abnorm extensn": 2,
        "Abnormal extension": 2,
        "3 Abnorm flexion": 3,
        "Abnormal Flexion": 3,
        "4 Flex-withdraws": 4,
        "Flex-withdraws": 4,
        "5 Localizes Pain": 5,
        "Localizes Pain": 5,
        "6 Obeys Commands": 6,
        "Obeys Commands": 6,
    },
    # "Pupillary size left" : {
    #     '2 mm' : 2,
    #     '3 mm' : 3,
    #     '4 mm' : 4,
    #     '5 mm' : 5,
    #     '6 mm' : 6,
    #     '7 mm' : 7,
    #     'Fully Dilated' : np.nan,
    #     'Pinpoint' : np.nan
    # },
    # "Pupillary size right" : {
    #     '2 mm' : 2,
    #     '3 mm' : 3,
    #     '4 mm' : 4,
    #     '5 mm' : 5,
    #     '6 mm' : 6,
    #     '7 mm' : 7,
    #     'Fully Dilated' : np.nan,
    #     'Pinpoint' : np.nan
    # },
    # "Pupillary response left" : {
    # 'Brisk': 1,
    # 'Nonreactive': 3,
    # 'Other/Remarks': np.nan,
    # 'Sluggish': 2
    # },
    # "Pupillary response right" : {
    # 'Brisk': 1,
    # 'Nonreactive': 3,
    # 'Other/Remarks': np.nan,
    # 'Sluggish': 2
    # },
    # "Urine Color": {
    # 'Amber': 3,             # Moderate risk (indicating dehydration)
    # 'Brown': 4,             # High risk (possible liver or kidney issues)
    # 'Green': 2,             # Low risk (usually caused by harmless factors like food dyes)
    # 'Icteric': 4,           # High risk (indicating potential liver problems)
    # 'Light Yellow': 1,      # Very low risk (normal and healthy)
    # 'Orange': 3,            # Moderate risk (could indicate dehydration or liver issues)
    # 'Other/Remarks': np.nan,  # Variable - Risk depends on the underlying cause.
    # 'Pink': 4,              # High risk (could indicate blood in the urine)
    # 'Red': 5,               # Very high risk (indicates presence of blood, which could be from serious conditions)
    # 'Yellow': 1             # Very low risk (normal and healthy)
    # },
    # 'Urine Appearance' : {
    # 'Clear': 1,             # Very low risk (typically indicates good hydration and health)
    # 'Clots': 4,             # High risk (indicates potential bleeding in the urinary tract)
    # 'Cloudy': 3,            # Moderate risk (may indicate infection or other underlying issues)
    # 'Other/Remarks': np.nan,  # Variable - Risk depends on the underlying cause.
    # 'Sediment': 3,          # Moderate risk (may indicate presence of crystals or cells)
    # 'Sludge': 4             # High risk (may indicate presence of thickened or viscous materials)
    # }
}

data_path = "/ssd-shared/new_split/in-hospital-mortality/train/"
folder_path = "/ssd-shared/shivesh_eicu/in-hospital-mortality/train"
file_extension = "timeseries.csv"


# for file in tqdm(os.listdir(data_path)):
#     if file.endswith(file_extension):
#         if file == "listfile.csv":
#             continue
#         df = pd.read_csv(f"{data_path}/{file}", header=0)
#         df.rename(columns={"itemoffset": "Hours"}, inplace=True)
#         if len(df.columns) > 3:
#             formatted_df = pd.melt(
#                 df, id_vars=["Hours"], var_name="Variable", value_name="Value"
#             )
#             sorted_df = formatted_df.sort_values(by="Hours")
#             for variable, config in category_config.items():
#                 mask = sorted_df["Variable"] == variable
#                 sorted_df.loc[mask, "Value"] = sorted_df.loc[mask, "Value"].replace(
#                     config
#                 )
#             range_file = (
#                 "../mimic3-benchmarks/mimic3benchmark/resources/variable_ranges.csv"
#             )
#             var_range = pd.read_csv(range_file)
#             df_new = remove_outliers_for_variable(sorted_df, var_range)
#             df_new.to_csv(f"{folder_path}/{file}", index=False)


# data_path = "/ssd-shared/new_split/in-hospital-mortality/test/"
# folder_path = "/ssd-shared/shivesh_eicu/in-hospital-mortality/test"
# file_extension = "timeseries.csv"


# for file in tqdm(os.listdir(data_path)):
#     if file.endswith(file_extension):
#         if file == "listfile.csv":
#             continue
#         df = pd.read_csv(f"{data_path}/{file}", header=0)
#         df.rename(columns={"itemoffset": "Hours"}, inplace=True)
#         if len(df.columns) > 3:
#             formatted_df = pd.melt(
#                 df, id_vars=["Hours"], var_name="Variable", value_name="Value"
#             )
#             sorted_df = formatted_df.sort_values(by="Hours")
#             for variable, config in category_config.items():
#                 mask = sorted_df["Variable"] == variable
#                 sorted_df.loc[mask, "Value"] = sorted_df.loc[mask, "Value"].replace(
#                     config
#                 )
#             range_file = (
#                 "../mimic3-benchmarks/mimic3benchmark/resources/variable_ranges.csv"
#             )
#             var_range = pd.read_csv(range_file)
#             df_new = remove_outliers_for_variable(sorted_df, var_range)
#             df_new.to_csv(f"{folder_path}/{file}", index=False)

data_path = "/ssd-shared/new_split/phenotyping/train/"
folder_path = "/ssd-shared/shivesh_eicu/phenotyping/train"
file_extension = "timeseries.csv"


for file in tqdm(os.listdir(data_path)):
    if file.endswith(file_extension):
        if file == "listfile.csv":
            continue
        df = pd.read_csv(f"{data_path}/{file}", header=0)
        df.rename(columns={"itemoffset": "Hours"}, inplace=True)
        if len(df.columns) > 3:
            formatted_df = pd.melt(
                df, id_vars=["Hours"], var_name="Variable", value_name="Value"
            )
            sorted_df = formatted_df.sort_values(by="Hours")
            for variable, config in category_config.items():
                mask = sorted_df["Variable"] == variable
                sorted_df.loc[mask, "Value"] = sorted_df.loc[mask, "Value"].replace(
                    config
                )
            range_file = (
                "../mimic3-benchmarks/mimic3benchmark/resources/variable_ranges.csv"
            )
            var_range = pd.read_csv(range_file)
            df_new = remove_outliers_for_variable(sorted_df, var_range)
            df_new.to_csv(f"{folder_path}/{file}", index=False)

data_path = "/ssd-shared/new_split/phenotyping/test/"
folder_path = "/ssd-shared/shivesh_eicu/phenotyping/test"
file_extension = "timeseries.csv"


for file in tqdm(os.listdir(data_path)):
    if file.endswith(file_extension):
        if file == "listfile.csv":
            continue
        df = pd.read_csv(f"{data_path}/{file}", header=0)
        df.rename(columns={"itemoffset": "Hours"}, inplace=True)
        if len(df.columns) > 3:
            formatted_df = pd.melt(
                df, id_vars=["Hours"], var_name="Variable", value_name="Value"
            )
            sorted_df = formatted_df.sort_values(by="Hours")
            for variable, config in category_config.items():
                mask = sorted_df["Variable"] == variable
                sorted_df.loc[mask, "Value"] = sorted_df.loc[mask, "Value"].replace(
                    config
                )
            range_file = (
                "../mimic3-benchmarks/mimic3benchmark/resources/variable_ranges.csv"
            )
            var_range = pd.read_csv(range_file)
            df_new = remove_outliers_for_variable(sorted_df, var_range)
            df_new.to_csv(f"{folder_path}/{file}", index=False)

raise
# folder_path = "data_17/decompensation/train"
folder_path = "/ssd-shared/shivesh_eicu/decompensation/train"
file_extension = "timeseries.csv"


for file in tqdm(os.listdir(folder_path)):
    if file.endswith(file_extension):
        if file == "listfile.csv":
            continue
        df = pd.read_csv(f"{folder_path}/{file}", header=0)
        if len(df.columns) > 3:
            formatted_df = pd.melt(
                df, id_vars=["Hours"], var_name="Variable", value_name="Value"
            )
            sorted_df = formatted_df.sort_values(by="Hours")
            for variable, config in category_config.items():
                mask = sorted_df["Variable"] == variable
                sorted_df.loc[mask, "Value"] = sorted_df.loc[mask, "Value"].replace(
                    config
                )
            range_file = "mimic3benchmark/resources/variable_ranges.csv"
            var_range = pd.read_csv(range_file)
            df_new = remove_outliers_for_variable(sorted_df, var_range)
            df_new.to_csv(f"{folder_path}/{file}", index=False)


# folder_path = "data_17/decompensation/test"
folder_path = "/ssd-shared/shivesh_eicu/decompensation/test"
file_extension = "timeseries.csv"


for file in tqdm(os.listdir(folder_path)):
    if file.endswith(file_extension):
        if file == "listfile.csv":
            continue
        df = pd.read_csv(f"{folder_path}/{file}", header=0)
        if len(df.columns) > 3:
            formatted_df = pd.melt(
                df, id_vars=["Hours"], var_name="Variable", value_name="Value"
            )
            sorted_df = formatted_df.sort_values(by="Hours")
            for variable, config in category_config.items():
                mask = sorted_df["Variable"] == variable
                sorted_df.loc[mask, "Value"] = sorted_df.loc[mask, "Value"].replace(
                    config
                )
            range_file = "mimic3benchmark/resources/variable_ranges.csv"
            var_range = pd.read_csv(range_file)
            df_new = remove_outliers_for_variable(sorted_df, var_range)
            df_new.to_csv(f"{folder_path}/{file}", index=False)

folder_path = "data_17/length-of-stay/train"
file_extension = "timeseries.csv"


for file in tqdm(os.listdir(folder_path)):
    if file.endswith(file_extension):
        if file == "listfile.csv":
            continue
        df = pd.read_csv(f"{folder_path}/{file}", header=0)
        if len(df.columns) > 3:
            formatted_df = pd.melt(
                df, id_vars=["Hours"], var_name="Variable", value_name="Value"
            )
            sorted_df = formatted_df.sort_values(by="Hours")
            for variable, config in category_config.items():
                mask = sorted_df["Variable"] == variable
                sorted_df.loc[mask, "Value"] = sorted_df.loc[mask, "Value"].replace(
                    config
                )
            range_file = "mimic3benchmark/resources/variable_ranges.csv"
            var_range = pd.read_csv(range_file)
            df_new = remove_outliers_for_variable(sorted_df, var_range)
            df_new.to_csv(f"{folder_path}/{file}", index=False)


folder_path = "data_17/length-of-stay/test"
file_extension = "timeseries.csv"


for file in tqdm(os.listdir(folder_path)):
    if file.endswith(file_extension):
        if file == "listfile.csv":
            continue
        df = pd.read_csv(f"{folder_path}/{file}", header=0)
        if len(df.columns) > 3:
            formatted_df = pd.melt(
                df, id_vars=["Hours"], var_name="Variable", value_name="Value"
            )
            sorted_df = formatted_df.sort_values(by="Hours")
            for variable, config in category_config.items():
                mask = sorted_df["Variable"] == variable
                sorted_df.loc[mask, "Value"] = sorted_df.loc[mask, "Value"].replace(
                    config
                )
            range_file = "mimic3benchmark/resources/variable_ranges.csv"
            var_range = pd.read_csv(range_file)
            df_new = remove_outliers_for_variable(sorted_df, var_range)
            df_new.to_csv(f"{folder_path}/{file}", index=False)
