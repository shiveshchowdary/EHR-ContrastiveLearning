import pandas as pd
import numpy as np
from tqdm import tqdm

data_path = "/ssd-shared/new_split/phenotyping/train/"
listfile = "/ssd-shared/new_split/phenotyping/train_listfile.csv"
listfile = pd.read_csv(listfile)
listfile.to_csv("/ssd-shared/shivesh_eicu/phenotyping/train_listfile.csv", index=False)
for f in tqdm(listfile["stay"].unique()):
    d_f = data_path + f
    df = pd.read_csv(d_f)
    df = pd.melt(df, id_vars=["itemoffset"], var_name="Variable", value_name="Value")
    df.dropna(inplace=True)
    df.rename(columns={"itemoffset": "Hours"}, inplace=True)
    df = df.sort_values(by="Hours")
    df.to_csv("/ssd-shared/shivesh_eicu/phenotyping/train/" + f, index=False)


data_path = "/ssd-shared/new_split/phenotyping/train/"
listfile = "/ssd-shared/new_split/phenotyping/val_listfile.csv"
listfile = pd.read_csv(listfile)
listfile.to_csv("/ssd-shared/shivesh_eicu/phenotyping/val_listfile.csv", index=False)

for f in tqdm(listfile["stay"].unique()):
    d_f = data_path + f
    df = pd.read_csv(d_f)
    df = pd.melt(df, id_vars=["itemoffset"], var_name="Variable", value_name="Value")
    df.dropna(inplace=True)
    df.rename(columns={"itemoffset": "Hours"}, inplace=True)
    df = df.sort_values(by="Hours")
    df.to_csv("/ssd-shared/shivesh_eicu/phenotyping/train/" + f, index=False)


data_path = "/ssd-shared/new_split/phenotyping/test/"
listfile = "/ssd-shared/new_split/phenotyping/test_listfile.csv"
listfile = pd.read_csv(listfile)
listfile.to_csv("/ssd-shared/shivesh_eicu/phenotyping/test_listfile.csv", index=False)
for f in tqdm(listfile["stay"].unique()):
    d_f = data_path + f
    df = pd.read_csv(d_f)
    df = pd.melt(df, id_vars=["itemoffset"], var_name="Variable", value_name="Value")
    df.dropna(inplace=True)
    df.rename(columns={"itemoffset": "Hours"}, inplace=True)
    df = df.sort_values(by="Hours")
    df.to_csv("/ssd-shared/shivesh_eicu/phenotyping/test/" + f, index=False)
