from data.inhospital_mortality.dataset import (
    MimicDataSetInHospitalMortality,
    eICUDataSetInHospitalMortality,
)
from data.phenotype.dataset import MimicDataSetPhenotyping, eICUDataSetPhenotyping

from data.combined.dataset import MimicDataSetCombined

from models.model import *
from utils.normalizer import Normalizer
from evaluation.metrics import *
import os
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import pickle

from tqdm import tqdm
from utils.scheduler import cosine_lr


class EncoderWrapper(nn.Module):
    def __init__(self, backbone, projector=None, forecast_model=None):
        super().__init__()
        self.backbone = backbone
        assert (projector is not None) or (forecast_model is not None)
        self.projector = projector
        self.forecast_model = forecast_model

    def forward(self, x, mask, pre_training_mask=None):
        z = self.backbone(x=x, mask=mask, pre_training_mask=pre_training_mask)
        if self.projector is None:
            embedding = None
        else:
            embedding = self.projector(z[:, -1, :].squeeze(1))

        if self.forecast_model is None:
            forecast = None
        else:
            forecast = self.forecast_model(z)[:, :-1]

        return embedding, forecast

with open("resources/normalizer.pkl", "rb") as file:
    normalizer = pickle.load(file)


mean_variance = normalizer.mean_var_dict

with open("resources/normalizer_eicu.pkl", "rb") as file:
    normalizer_eicu = pickle.load(file)

mean_variance_eicu = normalizer_eicu.mean_var_dict


def get_dataloaders(args):
    if args.dataset == "ihm":
        train_data_path = f"{args.data_dir}/train_listfile.csv"
        val_data_path = f"{args.data_dir}/val_listfile.csv"
        data_dir = f"{args.data_dir}/train/"

        train_csv = pd.read_csv(train_data_path)
        val_csv = pd.read_csv(val_data_path)

        train_ds = MimicDataSetInHospitalMortality(
            data_dir,
            train_csv,
            mean_variance,
            "training",
            args.MAX_LEN,
            args.data_usage,
        )
        val_ds = MimicDataSetInHospitalMortality(
            data_dir, val_csv, mean_variance, "validation", args.MAX_LEN
        )

        train_dataloader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=32,
            pin_memory=True,
        )
        val_dataloader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=32
        )

        return train_dataloader, val_dataloader

    if args.dataset == "phenotype":
        train_data_path = f"{args.data_dir}/train_listfile.csv"
        val_data_path = f"{args.data_dir}/val_listfile.csv"
        data_dir = f"{args.data_dir}/train/"

        train_csv = pd.read_csv(train_data_path)
        val_csv = pd.read_csv(val_data_path)

        train_ds = MimicDataSetPhenotyping(
            data_dir,
            train_csv,
            mean_variance,
            "training",
            args.MAX_LEN,
            args.data_usage,
        )
        val_ds = MimicDataSetPhenotyping(
            data_dir, val_csv, mean_variance, "validation", args.MAX_LEN
        )

        train_dataloader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=32,
            pin_memory=True,
        )
        val_dataloader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=32
        )

        return train_dataloader, val_dataloader

    if args.dataset == "eicu-ihm":
        train_data_path = f"{args.data_dir}/train_listfile.csv"
        val_data_path = f"{args.data_dir}/val_listfile.csv"
        data_dir = f"{args.data_dir}/train/"

        train_csv = pd.read_csv(train_data_path)
        val_csv = pd.read_csv(val_data_path)

        train_ds = eICUDataSetInHospitalMortality(
            data_dir,
            train_csv,
            mean_variance_eicu,
            "training",
            args.MAX_LEN,
            args.data_usage,
        )
        val_ds = eICUDataSetInHospitalMortality(
            data_dir, val_csv, mean_variance_eicu, "validation", args.MAX_LEN
        )

        train_dataloader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=32,
            pin_memory=True,
        )
        val_dataloader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=32
        )

        return train_dataloader, val_dataloader

    if args.dataset == "eicu-phenotype":
        train_data_path = f"{args.data_dir}/train_listfile.csv"
        val_data_path = f"{args.data_dir}/val_listfile.csv"
        data_dir = f"{args.data_dir}/train/"

        train_csv = pd.read_csv(train_data_path)
        val_csv = pd.read_csv(val_data_path)

        train_ds = eICUDataSetPhenotyping(
            data_dir,
            train_csv,
            mean_variance_eicu,
            "training",
            args.MAX_LEN,
            args.data_usage,
        )
        val_ds = eICUDataSetPhenotyping(
            data_dir, val_csv, mean_variance_eicu, "validation", args.MAX_LEN
        )

        train_dataloader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=32,
            pin_memory=True,
        )
        val_dataloader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=32
        )

        return train_dataloader, val_dataloader



def load_bottle_neck(args):
    bottleneck_model = BottleNeckModel(
        args.embed_dim, args.num_heads, args.d_ff, args.num_variables, args.N, args.use_lstm
    ).to(DEVICE)
    head = ProjectionHead(args.embed_dim).to(DEVICE)
    model = EncoderWrapper(bottleneck_model, head)

    if args.method == "Contrastive":
        print("Loading Contrastive Model")
        if not args.use_lstm:
            file_path = f"saved_models/Contrastive_Pre_Trained_Model_new_{args.load_epoch}.pth"
        else:
            file_path = f"saved_models/Contrastive_Pre_Trained_Model_lstm_{args.load_epoch}.pth"
        state_dict = torch.load(file_path, map_location="cpu")
        state_dict = {
            key: value for key, value in state_dict.items() if "backbone" in key
        }
        state_dict = {
            key.replace("backbone.", ""): value for key, value in state_dict.items()
        }
    
        bottleneck_model.load_state_dict(state_dict)
        return bottleneck_model
    
        
    if args.method == "MaskedValuePrediction":
        print("Loading Masked Value Prediction Model")
        if not args.use_lstm:
            file_path = f"saved_models/MLM_Pre_Trained_Model_new_{args.load_epoch}.pth"
        else:
            file_path = f"saved_models/MLM_Pre_Trained_Model_lstm_{args.load_epoch}.pth"

        state_dict = torch.load(file_path, map_location="cpu")
        state_dict = {
            key.replace("bottleneck_model.", ""): value
            for key, value in state_dict.items()
            if "bottleneck_model" in key
        }
        state_dict = {
            key.replace("backbone.", ""): value for key, value in state_dict.items()
        }
        bottleneck_model.load_state_dict(state_dict)
        return bottleneck_model
        
    if args.method == "Combined":
        print("Loading Combined Model")
        if not args.use_lstm:
            
            file_path = f"saved_models/Combined_Pre_Trained_Model_{args.load_epoch}.pth"
        else:
            file_path = f"saved_models/Combined_Pre_Trained_Model_lstm_{args.load_epoch}.pth"
        print(f"Loading from {file_path}")
        state_dict = torch.load(file_path, map_location="cpu")
        try:
            bottleneck_model.load_state_dict(state_dict)
            return bottleneck_model

        except:
            state_dict = {
                key: value for key, value in state_dict.items() if "backbone" in key
            }
            state_dict = {
                key.replace("backbone.", ""): value for key, value in state_dict.items()
            }
        
            bottleneck_model.load_state_dict(state_dict)
            return bottleneck_model
    


def get_data_pretraining(data_dir):
    train_data_dir = data_dir + "/train/"
    test_data_dir = data_dir + "/test/"

    train_dir_list = os.listdir(train_data_dir)
    test_dir_list = os.listdir(test_data_dir)

    train_episodes_list = []
    for d in tqdm(train_dir_list):
        d = train_data_dir + d
        dirs = os.listdir(d)
        for ep in dirs:
            if "_timeseries" in ep:
                train_episodes_list.append(d + "/" + ep)

    test_episodes_list = []
    for d in tqdm(test_dir_list):
        d = test_data_dir + d
        dirs = os.listdir(d)
        for ep in dirs:
            if "_timeseries" in ep:
                test_episodes_list.append(d + "/" + ep)
    return train_episodes_list, test_episodes_list


def imputation_test(args):
    bottleneck_model = load_bottle_neck(args)

    prediction_head = PredictionModel(args.embed_dim, 1).to(DEVICE)
    model = Model(bottleneck_model, prediction_head)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    criterion_mse = nn.MSELoss(reduction="none")
    criterion_mae = nn.L1Loss(reduction="none")

    train_episodes_list, test_episodes_list = get_data_pretraining(args.data_dir)

    test_ds = MimicDataSetCombined(
        test_episodes_list,
        normalizer.mean_var_dict,
        "testing",
        1024,
        512,
    )

    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size)

    res_dict = {"MAE": [], "MSE": []}

    mse, mae = mlm_metrics(model, test_dataloader, criterion_mse, criterion_mae, True)
    res_dict["MAE"].append(mse)
    res_dict["MSE"].append(mae)

    print(res_dict)
    with open(
        f"results/method_{method}_pretrained_on_{args.method}_task_{args.dataset}.json",
        "w",
    ) as f:
        json.dump(res_dict, f)


def finetune_ihm(args):
    bottleneck_model = load_bottle_neck(args)

    train_dataloader, val_dataloader = get_dataloaders(args)

    prediction_head = PredictionModel(args.embed_dim, 1).to(DEVICE)
    model = Model(bottleneck_model, prediction_head)
    if not args.linear_eval:
        optimizer_pre = torch.optim.Adam(
            model.bottleneck_model.parameters(), lr=args.pre_lr
        )
    optimizer_pred = torch.optim.Adam(model.out_model.parameters(), lr=args.head_lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    print("Training...............")
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    early_stopping_counter = 0
    patience = args.patience
    scheduler = cosine_lr(
        optimizer_pred,
        args.head_lr,
        0,
        len(train_dataloader) * args.epochs,
    )

    step = -1

    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in train_dataloader:
            step += 1
            lr = scheduler(step)
            inp = [enc_inp.cuda() for enc_inp in batch["encoder_input"]]
            mask = batch["encoder_mask"].cuda()
            y = batch["label"].cuda()
            class_token_mask = torch.ones(mask.size(0), 1, 1).cuda()
            mask = torch.cat((mask, class_token_mask), dim=2)
            _, outputs = model(inp, mask)
            loss = criterion(outputs, y.float().view(-1, 1))
            total_loss += loss.item()
            if not args.linear_eval:
                optimizer_pre.zero_grad()
            optimizer_pred.zero_grad()

            loss.backward()

            if not args.linear_eval:
                optimizer_pre.step()
            optimizer_pred.step()

        auc_roc, auc_prc, val_loss = calculate_roc_auc_prc(model, val_dataloader)

        print(f"Epoch {epoch + 1}/{args.epochs}, Validation AUC-ROC: {auc_roc:.3f}")
        print(f"Epoch {epoch + 1}/{args.epochs}, Validation AUC-PRC: {auc_prc:.3f}")
        print(
            f"Epoch {epoch + 1}/{args.epochs}, Training Loss: {total_loss/len(train_dataloader):.3f}"
        )
        print(f"Epoch {epoch + 1}/{args.epochs}, Validation Loss: {val_loss:.3f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break
    if args.linear_eval:
        method = "linear_eval"
    else:
        method = f"semi_supervised_datausage{args.data_usage}"

    torch.save(
        model.state_dict(),
        "saved_models/"
        + f"method_{method}_pretrained_on_{args.method}_task_{args.dataset}.pth",
    )

    print("Testing..........")

    test_data_dir = f"{args.data_dir}/test/"
    test_data_path = f"{args.data_dir}/test_listfile.csv"

    res_dict = {"auc_roc": [], "auc_pr": []}
    data = pd.read_csv(test_data_path)
    for k in range(args.K):
        resampled_data = data
        resampled_data.reset_index(drop=True, inplace=True)
        if args.dataset == "eicu-ihm":
            test_ds = eICUDataSetInHospitalMortality(
                test_data_dir,
                resampled_data,
                mean_variance_eicu,
                "testing",
                args.MAX_LEN,
            )
        else:
            test_ds = MimicDataSetInHospitalMortality(
                test_data_dir, resampled_data, mean_variance, "testing", args.MAX_LEN
            )
        test_dataloader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=32
        )
        auc_roc, auc_prc, test_loss = calculate_roc_auc_prc(model, test_dataloader)
        res_dict["auc_roc"].append(auc_roc)
        res_dict["auc_pr"].append(auc_prc)

    print(res_dict)
    with open(
        f"results/method_{method}_pretrained_on_{args.method}_task_{args.dataset}.json",
        "w",
    ) as f:
        json.dump(res_dict, f)


def finetune_phenotype(args):
    bottleneck_model = load_bottle_neck(args)

    train_dataloader, val_dataloader = get_dataloaders(args)

    prediction_head = PredictionModel(args.embed_dim, 25).to(DEVICE)
    model = Model(bottleneck_model, prediction_head)
    if not args.linear_eval:
        optimizer_pre = torch.optim.Adam(
            model.bottleneck_model.parameters(), lr=args.pre_lr
        )
    optimizer_pred = torch.optim.Adam(model.out_model.parameters(), lr=args.head_lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    print("Training...............")
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    early_stopping_counter = 0
    patience = args.patience
    scheduler = cosine_lr(
        optimizer_pred,
        args.head_lr,
        0,
        len(train_dataloader) * args.epochs,
    )

    step = -1
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in train_dataloader:
            step += 1
            lr = scheduler(step)
            inp = [enc_inp.cuda() for enc_inp in batch["encoder_input"]]
            mask = batch["encoder_mask"].cuda()
            y = batch["label"].cuda()
            class_token_mask = torch.ones(mask.size(0), 1, 1).to(mask.device)
            mask = torch.cat((mask, class_token_mask), dim=2)
            if args.linear_eval:
                with torch.no_grad():
                    outputs = model.bottleneck_model(inp, mask)
            else:
                outputs = model.bottleneck_model(inp, mask)

            outputs = model.out_model(outputs, mask)
            loss = criterion(outputs, y.float())
            total_loss += loss.item()
            if not args.linear_eval:
                optimizer_pre.zero_grad()
            optimizer_pred.zero_grad()

            loss.backward()

            if not args.linear_eval:
                optimizer_pre.step()
            optimizer_pred.step()

        roc_auc_macro, roc_auc_micro, val_loss = calculate_macro_micro_roc(
            model, val_dataloader
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs}, Validation Macro AUC-ROC: {roc_auc_macro:.3f}"
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs}, Validation Micro AUC-ROC: {roc_auc_micro:.3f}"
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs}, Training Loss: {total_loss/len(train_dataloader):.3f}"
        )
        print(f"Epoch {epoch + 1}/{args.epochs}, Validation Loss: {val_loss:.3f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break
    if args.linear_eval:
        method = "linear_eval"
    else:
        method = f"semi_supervised_datausage{args.data_usage}"

    torch.save(
        model.state_dict(),
        "saved_models/"
        + f"method_{method}_pretrained_on_{args.method}_task_{args.dataset}.pth",
    )

    print("Testing..........")

    test_data_dir = f"{args.data_dir}/test/"
    test_data_path = f"{args.data_dir}/test_listfile.csv"

    res_dict = {"macro": [], "micro": []}
    data = pd.read_csv(test_data_path)
    for k in range(args.K):
        resampled_data = data  
        resampled_data.reset_index(drop=True, inplace=True)
        if args.dataset == "eicu-phenotype":
            test_ds = eICUDataSetPhenotyping(
                test_data_dir,
                resampled_data,
                mean_variance_eicu,
                "testing",
                args.MAX_LEN,
            )
        else:
            test_ds = MimicDataSetPhenotyping(
                test_data_dir, resampled_data, mean_variance, "testing", args.MAX_LEN
            )
        test_dataloader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=32
        )
        roc_auc_macro, roc_auc_micro, test_loss = calculate_macro_micro_roc(
            model, test_dataloader
        )
        res_dict["macro"].append(roc_auc_macro)
        res_dict["micro"].append(roc_auc_micro)

    print(res_dict)
    with open(
        f"results/method_{method}_pretrained_on_{args.method}_task_{args.dataset}.json",
        "w",
    ) as f:
        json.dump(res_dict, f)
