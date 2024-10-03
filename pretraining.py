from data.contrastive_learning.dataset import MimicDataSetContrastiveLearning
from data.masked_value_prediction.dataset import MaskedMimicDataSetInHospitalMortality
from data.combined.dataset import MimicDataSetCombined
from evaluation.metrics import contrastive_metrics, mlm_metrics
from models.model import *
from utils.normalizer import Normalizer
from utils.scheduler import cosine_lr
import os
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from utils.losses import SIMCLRLoss
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm import tqdm
import pandas as pd
import numpy as np

from libauc.losses import GCLoss
from libauc import optimizers
import random
import pickle

from tqdm import tqdm

from utils.distributed import init_distributed_mode


with open("resources/normalizer.pkl", "rb") as file:
    normalizer = pickle.load(file)


mean_variance = normalizer.mean_var_dict


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


def train_contrastive_model(args):
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    gpu = torch.device(args.device)
    train_episodes_list, test_episodes_list = get_data_pretraining(args.data_dir)
    # random.shuffle(train_episodes_list)

    train_ds = MimicDataSetContrastiveLearning(
        train_episodes_list[:-2000],
        normalizer.mean_var_dict,
        "training",
        args.MAX_LEN,
        args.CROP_SIZE,
    )
    val_ds = MimicDataSetContrastiveLearning(
        train_episodes_list[-2000:],
        normalizer.mean_var_dict,
        "validation",
        args.MAX_LEN,
        args.CROP_SIZE,
    )
    test_ds = MimicDataSetContrastiveLearning(
        test_episodes_list,
        normalizer.mean_var_dict,
        "validation",
        args.MAX_LEN,
        args.CROP_SIZE,
    )

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)

    train_dataloader = DataLoader(
        train_ds, batch_size=per_device_batch_size, sampler=sampler, num_workers=16
    )
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=64)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=64)

    backbone = BottleNeckModel(
        args.embed_dim, args.num_heads, args.d_ff, args.num_variables, args.N, args.use_lstm
    ).to(DEVICE)
    head = ProjectionHead(args.embed_dim).to(DEVICE)

    model = EncoderWrapper(backbone, head)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=True
    )

    criterion = GCLoss(
        "unimodal",
        N=50000,
        tau=args.temperature,
        gamma=args.gamma,
        distributed=True,
    )
    optimizer = optimizers.SogCLR(
        model.parameters(),
        mode="adamw",
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        momentum=0.9,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    print("Training Contrastive Model.........")
    scheduler = cosine_lr(
        optimizer,
        args.base_lr,
        len(train_dataloader) * 5,
        len(train_dataloader) * args.epochs,
    )
    steps = 0
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        acc_count1 = 0
        acc_count2 = 0
        for batch, index in train_dataloader:
            steps += 1
            lr = scheduler(steps)

            inp1 = [inp.cuda(gpu, non_blocking=True) for inp in batch["encoder_input1"]]
            inp2 = [inp.cuda(gpu, non_blocking=True) for inp in batch["encoder_input2"]]
            mask1 = batch["encoder1_mask"].cuda(gpu, non_blocking=True)
            mask2 = batch["encoder2_mask"].cuda(gpu, non_blocking=True)
            class_token_mask = torch.zeros(mask1.size(0), 1, 1).cuda(
                gpu, non_blocking=True
            )
            mask1 = torch.cat((mask1, class_token_mask), dim=2)
            mask2 = torch.cat((mask2, class_token_mask), dim=2)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(True):
                out1, _ = model(inp1, mask1)
                out2, _ = model(inp2, mask2)
                H1 = F.normalize(out1, dim=1)
                H2 = F.normalize(out2, dim=1)
                loss = criterion(H1, H2, index.cuda(gpu))

            if not torch.isfinite(loss).all():
                raise

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            with torch.no_grad():
                logits = torch.matmul(H1, H2.t())
                n = logits.size(0)
                labels = torch.arange(n)

                predictions_1 = logits.argmax(dim=1).to("cpu")
                predictions_2 = logits.t().argmax(dim=1).to("cpu")

                correct_predictions_1 = (predictions_1 == labels).sum().item()
                correct_predictions_2 = (predictions_2 == labels).sum().item()
                acc_count1 += correct_predictions_1
                acc_count2 += correct_predictions_2

        total_count = args.batch_size * len(train_dataloader)

        acc1 = acc_count1 / total_count
        acc2 = acc_count2 / total_count


        if args.rank == 0:
            epochs = args.epochs

            print(
                f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss/len(train_dataloader):.3f}"
            )
            print(f"Epoch {epoch + 1}/{epochs}, Train Accuracy: {(acc1 + acc2)/2:.3f}")

            if (epoch + 1) % args.save_interval == 0:
                if not args.use_lstm:
                    file_path = f"Contrastive_Pre_Trained_Model_new_{epoch}.pth"
                else:
                    file_path = f"Contrastive_Pre_Trained_Model_lstm_{epoch}.pth"

                torch.save(model.module.state_dict(), "saved_models/" + file_path)

    return


def train_masked_pretraining_model(args):
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    gpu = torch.device(args.device)
    train_episodes_list, test_episodes_list = get_data_pretraining(args.data_dir)

    train_ds = MaskedMimicDataSetInHospitalMortality(
        train_episodes_list[:-2000], normalizer.mean_var_dict, "training", args.MAX_LEN
    )
    val_ds = MaskedMimicDataSetInHospitalMortality(
        train_episodes_list[-2000:],
        normalizer.mean_var_dict,
        "validation",
        args.MAX_LEN,
    )

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)

    train_dataloader = DataLoader(
        train_ds, batch_size=per_device_batch_size, sampler=sampler, num_workers=16
    )
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size)

    bottleneckmodel = BottleNeckModel(
        args.embed_dim, args.num_heads, args.d_ff, args.num_variables, args.N, args.use_lstm
    ).to(DEVICE)
    forecast_model = ForecastModel(args.embed_dim).to(DEVICE)
    model = Model(bottleneckmodel, forecast_model).to(DEVICE)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    criterion = nn.MSELoss(reduction="none")
    criterion_mae = nn.L1Loss(reduction="none")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    print("Training  Masked Value Prediction Model.........")
    scheduler = cosine_lr(
        optimizer,
        args.base_lr,
        len(train_dataloader) * 0,
        len(train_dataloader) * args.epochs,
    )

    steps = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        model_mlm.train()
        lr = scheduler(steps)
        for batch, idx in train_dataloader:
            steps += 1
            inp = [inp.cuda(gpu, non_blocking=True) for inp in batch["encoder_input"]]
            mask = batch["encoder_mask"].cuda(gpu, non_blocking=True)
            class_token_mask = torch.zeros(mask.size(0), 1, 1).cuda(
                gpu, non_blocking=True
            )
            batch["labels"] = batch["labels"].cuda(gpu, non_blocking=True)
            mask = torch.cat((mask, class_token_mask), dim=2)
            pretraining_mask = batch["pretraining_mask"].cuda(gpu, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                _, forecast = model(inp, mask, pretraining_mask)
                forecast = forecast[:, :-1]
                masked_tokens = pretraining_mask == 1
                loss = criterion(
                    forecast[masked_tokens], batch["labels"][masked_tokens]
                ).mean()
            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        epochs = args.epochs

        print(
            f"Epoch {epoch + 1}/{args.epochs}, Training Loss: {total_loss/len(train_dataloader):.3f}"
        )
        if (epoch + 1) % args.save_interval == 0:
           
            if not args.use_lstm:
                file_path = f"MLM_Pre_Trained_Model_new_{epoch}.pth"
            else:
                file_path = f"MLM_Pre_Trained_Model_lstm_{epoch}.pth"
            torch.save(
                model.module.state_dict(),
                "saved_models/" + file_path,
            )
            torch.save(
                forecast_model.state_dict(),
                "saved_models/" + "forecast_" + file_path
            )

    test_ds = MaskedMimicDataSetInHospitalMortality(
        test_episodes_list,
        normalizer.mean_var_dict,
        "validation",
        args.MAX_LEN,
        measurement_impute_idx=1,
    )

    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=32)

    test_results = []
    model.eval()
    with torch.no_grad():
        for measurement_idx in range(1, 69):
            test_ds.measurement_impute_idx = measurement_idx
            mse, mae = mlm_metrics(
                model, test_dataloader, criterion, criterion_mae, unnormalize=True
            )
            mae = mae.item()
            print(mse, ", ", mae)
            test_results.append([mse, mae])

    print(test_results)


def train_combined_model(args):
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    gpu = torch.device(args.device)
    train_episodes_list, test_episodes_list = get_data_pretraining(args.data_dir)

    train_ds = MimicDataSetCombined(
        train_episodes_list[:-2000],
        normalizer.mean_var_dict,
        "training",
        args.MAX_LEN,
        args.CROP_SIZE,
    )

    val_ds = MimicDataSetCombined(
        train_episodes_list[-2000:],
        normalizer.mean_var_dict,
        "training",
        args.MAX_LEN,
        args.CROP_SIZE,
    )
    test_ds = MimicDataSetCombined(
        test_episodes_list,
        normalizer.mean_var_dict,
        "testing",
        args.MAX_LEN,
        args.CROP_SIZE,
    )

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=per_device_batch_size,
        sampler=sampler,
        num_workers=16,
        pin_memory=True,
    )
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size)

    bottleneckmodel = BottleNeckModel(
        args.embed_dim, args.num_heads, args.d_ff, args.num_variables, args.N, args.use_lstm
    ).to(DEVICE)
    forecast_model = ForecastModel(args.embed_dim).to(DEVICE)
    head = ProjectionHead(args.embed_dim).to(DEVICE)
    model = EncoderWrapper(
        bottleneckmodel, projector=head, forecast_model=forecast_model
    )

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    criterion_mse = nn.MSELoss(reduction="none")
    criterion_mae = nn.L1Loss(reduction="none")

    contrastive_criterion = SIMCLRLoss(temperature = args.temperature).to(args.device)


    optimizer = torch.optim.Adam(model.parameters(), args.base_lr, weight_decay=args.weight_decay)


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    print("Training Combined Model.........")
    scheduler = cosine_lr(
        optimizer,
        args.base_lr,
        len(train_dataloader) * 5,
        len(train_dataloader) * args.epochs,
    )

    steps = -1

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.epochs):
        total_loss_mse = 0
        total_loss_contrastive = 0
        acc_count1 = 0
        acc_count2 = 0
        model.train()
        lr = scheduler(steps)
        print("LR = ", optimizer.param_groups[-1]["lr"])
        for batch, index in train_dataloader:
            steps += 1
            lr = scheduler(steps)

            inp1 = [inp.cuda(gpu, non_blocking=True) for inp in batch["encoder_input1"]]
            inp2 = [inp.cuda(gpu, non_blocking=True) for inp in batch["encoder_input2"]]
            batch["labels"] = batch["labels"].cuda(gpu, non_blocking=True)

            mask1 = batch["encoder1_mask"].cuda(gpu, non_blocking=True)
            class_token_mask = torch.zeros(mask1.size(0), 1, 1).cuda(
                gpu, non_blocking=True
            )

            mask1 = torch.cat((mask1, class_token_mask), dim=2)

            mask2 = batch["encoder2_mask"].cuda(gpu, non_blocking=True)
            class_token_mask = torch.zeros(mask2.size(0), 1, 1).cuda(
                gpu, non_blocking=True
            )
            mask2 = torch.cat((mask2, class_token_mask), dim=2)

            pretraining_mask = batch["pretraining_mask"]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(True):
                embedding1, forecast = model(inp1, mask1, pretraining_mask)
                embedding2, _ = model(inp2, mask2)
                masked_tokens = pretraining_mask == 1
                loss = criterion_mse(
                    forecast[masked_tokens], batch["labels"][masked_tokens]
                ).mean()

                total_loss_mse += loss.item()

                
                loss_inp = {
                    "aug1_embed" : embedding1,
                    "aug2_embed" : embedding2
                }
                loss_dict = contrastive_criterion(loss_inp)
                loss_c = loss_dict['loss']
                acc = loss_dict['ssl_acc']
                
                if not torch.isfinite(loss_c).all():
                    print(
                        H1.min(),
                        H1.max(),
                        H2.min(),
                        H2.max(),
                        (H1 @ H1.T / 0.01).min(),
                        (H1 @ H1.T / 0.01).max(),
                    )
                    raise

                loss = loss + loss_c

            if not torch.isfinite(loss).all():
                for name, param in model.named_parameters():
                    print(loss)
                    if not torch.isfinite(param).all():
                        print(name)
                raise

            total_loss_contrastive += loss_c.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

       

        epochs = args.epochs

        print(
            f"Epoch {epoch + 1}/{epochs}, Training Contrastive Loss: {total_loss_contrastive/len(train_dataloader):.3f}"
        )
        print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {acc:.3f}")
        print(
            f"Epoch {epoch + 1}/{epochs}, Training MSE Loss: {total_loss_mse/len(train_dataloader):.3f}"
        )

       
        if (epoch + 1) % args.save_interval == 0:
            if not args.use_lstm:
                file_path = f"Combined_Pre_Trained_Model_{args.temperature}_{args.batch_size}_{args.num_heads}_{args.base_lr}_{epoch}.pth"
            else:
                file_path = f"Combined_Pre_Trained_Model_lstm_{epoch}.pth"
                
            torch.save(model.module.state_dict(), "saved_models/" + file_path)

    print(mlm_metrics(model, test_dataloader, criterion_mse, criterion_mae, True))