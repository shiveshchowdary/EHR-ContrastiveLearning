import json
import torch
import math
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
from torchmimic.metrics import kappa
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import numpy as np

file_path = os.getcwd() + "/resources/IdNameDict.json"


with open(file_path, "r") as f:
    var_dict = json.load(f)


@torch.no_grad()
def contrastive_metrics(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    acc_count1 = 0
    acc_count2 = 0
    batch_size = 0
    for batch, index in data_loader:
        if batch_size == 0:
            batch_size = batch["encoder1_mask"].size(0)
        inp1 = [inp.cuda() for inp in batch["encoder_input1"]]
        inp2 = [inp.cuda() for inp in batch["encoder_input2"]]
        mask1 = batch["encoder1_mask"].cuda()
        mask2 = batch["encoder2_mask"].cuda()
        class_token_mask = torch.ones(mask1.size(0), 1, 1).cuda()
        mask1 = torch.cat((mask1, class_token_mask), dim=2)
        mask2 = torch.cat((mask2, class_token_mask), dim=2)

        with torch.cuda.amp.autocast(True):
            out1, _ = model(inp1, mask1)
            out2, _ = model(inp2, mask2)
            H1 = F.normalize(out1, dim=1)
            H2 = F.normalize(out2, dim=1)
            logits = torch.matmul(H1, H2.t())
            loss = criterion(H1, H2, index.cuda())

        n = logits.size(0)
        labels = torch.arange(n)
        predictions_1 = logits.argmax(dim=1).to("cpu")
        predictions_2 = logits.t().argmax(dim=1).to("cpu")

        correct_predictions_1 = (predictions_1 == labels).sum().item()
        correct_predictions_2 = (predictions_2 == labels).sum().item()

        acc_count1 += correct_predictions_1
        acc_count2 += correct_predictions_2

        total_loss += loss.item()

    total_count = len(data_loader) * batch_size
    accuracy1 = acc_count1 / total_count
    accuracy2 = acc_count2 / total_count
    accuracy = (accuracy1 + accuracy2) / 2
    return total_loss / len(data_loader), accuracy


@torch.no_grad()
def mlm_metrics(
    model, data_loader, criterion, criterion_mae, combined=False, unnormalize=False
):
    model.eval()
    total_loss = 0
    total_loss_mae = 0

    for inputs, idx in data_loader:
        pretraining_mask = inputs["pretraining_mask"].cuda()
        inputs["labels"] = inputs["labels"].cuda()
        if not combined:
            mask = inputs["encoder_mask"].cuda()
            inp = [inp.cuda() for inp in inputs["encoder_input"]]
        else:
            mask = inputs["encoder1_mask"].cuda()
            inp = [inp.cuda() for inp in inputs["encoder_input1"]]

        class_token_mask = torch.ones(mask.size(0), 1, 1).cuda()
        mask = torch.cat((mask, class_token_mask), dim=2)
        _, outputs = model(inp, mask, pretraining_mask)

        if unnormalize:

            mean_variance = data_loader.dataset.mean_variance
            mean_variance["Hours"] = {"mean": 0, "variance": 1}

            for sample_idx in range(inp[2].size(0)):
                outputs[sample_idx] = torch.FloatTensor(
                    [
                        (
                            float(i)
                            * math.sqrt(
                                mean_variance[var_dict[str(var.item())]]["variance"]
                            )
                        )
                        + mean_variance[var_dict[str(var.item())]]["mean"]
                        for i, var in zip(
                            outputs[sample_idx], inp[1][sample_idx]
                        )  
                    ]
                )

                inputs["labels"][sample_idx] = torch.FloatTensor(
                    [
                        (
                            float(i)
                            * math.sqrt(
                                mean_variance[var_dict[str(var.item())]]["variance"]
                            )
                        )
                        + mean_variance[var_dict[str(var.item())]]["mean"]
                        for i, var in zip(
                            inputs["labels"][sample_idx], inp[1][sample_idx]
                        )  
                    ]
                )

        masked_tokens = pretraining_mask == 1
        loss = criterion(outputs[masked_tokens], inputs["labels"][masked_tokens]).mean()
        total_loss += loss.item()

        mae_loss = criterion_mae(
            outputs[masked_tokens], inputs["labels"][masked_tokens]
        ).mean()
        total_loss_mae += mae_loss
    return total_loss / len(data_loader), total_loss_mae / len(data_loader)


@torch.no_grad()
def calculate_roc_auc_prc(model, data_loader):
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    all_probabilities = []
    all_labels = []
    total_loss = 0
    batches = 0
    for inputs in data_loader:
        print(f"[{batches}/{len(data_loader)}]", end="\r")
        batches += 1
        inp = [enc_inp.cuda() for enc_inp in inputs["encoder_input"]]
        mask = inputs["encoder_mask"].cuda()
        class_token_mask = torch.ones(mask.size(0), 1, 1).cuda()
        mask = torch.cat((mask, class_token_mask), dim=2)
        _, outputs = model(inp, mask)
        labels = inputs["label"].cuda()
        logits = torch.sigmoid(outputs)
        if len(labels.shape) == 1:
            loss = criterion(outputs, labels.float().view(-1, 1))
        else:
            loss = criterion(outputs, labels.float())
        total_loss += loss.item()
        all_probabilities.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())


    logits_all = np.concatenate(all_probabilities)
    labels_all = np.concatenate(all_labels)
    total_loss = total_loss / batches

    roc_auc = roc_auc_score(labels_all, logits_all)
    auc_prc = average_precision_score(labels_all, logits_all)
    return roc_auc, auc_prc, total_loss


@torch.no_grad()
def calculate_macro_micro_roc(model, data_loader):
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    all_probabilities = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for inputs in data_loader:
            inp = [enc_inp.cuda() for enc_inp in inputs["encoder_input"]]
            mask = inputs["encoder_mask"].cuda()
            class_token_mask = torch.ones(mask.size(0), 1, 1).cuda()
            mask = torch.cat((mask, class_token_mask), dim=2)
            _, outputs = model(inp, mask)
            labels = inputs["label"].cuda()
            logits = torch.sigmoid(outputs)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            all_probabilities.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    logits_all = np.concatenate(all_probabilities)
    labels_all = np.concatenate(all_labels)
    total_loss = total_loss / len(data_loader)
    # print(total_loss)
    # print(logits_all)
    # print(labels_all)
    roc_auc_macro = roc_auc_score(labels_all, logits_all, average="macro")
    roc_auc_micro = roc_auc_score(labels_all, logits_all, average="micro")
    return roc_auc_macro, roc_auc_micro, total_loss
