from utils.normalizer import Normalizer
from downstream import (
    finetune_ihm,
    finetune_phenotype,
    imputation_test,
)
import torch
import random
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--method",
    choices=["baseline", "MaskedValuePrediction", "Contrastive", "Combined"],
    default="baseline",
    help="loading pretrained models. possible values (1. baseline, 2. MaskedValuePrediction, 3. Contrastive, 4. Combined)",
)
parser.add_argument(
    "--dataset",
    choices=[
        "ihm",
        "phenotype",
        "imputation",
        "eicu-ihm",
        "eicu-phenotype",
        "eicu-imputation",
    ],
    default="phenotype",
    help="dataset to downstream on. possible values (1 . ihm 2. phenotype). for loading eICU datasets just mention eicu-{datasetname}",
)
parser.add_argument("--embed_dim", type=int, default=64, help="embedding dimension")
parser.add_argument("--num_heads", type=int, default=8, help="number of heads")
parser.add_argument("--N", type=int, default=2, help="number of encoder blocks")
parser.add_argument("--d_ff", type=int, default=128, help="d_ff")
parser.add_argument("--num_variables", type=int, default=68, help="number of variables")
parser.add_argument(
    "--pre_lr", type=float, default=8e-4, help="learning rate of bottle neck"
)
parser.add_argument(
    "--head_lr", type=float, default=8e-4, help="learning rate of prediction head"
)

parser.add_argument("--MAX_LEN", type=int, default=1024, help="Max Sequence Length")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
parser.add_argument(
    "--data_dir",
    type=str,
    default="../mimic3-benchmarks/data_17/phenotyping",
    help="data directory of the dataset",
)
parser.add_argument(
    "--data_usage",
    type=float,
    default=1,
    help="percent of data to use for semi supervised learing",
)
parser.add_argument(
    "--linear_eval",
    action="store_true",
    help=" use 1 for line linear evaluation 0 to train everything make sure the data_usage is 1 for linear evaluation",
)
parser.add_argument(
    "--patience", type=int, default=10, help="patience for early stopping"
)

parser.add_argument(
    "--load_epoch", type=int, default=399, help="epoch checkpoint to load"
)
parser.add_argument("--epochs", type=int, default=1, help="epochs")
parser.add_argument(
    "--K", type=int, default=200, help="number of resamples for testing"
)
parser.add_argument(
    "--use_lstm", 
    type=bool, 
    default=False, 
    help="Flag to indicate whether to use LSTM instead of Transformer encoder."
)
parser.add_argument("--seed", type=int, default=1992)


args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)


if args.dataset in ["ihm", "eicu-ihm"]:
    finetune_ihm(args)

elif args.dataset in ["phenotype", "eicu-phenotype"]:
    finetune_phenotype(args)
    
elif args.dataset == "imputation":
    imputation_test(args)
    
else:
    print("No downstream dataset found")
    



