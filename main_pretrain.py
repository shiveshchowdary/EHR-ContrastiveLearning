from utils.normalizer import Normalizer
from pretraining import (
    train_contrastive_model,
    train_masked_pretraining_model,
    train_combined_model
)
import torch
import random
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--pretraining_method",
    type=str,
    default="MaskedValuePrediction",
    help="Method to pretrain on possible values (1. MaskedValuePrediction, 2. Contrastive, 3. Combined)",
)
parser.add_argument("--embed_dim", type=int, default=64, help="embedding dimension")
parser.add_argument("--num_heads", type=int, default=8, help="number of heads")
parser.add_argument("--N", type=int, default=2, help="number of encoder blocks")
parser.add_argument("--d_ff", type=int, default=128, help="d_ff")
parser.add_argument("--num_variables", type=int, default=68, help="number of variables")
parser.add_argument("--temperature", type=float, default=0.5, help="temperature")
parser.add_argument("--base_lr", type=float, default=0.001, help="base learning rate")
parser.add_argument("--MAX_LEN", type=int, default=1024, help="Max Sequence Length")
parser.add_argument(
    "--CROP_SIZE",
    type=float,
    default=512,
    help="Local Crop Size for Contrastive Learning",
)
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--gamma", type=float, default=0.9, help="gamma")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
parser.add_argument(
    "--data_dir",
    type=str,
    default="../mimic3-benchmarks/data/root",
    help="data directory of root",
)
parser.add_argument("--epochs", type=int, default=400, help="epochs")
parser.add_argument("--save_interval", type=int, default=50, help="save interval")
parser.add_argument(
    "--world-size", default=1, type=int, help="number of distributed processes"
)
parser.add_argument("--local-rank", default=-1, type=int)
parser.add_argument(
    "--dist-url", default="env://", help="url used to set up distributed training"
)
parser.add_argument(
    "--device", default="cuda", help="device to use for training / testing"
)

parser.add_argument("--seed", type=int, default=1992)
parser.add_argument(
    "--use_lstm", 
    type=bool, 
    default=False, 
    help="Flag to indicate whether to use LSTM instead of Transformer encoder."
)

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if args.pretraining_method == "Contrastive":
    train_contrastive_model(args)

if args.pretraining_method == "MaskedValuePrediction":
    train_masked_pretraining_model(args)

if args.pretraining_method == "Combined":
    train_combined_model(args)
