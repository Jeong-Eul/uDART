import argparse
import os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import *
from data_provider import *

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Visualize Attention Map",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data
parser.add_argument("--eicu_data_dir", default="./dataset/eicu_ARDS.csv", type=str, dest="eicu_data_dir")
parser.add_argument("--mimic_data_dir", default="./dataset/mimic_ARDS.csv", type=str, dest="mimic_data_dir")
parser.add_argument('--seed', default=42, type=int , dest='seed')
parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
parser.add_argument("--label_type", default='positive', type=str, dest="label_type", help = 'positive or negative')

# Model
parser.add_argument("--num_cont", default=61, type=int, dest="num_cont", help = "Nums of Continuous Features")
parser.add_argument("--num_cat", default=57, type=int, dest="num_cat", help = "Nums of Categorical Features But Not Use")
parser.add_argument("--dim", default=32, type=int, dest="dim", help = "Embedding Dimension of Input Data ")
parser.add_argument("--dim_head", default=16, type=int, dest="dim_head", help = "Dimension of Attention(Q,K,V)")
parser.add_argument("--depth", default=6, type=int, dest="depth", help = "Nums of Attention Layer Depth")
parser.add_argument("--heads", default=8, type=int, dest="heads", help='Nums of Attention head')
parser.add_argument("--attn_dropout", default=0.1, type=float, dest="attn_dropout", help='Ratio of Attention Layer dropout')
parser.add_argument("--ff_dropout", default=0.1, type=float, dest="ff_dropout", help='Ratio of FeedForward Layer dropout')

# Others
parser.add_argument("--ckpt_dir", default="./checkpoint/1st", type=str, dest="ckpt_dir")
parser.add_argument("--result_dir", default="./result/1st", type=str, dest="result_dir")


args = parser.parse_args()

def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

fix_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(f'{args.result_dir}/{args.label_type}'):
    os.makedirs(os.path.join(f'{args.result_dir}/{args.label_type}'))

## Build Dataset 
print(f'Build Visualize Dataset ....')

dataset_train = TableDataset(data_path=args.mimic_data_dir, data_type='mimic',mode='train',seed=args.seed)

# Tuple Containing the number of unique values within each category
card_categories = []
for col in dataset_train.df_cat.columns:
    card_categories.append(dataset_train.df_cat[col].nunique())

dataset_vis = VisDataset(data_path=args.eicu_data_dir,seed=args.seed, label_type = args.label_type)
dataloader_vis = DataLoader(dataset_vis,batch_size=args.batch_size, shuffle=False, num_workers=4)

columns = ['CLS_Token'] + dataset_vis.df_cat.columns[:10].tolist() + dataset_vis.df_num.columns.tolist()
print(columns)
print()
print(dataset_vis.df_cat.columns)
print(dataset_vis.df_cat.columns[:10])
print(dataset_vis.df_cat.columns[57:])