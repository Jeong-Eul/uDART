import argparse
import os
import sys
import time
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score ,f1_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from model import *
from data_provider import *
from data_provider_positive import *
from losses import *
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser(description="Train the DANN_FTT", 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    parser.add_argument("--hirid_data_dir", default="./Data/HiRID_ARDS_12H_SPLIT.csv.gz", type=str, dest="hirid_data_dir")
    parser.add_argument("--pic_data_dir", default="./Data/MIMIC_ARDS_12H_SPLIT.csv.gz", type=str, dest="pic_data_dir")
    parser.add_argument('--seed', default=42, type=int , dest='seed')

    # Model
    parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
    parser.add_argument("--num_cont", default=61, type=int, dest="num_cont", help = "Nums of Continuous Features")
    parser.add_argument("--num_cat", default=57, type=int, dest="num_cat", help = "Nums of Categorical Features But Not Use")
    parser.add_argument("--dim", default=32, type=int, dest="dim", help = "Embedding Dimension of Input Data ")
    parser.add_argument("--dim_head", default=16, type=int, dest="dim_head", help = "Dimension of Attention(Q,K,V)")
    parser.add_argument("--depth", default=6, type=int, dest="depth", help = "Nums of Attention Layer Depth")
    parser.add_argument("--heads", default=8, type=int, dest="heads", help='Nums of Attention head')
    parser.add_argument("--attn_dropout", default=0.1, type=float, dest="attn_dropout", help='Ratio of Attention Layer dropout')
    parser.add_argument("--ff_dropout", default=0.1, type=float, dest="ff_dropout", help='Ratio of FeedForward Layer dropout')

    # Others
    parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
    parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")
    parser.add_argument("--printiter", default=500, type=int, dest="printiter", help="Number of iters to print")
    parser.add_argument("--mode", default='train', type=str, dest="mode", help="choose train / test")

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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## Make folder 
    if not os.path.exists(args.result_dir):
        os.makedirs(os.path.join(args.result_dir))

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(os.path.join(args.ckpt_dir))

    ## Build Dataset 
    print(f'Build Source Domain Dataset : {args.hirid_data_dir} ....')
    dataset_s_train = PositiveDataset(data_path=args.hirid_data_dir, data_type='hirid',mode='s_train',seed=args.seed)
    loader_s_train = DataLoader(dataset_s_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    if args.mode == "attn":
        dataset_s_val = PositiveDataset(data_path=args.hirid_data_dir, data_type='hirid',mode='s_test',seed=args.seed)
        loader_s_val = DataLoader(dataset_s_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(dataset_s_val.df_num.shape)
    else:
        print('you must set the mode as attn')

    # Tuple Containing the number of unique values within each category
    card_categories = []
    for col in dataset_s_train.df_cat.columns:
        card_categories.append(dataset_s_train.df_cat[col].nunique())
        

    ## Prepare Model
    # FTT (feature extractor)
    ft_transformer_config = {
        'categories' : card_categories,      
        'num_continuous' : args.num_cont,                
        'dim' : args.dim,                                           
        'depth' : args.depth,                         
        'heads' : args.heads, 
        'dim_head' : args.dim_head,                      
        'attn_dropout' : args.attn_dropout,              
        'ff_dropout' : args.ff_dropout  
    }

    model = DANN(
        dim_feat = args.dim,
        transformer_config = ft_transformer_config                 
    ).to(device)

   
    print(f'Extracting Start....')
    print(f'Checkpoint Load....')
    model.load_state_dict(torch.load(f'{args.ckpt_dir}/Best_DANN_Transformer.pth')['model_state_dict'])
    alpha = 0
    n_correct = 0
    n_total = 0
    attn_maps = []
    with torch.no_grad():
        model.eval()

        for num_iter, batch_data in enumerate(tqdm(loader_s_val)):
            X_num, X_cat, label = batch_data
            X_num, X_cat, label = X_num.to(device), X_cat.to(device), label.to(device)
            
            attn_map, _, _, _ = model(X_cat,X_num,True,alpha)
            attn_maps.append(attn_map.cpu().numpy()) 

        final = np.concatenate(attn_maps, axis = 1)
        np.save(f'{args.result_dir}/attention_map.npy', final)
    print('Save complete')