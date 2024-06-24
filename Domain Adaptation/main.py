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
    parser.add_argument("--mimic_data_dir", default="./Data/MIMIC_ARDS_12H_SPLIT.csv.gz", type=str, dest="mimic_data_dir")
    parser.add_argument('--seed', default=42, type=int , dest='seed')

    # Train Method
    parser.add_argument('--optimizer', default='AdamW', type=str, dest='optim')
    parser.add_argument("--lr", default=1e-3, type=float,dest="lr")
    parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
    parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
    parser.add_argument("--weight_decay", default=1e-5, type=float, dest="weight_decay")
    parser.add_argument("--scheduler", action='store_true', dest="scheduler", help="True or False to enable CosineAnnealing scheduler")
    parser.add_argument("--T_max", default=100, type=int, dest="T_max")
    parser.add_argument("--early_stop", action='store_true', dest="early_stop", help="True or False to enable early stop")
    parser.add_argument("--patience", default = 3, type = int, dest="patience", help="Patience for early stop")

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
    parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
    parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
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
    dataset_s_train = TableDataset(data_path=args.hirid_data_dir, data_type='hirid',mode='s_train',seed=args.seed)
    loader_s_train = DataLoader(dataset_s_train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    if args.mode == "train":
        dataset_s_val = TableDataset(data_path=args.hirid_data_dir, data_type='hirid',mode='s_test',seed=args.seed)
        loader_s_val = DataLoader(dataset_s_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

        print(f'Build Target Domain Dataset : {args.mimic_data_dir} ....')
        dataset_t_train = TableDataset(data_path=args.mimic_data_dir, data_type='mimic',mode='t_train',seed=args.seed)
        loader_t_train = DataLoader(dataset_t_train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    else:
        print('Inference start....')
        print(f'Build Target Domain Dataset : {args.mimic_data_dir} ....')
        dataset_t_test = TableDataset(data_path=args.mimic_data_dir, data_type='mimic',mode='t_test',seed=args.seed)
        loader_t_test = DataLoader(dataset_t_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

        dataset_s_val = TableDataset(data_path=args.hirid_data_dir, data_type='hirid',mode='s_test',seed=args.seed)
        loader_s_val = DataLoader(dataset_s_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

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

    # DANN (label predictor, domain classifier)

    # setup optimizer
    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9 , weight_decay= args.weight_decay)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.T_max)

    if args.early_stop:
        patience = args.patience
        early_stop_counter = 0

    # loss_class = torch.nn.CrossEntropyLoss().to(device)
    # loss_domain = torch.nn.CrossEntropyLoss().to(device)

    loss_class = focal_loss(alpha = [1,2], gamma = 2).to(device)
    loss_domain = torch.nn.CrossEntropyLoss().to(device)
    
    ## Model Train and Eval
    start_epoch = 0
    Best_valid_loss = 1e9

    ## Train mode
    if args.mode == 'train':

        ## Tensorboard Setting
        writer_train = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train'))
        writer_val = SummaryWriter(log_dir=os.path.join(args.log_dir, 'val'))

        print(f'Train Start....')
        for epoch in range(start_epoch + 1, args.num_epoch + 1):
            model.train()
            running_loss = 0
            s_correct = 0
            n_total = 0
            precision_train = 0
            recall_train = 0

            writer_train.add_scalar('Train_Epoch_LR',optimizer.param_groups[0]["lr"],epoch)

            for num_iter, (s_batch_data,t_batch_data) in enumerate((zip(loader_s_train, (loader_t_train)))):
                
                len_dataloader = min(len(loader_s_train), len(loader_t_train))
                p = float(num_iter + epoch * len_dataloader) / args.num_epoch / len(loader_s_train)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                ### 1. training model using source domain data ###
                optimizer.zero_grad()
                s_X_num, s_X_cat, s_label = s_batch_data
                s_X_num, s_X_cat, s_label = s_X_num.to(device), s_X_cat.to(device), s_label.to(device)


                domain_label = torch.zeros(len(s_label)).long()
                domain_label = domain_label.to(device)
                
                s_attn_map, s_latent_vector, class_output, domain_output = model(s_X_cat,s_X_num,True, alpha)
                err_s_label = loss_class(class_output, s_label)
                err_s_domain = loss_domain(domain_output, domain_label)

                ### 2. training model using target domain data ###
                t_X_num, t_X_cat, _ = t_batch_data # Not use Target Domain Label
                t_X_num, t_X_cat = t_X_num.to(device), t_X_cat.to(device)
        
                
                domain_label = torch.ones(len(t_X_num)).long() 
                domain_label = domain_label.to(device)
                
                t_attn_map, t_latent_vector, _, domain_output = model(t_X_cat,t_X_num,True,alpha)
                err_t_domain = loss_domain(domain_output, domain_label)

                ### backward pass ###
                err = err_t_domain + err_s_domain + err_s_label
                err.backward()
                optimizer.step()
                
                ### Check Source Train Accuracy ###

                pred = class_output.detach().cpu().max(1, keepdim=True)[1]
                # s_correct += pred.eq(s_label.detach().view_as(pred)).cpu().sum()
                # n_total += len(s_label)
                precision_train += precision_score(s_label.detach().cpu(),pred)
                recall_train += recall_score(s_label.detach().cpu(),pred)
                # accu_train = s_correct.detach().numpy() * 1.0 / n_total

                running_loss += err.item()
                
                if num_iter % args.printiter == 0:
                    print("\nTRAIN: EPOCH %04d / %04d | ITER %04d / %04d | LOSS %.4f | err_s_label: %.4f, err_s_domain: %.4f, err_t_domain: %.4f | Source Train Precision : %.4f\n | Source Train Recall : %.4f"\
                        % (epoch, args.num_epoch, num_iter+1, len_dataloader, err.detach().cpu().numpy(), err_s_label.detach().cpu().item(), \
                        err_s_domain.detach().cpu().item(), err_t_domain.detach().cpu().item(), precision_train/(num_iter+1), recall_train/(num_iter+1)))
                    writer_train.add_scalar('Train_Iter_Classifier_Loss', err_s_label.detach().cpu().item(), (len_dataloader * epoch) + num_iter)
                    writer_train.add_scalar('Train_Iter_Domain_Loss',err_s_domain.detach().cpu().item() + err_t_domain.detach().cpu().item() , (len_dataloader * epoch) + num_iter)
                
            print(f'Epoch{epoch} / {args.num_epoch} Train Loss : {running_loss / len_dataloader} | Train Precision : {precision_train/len_dataloader} | Train Recall : {recall_train/len_dataloader}')

            # Schedluer Step
            if args.scheduler:
                scheduler.step()

            writer_train.add_scalar('Train_Epoch_loss', running_loss / len(loader_s_train), epoch)
            writer_train.add_scalar('Train_Epoch_Precision', precision_train/len_dataloader, epoch)
            print(f'---------Epoch{epoch} Training Finish---------')

            ### evaluation : Source Test Accuracy (사실 의미가 크지 않은 지표) ### 
            with torch.no_grad():
                model.eval()
                running_loss = 0
                n_correct = 0
                n_total = 0
                alpha = 0
                precision_valid = 0
                recall_valid = 0
                running_loss = 0
            
                for num_iter, batch_data in enumerate(tqdm(loader_s_val)):
                    X_num, X_cat, label = batch_data
                    # label = label.type(torch.LongTensor)
                    X_num, X_cat, label = X_num.to(device), X_cat.to(device), label.to(device)
                    
                    attn_map, latent_vector, class_output, _ = model(X_cat,X_num,True,alpha)
                    err_s_label = loss_class(class_output, label)
                    
                    running_loss += err_s_label.item()
                    pred = class_output.detach().cpu().max(1, keepdim=True)[1]
                    # n_correct += pred.eq(label.detach().view_as(pred)).cpu().sum()
                    
                    # n_total += len(label)
                    precision_valid += precision_score(label.detach().cpu(),pred)
                    recall_valid += recall_score(label.detach().cpu(),pred)
                # accu_valid = n_correct.detach().numpy() * 1.0 / n_total
                
                if num_iter % args.printiter == 0:
                    print("\VALID: EPOCH %04d / %04d | ITER %04d / %04d | LOSS %.4f | Source Test Precision : %.4f\n | Source Test Recall" \
                        % (epoch, args.num_epoch, num_iter+1, len(loader_s_val), err_s_label.detach().cpu().item(), precision_valid/(num_iter + 1), recall_valid/(num_iter + 1)))
                    
            print(f'Epoch{epoch} / {args.num_epoch} Valid Loss : {running_loss / len(loader_s_val)} | Valid Precision : {precision_valid / len(loader_s_val)} | Valid Recall : {recall_valid / len(loader_s_val)}')

            writer_val.add_scalar('Valid_Epoch_loss', running_loss / len(loader_s_val), epoch)
            writer_val.add_scalar('Valid_Epoch_Precision', precision_valid / len(loader_s_val), epoch)

            # early stop checking
            if early_stop_counter == patience:
                break

            if running_loss / len(loader_s_val) < Best_valid_loss:
                print(f'Best Loss {Best_valid_loss:.4f} -> {running_loss / len(loader_s_val):.4f} Update! & Save Checkpoint')
                Best_valid_loss = running_loss / len(loader_s_val)
                early_stop_counter = 0

                if args.scheduler:
                    torch.save({'model_state_dict' : model.state_dict(),
                                'optimizer_state_dict' : optimizer.state_dict(),
                                'scheduler_state_dict' : scheduler.state_dict()},f'{args.ckpt_dir}/Best_DANN_Transformer.pth')
                else:
                    torch.save({'model_state_dict' : model.state_dict(),
                                'optimizer_state_dict' : optimizer.state_dict()},f'{args.ckpt_dir}/Best_DANN_Transformer.pth')
            else:
                early_stop_counter += 1
            
            # each epoch save
            if args.scheduler:
                torch.save({'model_state_dict' : model.state_dict(),
                            'optimizer_state_dict' : optimizer.state_dict(),
                            'scheduler_state_dict' : scheduler.state_dict()},f'{args.ckpt_dir}/Epoch{epoch}_DANN_Transformer.pth')
            else:
                torch.save({'model_state_dict' : model.state_dict(),
                            'optimizer_state_dict' : optimizer.state_dict()},f'{args.ckpt_dir}/Epoch{epoch}_DANN_Transformer.pth')

            
            print(f'---------Epoch{epoch} Valid Finish---------')

        writer_train.close()
        writer_val.close()

    ### Inference(Test) mode : Target Test Accuracy ###
    else:
        print(f'Inference Start....')
        print(f'Checkpoint Load....')
        model.load_state_dict(torch.load(f'{args.ckpt_dir}/Best_DANN_Transformer.pth')['model_state_dict'])
        alpha = 0
        n_correct = 0
        n_total = 0
        
        with torch.no_grad():
            model.eval()
            running_loss = 0
            corr = 0
            output_list = []
            pos_pred_list = []
            label_list = []
            pred_list = []
            feature_test = pd.DataFrame()


            for num_iter, batch_data in enumerate(tqdm(loader_t_test)):
                X_num, X_cat, label = batch_data
                X_num, X_cat, label = X_num.to(device), X_cat.to(device), label.to(device)
                
                attn_map, latent_vector, class_output, _ = model(X_cat,X_num,True,alpha) # don't require domain output for test
                pred = class_output.detach().max(1, keepdim=True)[1]
                n_correct += pred.eq(label.detach().view_as(pred)).cpu().sum()
                n_total += len(label)
                
                # print(pred.detach().cpu().numpy().reshape(-1).shape)
                pred_list += list(pred.detach().cpu().numpy().reshape(-1))
                label_list += list(label.detach().cpu().numpy().reshape(-1))
                pos_pred_list += list(F.softmax(class_output).detach().cpu().numpy()[:,1]) # Positive Confidence Score

                feature = pd.DataFrame(np.array(latent_vector.detach().cpu().numpy()))
                feature_test = pd.concat([feature_test, feature], axis = 0)
            
            # print(len(label_list))
            # print(len(pred_list))
            accu = n_correct.detach().numpy() * 1.0 / n_total
            precision = precision_score(label_list, pred_list, average='macro')
            recall = recall_score(label_list, pred_list, average='macro')
            f1 = f1_score(label_list, pred_list, average='macro')
            confusion_matrices = confusion_matrix(label_list, pred_list)
            roc_auc = roc_auc_score(label_list, pred_list)

            result_csv = pd.DataFrame({'Positive_Confscore' : pos_pred_list})
            result_csv.to_csv(f'{args.result_dir}/DA_Target_test_result.csv',index = False)
            feature_test.to_csv(f'{args.result_dir}/DA_Target_test_feature.csv',index = False)
            print('--- Target Test Result ---')
            print(f'Accuracy : {accu:.4f} | Precision : {precision:.4f} | Recall : {recall:.4f} | \n f1 : {f1:.4f} | roc_auc : {roc_auc:.4f}')
            print(f'Confusion Matrix : {confusion_matrices}')

            running_loss = 0
            corr = 0
            output_list = []
            pos_pred_list = []
            label_list = []
            pred_list = []
            feature_valid = pd.DataFrame()

            for num_iter, batch_data in enumerate(tqdm(loader_s_val)):
                X_num, X_cat, label = batch_data
                X_num, X_cat, label = X_num.to(device), X_cat.to(device), label.to(device)
                
                attn_map, latent_vector, class_output, _ = model(X_cat,X_num,True,alpha) # don't require domain output for test
                pred = class_output.detach().max(1, keepdim=True)[1]
                n_correct += pred.eq(label.detach().view_as(pred)).cpu().sum()
                n_total += len(label)

                pred_list += list(pred.detach().cpu().numpy().reshape(-1))
                label_list += list(label.detach().cpu().numpy().reshape(-1))
                pos_pred_list += list(F.softmax(class_output).detach().cpu().numpy()[:,1]) # Positive Confidence Score

                feature = pd.DataFrame(np.array(latent_vector.detach().cpu().numpy()))
                feature_valid = pd.concat([feature_valid, feature], axis = 0)
                
            # final = np.concatenate(attn_map, axis = 1)
            # print(len(label_list))
            # print(len(pred_list))
            accu = n_correct.detach().numpy() * 1.0 / n_total
            precision = precision_score(label_list, pred_list, average='binary')
            recall = recall_score(label_list, pred_list, average='binary')
            f1 = f1_score(label_list, pred_list, average='binary')
            confusion_matrices = confusion_matrix(label_list, pred_list)
            roc_auc = roc_auc_score(label_list, pred_list)

            result_csv = pd.DataFrame({'Positive_Confscore' : pos_pred_list})
            result_csv.to_csv(f'{args.result_dir}/DA_Source_valid_result.csv',index = False)
            feature_valid.to_csv(f'{args.result_dir}/DA_Source_valid_feature.csv',index = False)
            print('--- Source Valid Result ---')
            print(f'Accuracy : {accu:.4f} | Precision : {precision:.4f} | Recall : {recall:.4f} | \n f1 : {f1:.4f} | roc_auc : {roc_auc:.4f}')
            print(f'Confusion Matrix : {confusion_matrices}')