import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings
from tqdm import tqdm
import random

warnings.filterwarnings('ignore')


def data_split(df, seed, stayid, mode):
    
    ## patientunitstayid를 기준으로 split 진행
    ## Source는 학습에 70%, Target 학습에 30% 사용
    random.seed(seed)

    all_list = df[stayid].unique().tolist()

    if mode.startswith('s'): 
        sample_size = int(len(all_list) * 0.7)
    else:
        sample_size = int(len(all_list) * 0.3)
    
    train_unitid = random.sample(all_list,sample_size)
    df_train, df_valid  = df.query('@train_unitid in ' + stayid), df.query('@train_unitid not in ' + stayid)

    return df_train, df_valid


class TableDataset(Dataset):
    def __init__(self,data_path,data_type,mode,seed):
        self.data_path = data_path
        self.data_type = data_type # hirid or pic
        self.mode = mode # s_train / s_test / t_train / t_test
        self.target = 'ARDS_next_12h'
        self.seed = seed

        self.cat_features = ['Creatinine_up', 'Lactate_up', 'Platelet_Count_up', 'Bilirubin_up', 'INR_up', 'pH_up', 'Antibiotics', 'Vasopressor', 'Sex']
        
        self.num_features = ['Time_since_ICU_admission', 'ABPd', 'ABPs', 'FiO2', 'HR', 'MAP', 'PaO2', 'Respiratory_rate', 'SpO2',
                             'Temperature', 'Urine_Output', 'FiO2_returns', 'FiO2_max_3h', 'FiO2_min_3h', 'FiO2_5MA', 'FiO2_10MA',
                             'FiO2_20MA', 'FiO2_RSI_3h', 'PaO2_returns', 'PaO2_max_3h', 'PaO2_min_3h', 'PaO2_5MA', 'PaO2_10MA', 'PaO2_20MA',
                             'PaO2_RSI_3h', 'MAP_returns', 'MAP_max_3h', 'MAP_min_3h', 'MAP_5MA', 'MAP_10MA', 'MAP_20MA', 'MAP_RSI_3h', 
                             'Respiratory_rate_returns', 'Respiratory_rate_max_3h', 'Respiratory_rate_min_3h','Respiratory_rate_5MA',
                             'Respiratory_rate_10MA', 'Respiratory_rate_20MA', 'Respiratory_rate_RSI_3h', 'PP', 'Creatinine', 'Lactate',
                             'Platelet_Count', 'Bilirubin', 'INR', 'pH', 'Fluid_Bolus', 'PaO2/FiO2', 'Age']

        # self.num_features = ['Time_since_ICU_admission', 'ABPd', 'ABPs', 'FiO2', 'HR', 'MAP', 'PaO2', 'Respiratory_rate', 'SpO2','Temperature', 
        #                         'Urine_Output', 'FiO2_returns', 'PP', 'Creatinine', 'Lactate', 'Platelet_Count', 'Bilirubin', 'INR', 'pH', 'Fluid_Bolus', 'PaO2/FiO2', 'Age']
        
        self.df_num, self.df_cat, self.y = self.__prepare_data__()

    def __prepare_data__(self):
        df_raw = pd.read_csv(self.data_path)

        scaler = StandardScaler()

        # if dataset is hirid (source)
        if self.data_type == 'hirid':
            df_train, df_valid = data_split(df_raw,self.seed,'SUBJECT_ID',self.mode)
            X_num_standard = df_train[self.num_features]
            scaler.fit(X_num_standard)

            if self.mode == "s_train":
                X_num = df_train[self.num_features]
                X_num_scaled = scaler.transform(X_num)
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_train[self.cat_features]
                y = df_train[self.target]
                return X_num, X_cat, y
            
             # s_test mode
            else:
                X_num = df_valid[self.num_features]
                X_num_scaled = scaler.transform(X_num)
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_valid[self.cat_features]
                y = df_valid[self.target]
                return X_num, X_cat, y

        # if dataset is pic (target)
        else:
            df_train, df_valid = data_split(df_raw,self.seed,'SUBJECT_ID', self.mode)
            X_num_standard = df_train[self.num_features]
            scaler.fit(X_num_standard)

            if self.mode == "t_train":
                X_num = df_train[self.num_features]
                X_num_scaled = scaler.transform(X_num)
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_train[self.cat_features]
                y = df_train[self.target]
                return X_num, X_cat, y
            
            # t_test mode
            else : 
                ## scaler fitting을 위한 과정
                X_num = df_valid[self.num_features]
                X_num_scaled = scaler.transform(X_num)
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_valid[self.cat_features]
                y = df_valid[self.target]
                
                return X_num, X_cat, y

    def __getitem__(self,index):
                
        X_num_features = torch.tensor(self.df_num.iloc[index,:].values,dtype=torch.float32)
        X_cat_features = torch.tensor(self.df_cat.iloc[index,:].values,dtype=torch.float32).long()
        label = torch.tensor(int(self.y.iloc[index]),dtype=torch.float32).long()

        return X_num_features, X_cat_features, label
    
    def __len__(self):
        return self.y.shape[0]
