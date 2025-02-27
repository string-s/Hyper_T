import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import InMemoryDataset, Dataset, Data
from tqdm import tqdm
from collections import defaultdict
import gc
import os
import pickle
def sort_and_save_tables(data_dir, output_dir=None):
    """
    对MIMIC-III的主要表格按SUBJECT_ID进行排序并保存
    """
    if output_dir is None:
        output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理PATIENTS表
    print("处理PATIENTS表...")
    patients = pd.read_csv(os.path.join(data_dir, 'PATIENTS.csv'))
    patients = patients.sort_values('SUBJECT_ID')
    patients.to_csv(os.path.join(output_dir, 'PATIENTS_SORTED.csv'), index=False)
    
    # 处理ADMISSIONS表 (已在前一个函数中处理)
    if not os.path.exists(os.path.join(output_dir, 'ADMISSIONS_PROCESSED.csv')):
        print("处理ADMISSIONS表...")
        admissions = pd.read_csv(os.path.join(data_dir, 'ADMISSIONS.csv'),
                                parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME'])
        admissions = admissions.sort_values(['SUBJECT_ID', 'ADMITTIME'])
        admissions.to_csv(os.path.join(output_dir, 'ADMISSIONS_SORTED.csv'), index=False)
    
    # 处理DIAGNOSES_ICD表
    print("处理DIAGNOSES_ICD表...")
    diagnoses = pd.read_csv(os.path.join(data_dir, 'DIAGNOSES_ICD.csv'))
    diagnoses = diagnoses.sort_values(['SUBJECT_ID', 'HADM_ID'])
    diagnoses.to_csv(os.path.join(output_dir, 'DIAGNOSES_ICD_SORTED.csv'), index=False)
    
    # 处理PROCEDURES_ICD表
    print("处理PROCEDURES_ICD表...")
    procedures = pd.read_csv(os.path.join(data_dir, 'PROCEDURES_ICD.csv'))
    procedures = procedures.sort_values(['SUBJECT_ID', 'HADM_ID'])
    procedures.to_csv(os.path.join(output_dir, 'PROCEDURES_ICD_SORTED.csv'), index=False)
    
    
    print("所有表格排序完成!")

def preprocess_admissions_table(data_dir, output_dir=None):
    """
    预处理ADMISSIONS.csv表，添加30天再入院标签
    """
    if output_dir is None:
        output_dir = data_dir
        
    # 读取ADMISSIONS表
    print("读取ADMISSIONS表...")
    admissions = pd.read_csv(os.path.join(data_dir, 'ADMISSIONS.csv'),
                            parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME'])
    
    # 按患者ID和入院时间排序
    print("按患者ID和入院时间排序...")
    admissions = admissions.sort_values(['SUBJECT_ID', 'ADMITTIME'])
    
    # 计算30天再入院标签
    print("计算30天再入院标签...")
    admissions['NEXT_ADMIT'] = admissions.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)
    admissions['DAYS_TO_NEXT'] = (admissions['NEXT_ADMIT'] - admissions['DISCHTIME']).dt.total_seconds() / (24 * 3600)
    admissions['READMISSION_30'] = ((admissions['DAYS_TO_NEXT'] <= 30) & (admissions['DAYS_TO_NEXT'] >= 0)).astype(int)
    
    # 填充缺失值（最后一次就诊）
    admissions['READMISSION_30'] = admissions['READMISSION_30'].fillna(0).astype(int)
    
    # 删除中间列
    admissions.drop(['NEXT_ADMIT', 'DAYS_TO_NEXT'], axis=1, inplace=True)
    
    # 保存预处理后的表
    print(f"保存预处理后的ADMISSIONS表...")
    admissions.to_csv(os.path.join(output_dir, 'ADMISSIONS_PROCESSED.csv'), index=False)
    
    # 输出样本统计
    readmit_pos = admissions['READMISSION_30'].sum()
    readmit_neg = len(admissions) - readmit_pos
    print(f"\n再入院标签统计:")
    print(f"  正样本(再入院): {readmit_pos} ({readmit_pos/len(admissions)*100:.1f}%)")
    print(f"  负样本(未再入院): {readmit_neg} ({readmit_neg/len(admissions)*100:.1f}%)")
    print(f"  总计: {len(admissions)}")
    
    return admissions

data_dir = "/root/autodl-tmp/zyh/Hyper_Time/data"
output_dir = "/root/autodl-tmp/zyh/Hyper_Time/data"
sort_and_save_tables(data_dir, output_dir)
