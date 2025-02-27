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

class MIMIC3Processor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.code_encoder = None
        self._load_tables_metadata()  # 只加载元数据，不加载全部内容
        
    def _load_tables_metadata(self):
        """只加载表格结构和少量样本，不加载全部数据"""
        # 获取列名和数据类型
        self.admissions_df = pd.read_csv(os.path.join(self.data_dir, 'ADMISSIONS.csv'), 
                                            nrows=10).dtypes
        self.patients_df = pd.read_csv(os.path.join(self.data_dir, 'PATIENTS.csv'), 
                                          nrows=10).dtypes
        self.diagnoses_df = pd.read_csv(os.path.join(self.data_dir, 'DIAGNOSES_ICD.csv'), 
                                           nrows=10).dtypes
        self.procedures_df = pd.read_csv(os.path.join(self.data_dir, 'PROCEDURES_ICD.csv'), 
                                            nrows=10).dtypes
        
        
        # 获取数据行数以便分批处理
        self.admissions_count = sum(1 for _ in open(os.path.join(self.data_dir, 'ADMISSIONS.csv'))) - 1
        self.patients_count = sum(1 for _ in open(os.path.join(self.data_dir, 'PATIENTS.csv'))) - 1
        self.diagnoses_count = sum(1 for _ in open(os.path.join(self.data_dir, 'DIAGNOSES_ICD.csv'))) - 1
        self.procedures_count = sum(1 for _ in open(os.path.join(self.data_dir, 'PROCEDURES_ICD.csv'))) - 1
    
    def load_tables_batch(self, batch_start, batch_size):
        """分批加载数据"""
        # 加载Admissions批次
        self.admissions = pd.read_csv(os.path.join(self.data_dir, 'ADMISSIONS.csv'),
                                     parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME'],
                                     skiprows=range(1, batch_start + 1) if batch_start > 0 else None,
                                     nrows=batch_size)
        self.admissions['HOSPITAL_EXPIRE_FLAG'] = self.admissions['HOSPITAL_EXPIRE_FLAG'].astype(np.int8)
        
        # 根据当前批次的SUBJECT_ID加载相关患者数据
        subject_ids = self.admissions['SUBJECT_ID'].unique()
        self.patients = pd.read_csv(os.path.join(self.data_dir, 'PATIENTS.csv'),
                                   parse_dates=['DOB', 'DOD'])
        self.patients = self.patients[self.patients['SUBJECT_ID'].isin(subject_ids)]
        self.patients['GENDER'] = self.patients['GENDER'].astype('category')
        
        # 根据当前批次的HADM_ID加载相关诊断和手术数据
        hadm_ids = self.admissions['HADM_ID'].unique()
        self.diagnoses = pd.read_csv(os.path.join(self.data_dir, 'DIAGNOSES_ICD.csv'))
        self.diagnoses = self.diagnoses[self.diagnoses['HADM_ID'].isin(hadm_ids)]
        self.diagnoses['ICD9_CODE'] = self.diagnoses['ICD9_CODE'].astype('category')
        
        self.procedures = pd.read_csv(os.path.join(self.data_dir, 'PROCEDURES_ICD.csv'))
        self.procedures = self.procedures[self.procedures['HADM_ID'].isin(hadm_ids)]
        self.procedures['ICD9_CODE'] = self.procedures['ICD9_CODE'].astype('category')
        
        gc.collect()
        return len(self.admissions)
    
    def _process_codes(self):
        """处理医疗代码（诊断+手术）"""
        # 诊断代码处理 - 只加载需要的数据
        d_icd_diag = pd.read_csv(os.path.join('/root/autodl-tmp/MIMI-CIII/D_ICD_DIAGNOSES.csv'))
        diag = self.diagnoses.merge(d_icd_diag, on='ICD9_CODE', how='left')
        diag['code'] = 'DIAG_' + diag['ICD9_CODE'].astype(str)
        diag = diag[['HADM_ID', 'code']]
        del d_icd_diag
        gc.collect()
        
        # 手术代码处理 - 只加载需要的数据
        d_icd_proc = pd.read_csv(os.path.join('/root/autodl-tmp/MIMI-CIII/D_ICD_PROCEDURES.csv'))
        proc = self.procedures.merge(d_icd_proc, on='ICD9_CODE', how='left')
        proc['code'] = 'PROC_' + proc['ICD9_CODE'].astype(str)
        proc = proc[['HADM_ID', 'code']]
        del d_icd_proc
        gc.collect()
        
        # 合并代码
        codes = pd.concat([diag, proc])
        del diag, proc
        gc.collect()

        codes = codes.groupby('HADM_ID')['code'].apply(list).reset_index()
        return codes

    def _generate_labels(self, df):
        """生成每个HADM_ID的标签"""
        # 院内死亡率（直接使用HOSPITAL_EXPIRE_FLAG）
        df['MORTALITY'] = df['HOSPITAL_EXPIRE_FLAG'].astype(np.int8)

        # 30天再入院标签计算
        df['READMISSION'] = df['READMISSION_30'].astype(np.int8)

        df.drop(['NEXT_ADMIT', 'DAYS_TO_NEXT'], axis=1, errors='ignore', inplace=True)
        gc.collect()

        return df

    def build_code_encoder(self, save_path=None):
        """构建全局代码编码器 - 分批处理"""
        all_codes = set(['UNK'])  # 添加未知代码
        batch_size = 10000
        
        for batch_start in range(0, self.admissions_count, batch_size):
            self.load_tables_batch(batch_start, batch_size)
            codes_df = self._process_codes()
            
            # 收集所有代码
            batch_codes = [code for sublist in codes_df['code'].dropna() for code in sublist]
            all_codes.update(batch_codes)
            
            # 释放内存
            del codes_df, batch_codes
            gc.collect()
        
        # 构建编码器
        self.code_encoder = LabelEncoder().fit(list(all_codes))
        
        # 保存编码器以便复用
        if save_path:
            with open(os.path.join(save_path, 'code_encoder.pkl'), 'wb') as f:
                pickle.dump(self.code_encoder, f)
        
        return self.code_encoder
    
    def load_code_encoder(self, path):
        """加载预训练的代码编码器"""
        with open(path, 'rb') as f:
            self.code_encoder = pickle.load(f)
        return self.code_encoder
    
    def _encode_data(self, df):
        """编码医疗代码"""
        if self.code_encoder is None:
            raise ValueError("Code encoder not initialized. Call build_code_encoder first.")

        def encode_visit(codes):
            if not isinstance(codes, list):
                return []
            return [self.code_encoder.transform([c])[0] 
                    if c in self.code_encoder.classes_ else 0
                    for c in codes]

        df['encoded_codes'] = df['code'].apply(encode_visit)
        return df

    def filter_by_disease(self, df, prefix='I'):
        """根据疾病代码前缀筛选数据"""
        def has_disease(codes):
            if not isinstance(codes, list):
                return False
            for code in codes:
                if isinstance(code, str):
                    if code.startswith(f"DIAG_{prefix}"):
                        return True
                    elif code.startswith("PROC_") and code[5:].startswith(prefix):
                        return True
            return False

        has_disease_series = df['code'].apply(has_disease)
        filtered_df = df[has_disease_series].copy()
        del has_disease_series
        gc.collect()
        return filtered_df


class MIMIC3StreamDataset(Dataset):
    """流式数据集，支持增量训练"""
    def __init__(self, root, processor, batch_size=1000, transform=None, pre_transform=None):
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.processor = processor
        self.batch_size = batch_size
        
        # 确保处理目录存在
        self.processed_dir_path = os.path.join(root, 'processed')
        os.makedirs(self.processed_dir_path, exist_ok=True)
        
        # 计算总批次数
        self.total_batches = (self.processor.admissions_count + self.batch_size - 1) // self.batch_size
        
        # 初始化其他属性
        self.processed_batches = []
        self.current_data = None
        self.current_batch_idx = 0
        
        # 加载或构建代码编码器 - 确保这一步在任何数据处理之前完成
        encoder_path = os.path.join(self.processed_dir_path, 'code_encoder.pkl')
        if os.path.exists(encoder_path):
            print(f"加载现有的代码编码器: {encoder_path}")
            self.processor.load_code_encoder(encoder_path)
        else:
            print(f"构建新的代码编码器并保存到: {encoder_path}")
            self.processor.build_code_encoder()
            # 确保编码器被保存
            os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.processor.code_encoder, f)
        
        # 检查已处理的批次
        for i in range(self.total_batches):
            if os.path.exists(os.path.join(self.processed_dir_path, f'data_batch_{i}.pt')):
                self.processed_batches.append(i)
        
        # 避免调用父类的 __init__ 方法，因为它会触发 _process() 方法
        # 我们将自己实现必要的方法
    
    @property
    def processed_dir(self):
        return self.processed_dir_path
    
    @property
    def processed_file_names(self):
        return [f'data_batch_{i}.pt' for i in range(self.total_batches)] + ['code_encoder.pkl']
    
    def len(self):
        return self.processor.admissions_count
    
    def get(self, idx):
        # 原有的 get 方法实现
        batch_idx = idx // self.batch_size
        if batch_idx != self.current_batch_idx or self.current_data is None:
            self.load_batch(batch_idx)
        local_idx = idx % self.batch_size
        if local_idx < len(self.current_data):
            return self.current_data[local_idx]
        else:
            # 处理边界情况
            return self.current_data[-1]  # 返回批次中的最后一个数据
    
    def process(self):
        # 处理所有批次
        for batch_idx in range(self.total_batches):
            if batch_idx not in self.processed_batches:
                self.process_batch(batch_idx)
    
    # 为每个样本单独创建边索引
    def create_edge_index(self, num_nodes):
        if num_nodes <= 0:
            return torch.zeros((2, 0), dtype=torch.long)
        
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # 不包括自环
                    edges.append([i, j])  
        if not edges:
            edges = [[0, 0]]  # 如果没有边，添加自环       
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def process_batch(self, batch_idx):
        """处理单个批次的数据"""
        print(f"处理批次 {batch_idx}/{self.total_batches}")
        
        # 加载批次数据
        start_idx = batch_idx * self.batch_size
        self.processor.load_tables_batch(start_idx, self.batch_size)
        
        # 确保代码编码器已初始化
        if not hasattr(self.processor, 'code_encoder') or self.processor.code_encoder is None:
            raise ValueError("Code encoder not initialized. This should not happen.")
        
        # 处理数据
        data_list = []
        
        # 使用load_tables_batch方法加载的数据，而不是_load_tables_metadata中的数据
        admissions_df = self.processor.admissions  # 注意这里使用admissions而不是admissions_df
        patients_df = self.processor.patients
        diagnoses_df = self.processor.diagnoses
        procedures_df = self.processor.procedures
        
        # 按患者ID分组处理
        for subject_id, adm_group in admissions_df.groupby('SUBJECT_ID'):
            # 按入院时间排序
            adm_group = adm_group.sort_values('ADMITTIME')
            
            # 获取患者基本信息 - 添加安全检查
            patient_matches = patients_df[patients_df['SUBJECT_ID'] == subject_id]
            if patient_matches.empty:
                print(f"警告: 跳过患者ID {subject_id}，在患者数据中未找到")
                continue
                
            patient_info = patient_matches.iloc[0]
            
            for _, admission in adm_group.iterrows():
                hadm_id = admission['HADM_ID']
                
                # 获取诊断和手术代码
                diag_codes = diagnoses_df[diagnoses_df['HADM_ID'] == hadm_id]['ICD9_CODE'].dropna().tolist()
                proc_codes = procedures_df[procedures_df['HADM_ID'] == hadm_id]['ICD9_CODE'].dropna().tolist()
                
                # 编码诊断和手术代码
                diag_indices = []
                for code in diag_codes:
                    if isinstance(code, str) and code in self.processor.code_encoder.classes_:
                        diag_indices.append(self.processor.code_encoder.transform([code])[0])
                    else:
                        diag_indices.append(0)  # 默认索引
                
                proc_indices = []
                for code in proc_codes:
                    if isinstance(code, str) and code in self.processor.code_encoder.classes_:
                        proc_indices.append(self.processor.code_encoder.transform([code])[0])
                    else:
                        proc_indices.append(0)  # 默认索引
                
                # 确保索引不为空
                if not diag_indices:
                    diag_indices = [0]  # 使用默认索引
                if not proc_indices:
                    proc_indices = [0]  # 使用默认索引
                
                # 创建节点特征
                num_nodes = len(diag_indices) + len(proc_indices)
                
                # 确保节点数量大于0
                if num_nodes == 0:
                    continue
                edge_index = self.create_edge_index(num_nodes)
                
                # 创建节点特征矩阵 - 使用整数索引，而不是浮点数零张量
                # 将所有代码索引合并为一个列表
                all_indices = diag_indices + proc_indices
                # 将索引转换为整数张量，形状为 (num_nodes, 1)
                x = torch.tensor(all_indices, dtype=torch.long).view(-1, 1)
                
                # 创建节点类型
                node_type = torch.zeros(num_nodes, dtype=torch.long)
                node_type[len(diag_indices):] = 1  # 0表示诊断，1表示手术
                
                # 准备标签
                # 1. 处理死亡率标签 - 如果存在HOSPITAL_EXPIRE_FLAG则使用，否则默认为0
                mortality_flag = 0
                if 'HOSPITAL_EXPIRE_FLAG' in admission:
                    mortality_flag = int(admission['HOSPITAL_EXPIRE_FLAG'])
                
                # 2. 处理30天再入院标签
                readmit_flag = 0
                if 'READMIT_30' in admission:
                    readmit_flag = 1 if admission['READMIT_30'] else 0
                elif 'READMISSION' in admission:
                    readmit_flag = int(admission['READMISSION'])
                else:
                    # 如果需要计算再入院标签，这里可以添加计算逻辑
                    # 但为简化起见，现在使用默认值0
                    pass
                
                # 创建PyG数据对象
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    node_type=node_type,
                    diag_indices=torch.tensor(diag_indices, dtype=torch.long),
                    proc_indices=torch.tensor(proc_indices, dtype=torch.long),
                    y_readmit=torch.tensor([readmit_flag], dtype=torch.float),
                    y_mortality=torch.tensor([mortality_flag], dtype=torch.float),
                    hadm_id=hadm_id
                )
                
                # 为了与后续处理保持一致，也添加合并的y标签
                data.y = torch.tensor([[readmit_flag, mortality_flag]], dtype=torch.float)
                
                # 添加时间信息
                if 'ADMITTIME' in admission and 'DISCHTIME' in admission:
                    admit_time = pd.to_datetime(admission['ADMITTIME'])
                    disch_time = pd.to_datetime(admission['DISCHTIME'])
                    los_days = (disch_time - admit_time).total_seconds() / (24 * 3600)
                    data.los = torch.tensor([los_days], dtype=torch.float)
                
                data_list.append(data)
        
        # 保存处理后的批次
        batch_file = os.path.join(self.processed_dir_path, f'data_batch_{batch_idx}.pt')
        torch.save(data_list, batch_file)
        self.processed_batches.append(batch_idx)
        
        # 返回处理后的数据
        self.current_data = data_list
        self.current_batch_idx = batch_idx

        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return data_list
        

    
    def _build_local_graph(self, df):
        """为当前批次构建局部图"""
        edge_index = []
        edge_attr = []
        
        # 构建代码字典，记录每个代码出现的时间
        code_dict = defaultdict(list)
        for _, row in df.iterrows():
            if not isinstance(row['encoded_codes'], list):
                continue
            curr_time = row['ADMITTIME'].timestamp()
            for code in row['encoded_codes']:
                code_dict[code].append(curr_time)
        
        # 按患者构建边
        all_patients = df.groupby('SUBJECT_ID')
        for _, patient in all_patients:
            visits = patient.sort_values('ADMITTIME')
            prev_codes = None
            prev_time = None
            
            for _, visit in visits.iterrows():
                curr_codes = list(set(visit['encoded_codes']))
                curr_time = visit['ADMITTIME']
                
                # 共现边（同一次就诊内的代码关联）
                for i in range(len(curr_codes)):
                    for j in range(i+1, len(curr_codes)):
                        edge_index.append([curr_codes[i], curr_codes[j]])
                        edge_attr.append(1.0)  # 同一次就诊的关联强度为1
                        # 添加反向边
                        edge_index.append([curr_codes[j], curr_codes[i]])
                        edge_attr.append(1.0)
                
                # 时序边（不同就诊之间的代码关联）
                if prev_codes is not None and prev_time is not None:
                    delta_t = (curr_time - prev_time).total_seconds()/(3600*24)
                    decay = np.exp(-delta_t / 365)  # 时间衰减因子
                    for src in prev_codes:
                        for dst in curr_codes:
                            edge_index.append([src, dst])
                            edge_attr.append(decay)
                
                prev_codes = curr_codes
                prev_time = curr_time
        
        if len(edge_index) == 0:  # 处理空图的情况
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        return edge_index, edge_attr
    
    def process(self):
        """处理所有批次（可选，用于预处理）"""
        for batch_idx in range(self.total_batches):
            if batch_idx not in self.processed_batches:
                self.process_batch(batch_idx)


class MIMIC3BatchDataset(InMemoryDataset):
    def __init__(self, root, processor, batch_indices=None):
        self.processor = processor
        self.batch_indices = batch_indices  # 指定要加载的批次索引
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ['combined_data.pt']
    
    def process(self):
        # 加载已处理的批次并合并
        stream_dataset = MIMIC3StreamDataset(self.root, self.processor)
        
        if self.batch_indices is None:
            # 加载所有已处理的批次
            self.batch_indices = stream_dataset.processed_batches
        
        all_data = []
        for batch_idx in self.batch_indices:
            batch_file = os.path.join(stream_dataset.processed_dir_path, f'data_batch_{batch_idx}.pt')
            if os.path.exists(batch_file):
                batch_data = torch.load(batch_file)
                all_data.extend(batch_data)
        
        if not all_data:
            raise ValueError("No processed data found. Process batches first using MIMIC3StreamDataset.")
        
        data, slices = self.collate(all_data)
        torch.save((data, slices), self.processed_paths[0])


# 增量训练辅助函数
def incremental_train(model, optimizer, stream_dataset, batch_size=32, epochs_per_batch=5):
    """对流式数据集进行增量训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 遍历所有批次
    for batch_idx in stream_dataset.processed_batches:
        print(f"训练批次 {batch_idx}/{stream_dataset.total_batches}")
        
        # 加载当前批次数据
        batch_file = os.path.join(stream_dataset.processed_dir_path, f'data_batch_{batch_idx}.pt')
        batch_data = torch.load(batch_file)
        
        # 创建数据加载器
        from torch_geometric.loader import DataLoader
        loader = DataLoader(batch_data, batch_size=batch_size, shuffle=True)
        
        # 训练当前批次
        for epoch in range(epochs_per_batch):
            model.train()
            total_loss = 0
            
            for data in loader:
                data = data.to(device)
                optimizer.zero_grad()
                
                # 前向传播（根据模型接口调整）
                readmit_pred, mortal_pred = model(data)
                
                # 计算损失
                import torch.nn.functional as F
                loss = F.binary_cross_entropy_with_logits(readmit_pred, data.y[:,0]) + \
                       F.binary_cross_entropy_with_logits(mortal_pred, data.y[:,1])
                
                # 反向传播
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"  Epoch {epoch+1}/{epochs_per_batch}, Loss: {total_loss/len(loader):.4f}")
        
        # 保存增量训练的模型
        torch.save(model.state_dict(), os.path.join(stream_dataset.processed_dir_path, f'model_after_batch_{batch_idx}.pt'))
        
        # 释放内存
        del batch_data, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return model

