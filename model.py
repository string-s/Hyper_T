import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from dhg import Hypergraph
from dhg.nn import HGNNConv
import numpy as np
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Parameter
from math import sqrt

class TGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, time_dim=64, heads=1):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.heads = heads

        # 时间编码器（将时间差编码为向量）
        self.time_enc = Linear(1, time_dim)
        
        # 注意力权重参数
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_time = Parameter(torch.Tensor(1, heads, time_dim))
        
        # 特征变换矩阵
        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)
        torch.nn.init.xavier_uniform_(self.att_time)

    def forward(self, x, edge_index, edge_time):
        # 输入处理

        x = self.lin(x).view(-1, self.heads, self.out_channels)
        num_nodes = x.size(0)

        # 时间编码
        delta_t = edge_time.unsqueeze(-1)  # [E, 1]
        time_feat = self.time_enc(delta_t).view(-1, self.heads, self.time_dim)  # [E, H, time_dim]

        # 计算注意力系数
        alpha_src = (x[edge_index[0]] * self.att_src).sum(dim=-1)  # [E, H]
        alpha_dst = (x[edge_index[1]] * self.att_dst).sum(dim=-1)  # [E, H]
        alpha_time = (time_feat * self.att_time).sum(dim=-1)  # [E, H]
        
        alpha = (alpha_src + alpha_dst + alpha_time) / sqrt(self.out_channels)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = torch.exp(-alpha)  # 时间衰减因子（值越小，权重越低）
        alpha = alpha / alpha.sum(dim=0, keepdim=True)  # 归一化

        # 消息传播
        out = self.propagate(edge_index, x=x, alpha=alpha)
        return out.mean(dim=1)  # 多头注意力聚合

    def message(self, x_j, alpha):
        return x_j * alpha.unsqueeze(-1)  # [E, H, out_dim] * [E, H, 1]

class TGNN(nn.Module):
    def __init__(self, num_codes, emb_dim=128, time_dim=64):
        super().__init__()
        self.emb = nn.Embedding(num_codes, emb_dim)
        self.time_enc = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, emb_dim)
        )
        
        self.tgnn_layers = nn.ModuleList([
            TGATConv(emb_dim, emb_dim*2, time_enc=False),  # 自定义时间编码
            TGATConv(emb_dim*2, emb_dim, time_enc=False)
        ])
        
    def forward(self, x, edge_index, edge_time):
        x = self.emb(x)
        time_feat = self.time_enc(edge_time.unsqueeze(-1))  # [E, emb_dim]
        
        for i, layer in enumerate(self.tgnn_layers):
            x_res = x
            x = layer(x, edge_index, edge_attr=time_feat)
            x = F.leaky_relu(x, 0.2)
            if i % 2 == 1:  # 残差连接
                x = x + x_res
        return x

class HyperGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.hgnn = HGNNConv(in_channels, out_channels)
        
        if use_attention:
            # 超边注意力权重
            self.att = nn.Sequential(
                nn.Linear(out_channels, 1),
                nn.Sigmoid()
            )

    def forward(self, hg, x):
        # 标准超图卷积
        x = self.hgnn(hg, x)
        
        if self.use_attention:
            # 计算超边注意力
            edge_features = hg.edge_features(x)  # [num_edges, out_channels]
            edge_weights = self.att(edge_features)  # [num_edges, 1]
            
            # 应用注意力权重
            x = hg.propagate(x, aggr="mean", edge_weight=edge_weights)
            
        return x

class DynamicHyperGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, time_dim=16):
        super().__init__()
        self.hyper_layers = nn.ModuleList([
            HyperGATConv(in_dim + time_dim, hidden_dim),
            HyperGATConv(hidden_dim, hidden_dim)
        ])
        self.time_proj = nn.Linear(1, time_dim)
        
    def forward(self, X, hyperedges, visit_times):
        # 添加时间特征
        time_feat = self.time_proj(torch.tensor(visit_times).float().unsqueeze(-1))
        X_aug = torch.cat([X, time_feat.repeat(X.size(0),1)], dim=1)
        
        # 动态超图构建
        HG = Hypergraph(X.size(0), self._build_dynamic_edges(hyperedges, visit_times))
        
        for layer in self.hyper_layers:
            X_aug = layer(HG, X_aug)
            X_aug = F.leaky_relu(X_aug, 0.2)
        return X_aug
    
    def _build_dynamic_edges(self, edges, times):
        """根据时间间隔调整超边权重"""
        weighted_edges = []
        max_time = max(times)
        for edge, t in zip(edges, times):
            decay = np.exp(-(max_time - t)/365)  # 时间衰减因子
            weighted_edges.append((list(edge), decay))
        return weighted_edges

class MoCoContrastiveLearner(nn.Module):
    def __init__(self, feat_dim=128, queue_size=4096, temp=0.07):
        super().__init__()
        # 查询编码器
        self.encoder_q = nn.Sequential(
            nn.Linear(feat_dim, feat_dim*2),
            nn.ReLU(),
            nn.Linear(feat_dim*2, feat_dim)
        )
        # 动量编码器
        self.encoder_k = nn.Sequential(
            nn.Linear(feat_dim, feat_dim*2),
            nn.ReLU(),
            nn.Linear(feat_dim*2, feat_dim))
        
        # 动量更新参数
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        # 初始化队列
        self.register_buffer("queue", torch.randn(queue_size, feat_dim))
        self.queue = F.normalize(self.queue, dim=0)
        self.queue_ptr = 0
        
    @torch.no_grad()
    def momentum_update(self, m=0.999):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
            
    def contrastive_loss(self, q, k):
        # 计算相似度
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach().t()])
        
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temp
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        # 更新队列
        batch_size = q.size(0)
        ptr = self.queue_ptr
        self.queue[ptr:ptr+batch_size] = k
        self.queue_ptr = (ptr + batch_size) % self.queue_size
        
        return F.cross_entropy(logits, labels)

# class EnhancedMedPredictor(nn.Module):
#     def __init__(self, num_codes, emb_dim=128):
#         super().__init__()
#         # 时序图模块
#         self.tgnn = TGNN(num_codes, emb_dim)
        
#         # 超图模块
#         self.hypergnn = DynamicHyperGNN(emb_dim, emb_dim)
        
#         # 对比学习
#         self.cl = MoCoContrastiveLearner(feat_dim=emb_dim)
        
#         # 多任务预测头
#         self.readmit_head = nn.Sequential(
#             nn.Linear(emb_dim*2, emb_dim),  # 融合时序和超图特征
#             nn.ReLU(),
#             nn.Linear(emb_dim, 1))
#         self.mortal_head = nn.Sequential(
#             nn.Linear(emb_dim*2, emb_dim),
#             nn.ReLU(),
#             nn.Linear(emb_dim, 1))
        
#     def forward(self, data):
#         # 时序图编码
#         code_feats = self.tgnn(data.x, data.edge_index, data.edge_time)
        
#         # 超图编码
#         hyper_feats = self.hypergnn(code_feats, data.hyperedges, data.visit_times)
        
#         # 患者级特征聚合
#         patient_feats = self._aggregate_features(code_feats, hyper_feats, data)
        
#         # 多任务预测
#         readmit_logits = self.readmit_head(patient_feats)
#         mortal_logits = self.mortal_head(patient_feats)
        
#         return readmit_logits, mortal_logits
    
#     def _aggregate_features(self, code_feats, hyper_feats, data):
#         """时序注意力聚合"""
#         # 获取每次就诊的代码索引
#         visit_indices = data.visit_indices
        
#         # 注意力权重计算
#         attn_weights = []
#         for visit in visit_indices:
#             codes = code_feats[visit]
#             attn = torch.matmul(codes, self.attn_query.weight)
#             attn = F.softmax(attn, dim=0)
#             attn_weights.append(attn)
        
#         # 加权聚合
#         aggregated = []
#         for i, (code, hyper) in enumerate(zip(code_feats, hyper_feats)):
#             agg = torch.cat([
#                 torch.sum(code * attn_weights[i], dim=0),
#                 hyper.mean(dim=0)
#             ])
#             aggregated.append(agg)
        
#         return torch.stack(aggregated)
    
class MedPredictor(nn.Module):
    def __init__(self, num_codes, emb_dim=128, tgat_heads=4):
        super().__init__()
        # 代码嵌入层
        self.code_emb = nn.Embedding(num_codes, emb_dim)
        # 添加feat_adj组件
        self.feat_adj = nn.Linear(emb_dim, emb_dim)  # 假设这是一个线性变换
        
        # 时间感知图网络
        self.tgnn = GATConv(
            emb_dim, 
            emb_dim,
            heads=tgat_heads,
            edge_dim=None  # 不使用边特征，避免维度不匹配
        )
        
        # 计算输出维度：emb_dim x 2（mean和max池化拼接）
        pred_input_dim = emb_dim * 2
        
        # 预测头
        self.readmit_head = nn.Sequential(
            nn.Linear(4096, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.mortal_head = nn.Sequential(
            nn.Linear(4096, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        # 检查数据格式并根据需要调整
        batch_size = 1
        if hasattr(data, 'batch') and data.batch is not None:
            batch_size = data.batch.max().item() + 1
        
        # 使用嵌入层处理代码索引
        if hasattr(data, 'code_indices') and data.code_indices is not None:
            # 如果我们有保存的代码索引，使用它们
            x = self.code_emb(data.code_indices)
        elif hasattr(data, 'x') and data.x.dtype == torch.long and data.x.dim() == 2 and data.x.size(1) == 1:
            # 如果x是包含索引的整数张量
            x = self.code_emb(data.x.squeeze())
        else:
            # 否则假设x已经包含特征
            x = data.x
            
            
        # 确保x是二维张量 [num_nodes, features]
        if x.dim() > 2:
            # 处理多维张量 - 使用平均池化将多头特征合并为单一特征
            if x.dim() == 3:
                # [num_nodes, heads, features] -> [num_nodes, features]
                x = x.mean(dim=1)
            else:
                # 其他情况，使用展平和线性层调整
                original_shape = x.shape
                x = x.view(original_shape[0], -1)
                # 如果维度太大，使用特征调整层
                if x.size(1) > 1000:  # 设置一个合理的阈值
                    # 重新整形为3D并平均
                    adj_dim = int(np.sqrt(x.size(1)))
                    x = x.view(x.size(0), -1, adj_dim).mean(dim=1)
                    
        # 应用特征调整层确保维度正确
        x = self.feat_adj(x)
            
        # GNN部分
        edge_index = data.edge_index
        
        # 图注意力网络
        x = F.leaky_relu(self.tgnn(
            x, edge_index,
        ))
        
        # 使用全局池化获取图级表示
        if hasattr(data, 'batch') and data.batch is not None:
            # 批处理图的情况
            from torch_geometric.nn import global_mean_pool, global_max_pool
            graph_embedding = torch.cat([
                global_mean_pool(x, data.batch),
                global_max_pool(x, data.batch)
            ], dim=-1)
        else:
            # 单一图的情况
            graph_embedding = torch.cat([
                x.mean(dim=0, keepdim=True),
                x.max(dim=0, keepdim=True)[0]
            ], dim=-1)
        
        # 调整图嵌入维度，确保与预测头匹配
        if graph_embedding.dim() == 3:
            graph_embedding = graph_embedding.view(graph_embedding.size(0), -1)
        
        # 预测任务
        readmit_logits = self.readmit_head(graph_embedding).squeeze()
        mortal_logits = self.mortal_head(graph_embedding).squeeze()
        
        # 处理单个样本的情况
        if batch_size == 1:
            readmit_logits = readmit_logits.view(1)
            mortal_logits = mortal_logits.view(1)
            
        return readmit_logits, mortal_logits
    
    def _build_hyperedges(self, data):
        """将每个HADM的就诊代码构建为超边"""
        return [data.x.tolist()]  # 每个HADM的代码作为一个超边
    
    def _aggregate_hadm(self, x, data):
        """聚合当前HADM的所有代码特征"""
        if hasattr(data, 'batch'):
        # 如果有batch信息，按batch聚合
            return torch.zeros(data.num_graphs, x.size(1)).to(x.device).scatter_add_(
                0, data.batch.unsqueeze(-1).repeat(1, x.size(1)), x
            )
        else:
        # 如果没有batch信息，简单平均
            return x.mean(dim=0)
