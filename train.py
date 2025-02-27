import argparse
import torch
import numpy as np
import os
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc
from data_preprocessing import MIMIC3Processor, MIMIC3StreamDataset, MIMIC3BatchDataset, incremental_train
from model import MedPredictor
import torch.nn.functional as F
import gc
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight_readmit=15.0, pos_weight_mortal=15.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight_readmit = torch.tensor(pos_weight_readmit)
        self.pos_weight_mortal = torch.tensor(pos_weight_mortal)
        
    def forward(self, readmit_logits, mortal_logits, y):
        # 分离再入院和死亡率标签
        readmit_labels = y[:, 0]
        mortal_labels = y[:, 1]
        
        # 使用内置的pos_weight参数
        readmit_loss = F.binary_cross_entropy_with_logits(
            readmit_logits, 
            readmit_labels.float(), 
            pos_weight=self.pos_weight_readmit.to(readmit_logits.device)
        )
        
        mortal_loss = F.binary_cross_entropy_with_logits(
            mortal_logits, 
            mortal_labels.float(), 
            pos_weight=self.pos_weight_mortal.to(mortal_logits.device)
        )
        
        # 总损失为两个任务损失的和
        total_loss = readmit_loss + mortal_loss
        
        return total_loss, readmit_loss, mortal_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/MIMI-CIII')
    parser.add_argument('--save_dir', type=str, default='./processed', help='预处理保存路径')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_batch_size', type=int, default=10000, help='数据处理批次大小')
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--tgat_heads', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5, help='每个批次的训练轮数')
    parser.add_argument('--total_batches', type=int, default=10, help='要处理的总批次数')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--mode', type=str, default='incremental', choices=['incremental', 'combined'], 
                        help='训练模式: incremental=增量训练, combined=合并数据训练')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的模型路径')
    parser.add_argument('--eval_every', type=int, default=1, help='每处理多少批次评估一次')
    parser.add_argument('--force_reprocess', action='store_true', help='强制重新处理所有批次数据')
    # 添加不平衡处理相关参数
    parser.add_argument('--pos_weight_readmit', type=float, default=1,
                        help='再入院任务中正样本的权重')
    parser.add_argument('--pos_weight_mortal', type=float, default=1,
                        help='死亡率任务中正样本的权重')
    parser.add_argument('--threshold_readmit', type=float, default=0.5,
                        help='再入院预测的决策阈值')
    parser.add_argument('--threshold_mortal', type=float, default=0.5,
                        help='死亡率预测的决策阈值')
    return parser.parse_args()

def evaluate(model, loader, device):
    model.eval()
    results = {'readmit': {'true': [], 'prob': []}, 
              'mortal': {'true': [], 'prob': []}}
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            readmit_logits, mortal_logits = model(batch)

            # 处理单样本和批量样本的情况
            if readmit_logits.ndim == 0:
                readmit_logits = readmit_logits.unsqueeze(0)
            if mortal_logits.ndim == 0:
                mortal_logits = mortal_logits.unsqueeze(0)

            results['readmit']['true'].extend(batch.y[:,0].cpu().numpy())
            results['readmit']['prob'].extend(torch.sigmoid(readmit_logits).cpu().numpy())
            results['mortal']['true'].extend(batch.y[:,1].cpu().numpy())
            results['mortal']['prob'].extend(torch.sigmoid(mortal_logits).cpu().numpy())
    # 统计正负样本数量
    readmit_true = np.array(results['readmit']['true'])
    mortal_true = np.array(results['mortal']['true'])
    
    readmit_pos = np.sum(readmit_true == 1)
    readmit_neg = np.sum(readmit_true == 0)
    mortal_pos = np.sum(mortal_true == 1)
    mortal_neg = np.sum(mortal_true == 0)
    
    print("\n样本分布统计:")
    print(f"  再入院预测: 正样本={readmit_pos}({readmit_pos/(readmit_pos+readmit_neg)*100:.1f}%), "
          f"负样本={readmit_neg}({readmit_neg/(readmit_pos+readmit_neg)*100:.1f}%), "
          f"总计={readmit_pos+readmit_neg}")
    print(f"  死亡率预测: 正样本={mortal_pos}({mortal_pos/(mortal_pos+mortal_neg)*100:.1f}%), "
          f"负样本={mortal_neg}({mortal_neg/(mortal_pos+mortal_neg)*100:.1f}%), "
          f"总计={mortal_pos+mortal_neg}")
    metrics = {}
    for task_name in ['readmit', 'mortal']:
        true = np.array(results[task_name]['true'])
        prob = np.array(results[task_name]['prob'])
        
        # 使用0.5作为默认阈值进行二分类
        pred = (prob >= 0.5).astype(int)
        
        metrics[task_name] = {
            'ACC': accuracy_score(true, pred),
            'F1': f1_score(true, pred, zero_division=0)
        }
        
        # 安全计算AUROC和PR-AUC
        unique_classes = np.unique(true)
        if len(unique_classes) > 1:
            # AUROC
            metrics[task_name]['AUROC'] = roc_auc_score(true, prob)
            
            # PR曲线和AUPRC
            precision, recall, _ = precision_recall_curve(true, prob)
            metrics[task_name]['AUPRC'] = auc(recall, precision)
            
            # 计算最优F1对应的阈值
            f1_scores = []
            thresholds = np.arange(0.1, 1.0, 0.05)
            for threshold in thresholds:
                pred_t = (prob >= threshold).astype(int)
                f1 = f1_score(true, pred_t, zero_division=0)
                f1_scores.append(f1)
            
            best_threshold_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_idx]
            best_pred = (prob >= best_threshold).astype(int)
            metrics[task_name]['Best_Threshold'] = best_threshold
            metrics[task_name]['Best_F1'] = f1_score(true, best_pred, zero_division=0)
        else:
            print(f"  警告: {task_name}预测任务只有一个类别的样本({unique_classes[0]})，无法计算AUROC/AUPRC")
            metrics[task_name]['AUROC'] = None
            metrics[task_name]['AUPRC'] = None
            metrics[task_name]['Best_Threshold'] = None
            metrics[task_name]['Best_F1'] = None
            
    return metrics

def incremental_training(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化处理器和流式数据集
    processor = MIMIC3Processor(args.data_dir)
    stream_dataset = MIMIC3StreamDataset(args.save_dir, processor, batch_size=args.data_batch_size)

    # 确保存储目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 初始化模型
    if args.resume and os.path.exists(args.resume):
        print(f"Loading model from {args.resume}")
        # 先创建模型实例
        model = MedPredictor(
            num_codes=len(processor.code_encoder.classes_),
            emb_dim=args.emb_dim,
            tgat_heads=args.tgat_heads
        )
        # 加载预训练权重
        model.load_state_dict(torch.load(args.resume))
    else:
        # 如果没有预训练模型，先处理一个批次以获取编码器
        if not stream_dataset.processed_batches:
            print("Processing first batch to initialize code encoder...")
            stream_dataset.process_batch(0)
            
        # 创建新模型
        model = MedPredictor(
            num_codes=len(processor.code_encoder.classes_),
            emb_dim=args.emb_dim,
            tgat_heads=args.tgat_heads
        )
    criterion = WeightedBCELoss(
        pos_weight_readmit=args.pos_weight_readmit, 
        pos_weight_mortal=args.pos_weight_mortal
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # 确定要处理的批次
    total_batches = min(args.total_batches, stream_dataset.total_batches)
    
    # 验证集 - 使用最后处理的批次
    val_batch_idx = None
    test_batch_idx = None
    
    # 增量训练
    best_auc = 0.0
    for batch_idx in range(total_batches):
        print(f"\n=== Processing and Training Batch {batch_idx}/{total_batches-1} ===")
        
        # 处理当前批次(如果尚未处理)
        if batch_idx not in stream_dataset.processed_batches:
            stream_dataset.process_batch(batch_idx)
        
        # 加载当前批次数据
        batch_file = os.path.join(stream_dataset.processed_dir, f'data_batch_{batch_idx}.pt')
        batch_data = torch.load(batch_file)
        # 验证数据格式
  
        for i, data in enumerate(batch_data):
            # 检查节点数量
            num_nodes = data.x.size(0)

            # 确保x是LongTensor类型 - 添加这个检查和修复
            if data.x.dtype != torch.long:
                
                # 如果是2D tensor则保持维度
                if data.x.dim() > 1:
                    data.x = data.x.long()
                else:
                    # 如果是1D tensor则添加维度
                    data.x = data.x.long().view(-1, 1)
            
            # 确保x维度正确 [num_nodes, feature_dim]
            if data.x.dim() != 2:
                
                if data.x.dim() == 1:
                    data.x = data.x.view(-1, 1)
                elif data.x.dim() > 2:
                    # 如果维度太多，压缩为2D
                    orig_shape = data.x.shape
                    data.x = data.x.view(orig_shape[0], -1)
            
            # 检查特征是否只有索引，需要将其扩展为独热编码或嵌入向量
            if data.x.shape[1] == 1:
                
                # 创建一个伪嵌入特征，在模型中会被正确的嵌入替换
                data.x = torch.zeros((num_nodes, 16), dtype=torch.float) 
                # 保存原始索引供模型使用
                data.code_indices = data.x.clone().long().squeeze()
            
            # 检查边索引
            if data.edge_index.shape[1] > 0:
                max_index = data.edge_index.max().item()
                if max_index >= num_nodes:

                    # 移除无效边
                    valid_edges = (data.edge_index[0] < num_nodes) & (data.edge_index[1] < num_nodes)
                    data.edge_index = data.edge_index[:, valid_edges]
                    
                    # 重新创建边属性
                    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                        data.edge_attr = torch.ones(data.edge_index.shape[1], dtype=torch.float)
            
            # 确保至少有一条边
            if data.edge_index.shape[1] == 0:
                print(f"警告: 样本 {i} 没有有效边，添加自环...")
                data.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    data.edge_attr = torch.ones(1, dtype=torch.float)
                else:
                    data.edge_attr = torch.ones(1, dtype=torch.float)
        # 划分当前批次的训练/验证数据
        indices = torch.randperm(len(batch_data)).tolist()
        train_size = int(0.8 * len(batch_data))
        train_data = [batch_data[i] for i in indices[:train_size]]
        val_data = [batch_data[i] for i in indices[train_size:]]
        
        # 保存验证和测试批次索引(使用最后两个批次)
        if batch_idx == total_batches - 2:
            val_batch_idx = batch_idx
        elif batch_idx == total_batches - 1:
            test_batch_idx = batch_idx
        
        # 创建数据加载器
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)
        
        # 训练当前批次
        model = model.to(device)
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            readmit_losses = 0.0
            mortal_losses = 0.0
            
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                
                readmit_pred, mortal_pred = model(data)
                
                # 处理单样本和批量样本的情况
                if readmit_pred.ndim == 0:
                    readmit_pred = readmit_pred.unsqueeze(0)
                if mortal_pred.ndim == 0:
                    mortal_pred = mortal_pred.unsqueeze(0)
                
                readmit_logits, mortal_logits = model(data)
                loss, readmit_loss, mortal_loss = criterion(readmit_logits, mortal_logits, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                readmit_losses += readmit_loss.item()
                mortal_losses += mortal_loss.item()
                if (batch_idx + 1) % args.eval_every == 0 or batch_idx == total_batches - 1:
                # 计算平均损失
                    avg_loss = total_loss / (batch_idx + 1)
                    avg_readmit_loss = readmit_losses / (batch_idx + 1)
                    avg_mortal_loss = mortal_losses / (batch_idx + 1)
            print(f"  Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        # 每隔指定批次评估一次
        if (batch_idx + 1) % args.eval_every == 0 or batch_idx == total_batches - 1:
            val_metrics = evaluate(model, val_loader, device)
            print(f"\nBatch {batch_idx} Validation Results:")
            # 检查AUROC是否为None
            if val_metrics['readmit']['AUROC'] is not None:
                print(f"AUROC: {val_metrics['readmit']['AUROC']:.4f}")
            else:
                print("AUROC: N/A (single class)")
            print(f"[Readmission] ACC: {val_metrics['readmit']['ACC']:.4f} "
                 f"AUC: {val_metrics['readmit']['AUROC']:.4f}")
            print(f"[Mortality]   ACC: {val_metrics['mortal']['ACC']:.4f} "
                 f"AUC: {val_metrics['mortal']['AUROC']:.4f}")
            if val_metrics['mortal']['AUROC'] is not None:
                print(f"AUROC: {val_metrics['readmit']['AUROC']:.4f}")
            else:
                print("AUROC: N/A (single class)")
            # 保存最佳模型
            current_auc = (val_metrics['readmit']['AUROC'] + val_metrics['mortal']['AUROC']) / 2
            if current_auc > best_auc:
                best_auc = current_auc
                torch.save(model.state_dict(), f"{args.save_dir}/best_model_batch_{batch_idx}.pth")
                print(f"New best model saved at batch {batch_idx}")
        
        # 保存当前批次的模型
        torch.save(model.state_dict(), f"{args.save_dir}/model_after_batch_{batch_idx}.pth")
        
        # 释放内存
        del batch_data, train_loader, val_loader, train_data, val_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 最终测试 - 使用最后一个批次或指定的测试批次
    print("\n=== Final Evaluation ===")
    
    if test_batch_idx is not None:
        # 加载最佳模型
        best_model_path = f"{args.save_dir}/best_model_batch_{val_batch_idx}.pth"
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path}")
        
        # 加载测试批次
        test_file = os.path.join(stream_dataset.processed_dir, f'data_batch_{test_batch_idx}.pt')
        test_data = torch.load(test_file)
        test_loader = DataLoader(test_data, batch_size=args.batch_size)
        
        # 评估
        test_metrics = evaluate(model, test_loader, device)
        print("\n=== Final Test Results ===")
        print(f"[Readmission] {test_metrics['readmit']}")
        print(f"[Mortality]   {test_metrics['mortal']}")
        

        # 释放内存
        del test_data, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def combined_training(args):
    """使用合并的数据集进行传统训练"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.force_reprocess:
        print("强制重新处理数据...")
        import os
        import glob
        processed_dir = os.path.join(args.save_dir, 'processed')
        if os.path.exists(processed_dir):
            batch_files = glob.glob(os.path.join(processed_dir, 'data_batch_*.pt'))
            for file in batch_files:
                print(f"删除旧批次文件: {file}")
                os.remove(file)
    
    # 初始化处理器
    processor = MIMIC3Processor(args.data_dir)
    
    # 确定要处理的批次
    stream_dataset = MIMIC3StreamDataset(args.save_dir, processor, batch_size=args.data_batch_size)
    total_batches = min(args.total_batches, stream_dataset.total_batches)
    
    # 处理所有需要的批次
    for batch_idx in range(total_batches):
        if batch_idx not in stream_dataset.processed_batches:
            stream_dataset.process_batch(batch_idx)
    
    # 使用合并数据集
    dataset = MIMIC3BatchDataset(args.save_dir, processor, batch_indices=range(total_batches))
    
    # 划分数据集
    indices = torch.randperm(len(dataset)).tolist()
    train_idx = indices[:int(0.7*len(dataset))]
    val_idx = indices[int(0.7*len(dataset)):int(0.85*len(dataset))]
    test_idx = indices[int(0.85*len(dataset)):] 
    
    train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=args.batch_size)
    test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size)
    
    # 初始化模型
    model = MedPredictor(
        num_codes=len(processor.code_encoder.classes_),
        emb_dim=args.emb_dim,
        tgat_heads=args.tgat_heads
    )
    model = model.to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # 训练循环
    best_auc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            readmit_pred, mortal_pred = model(batch)
            
            # 处理单样本和批量样本的情况
            if readmit_pred.ndim == 0:
                readmit_pred = readmit_pred.unsqueeze(0)
            if mortal_pred.ndim == 0:
                mortal_pred = mortal_pred.unsqueeze(0)
            
            loss = F.binary_cross_entropy_with_logits(readmit_pred, batch.y[:,0]) + \
                   F.binary_cross_entropy_with_logits(mortal_pred, batch.y[:,1])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证
        val_metrics = evaluate(model, val_loader, device)
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"[Readmission] ACC: {val_metrics['readmit']['ACC']:.4f} "
             f"AUC: {val_metrics['readmit']['AUROC']:.4f}")
        print(f"[Mortality]   ACC: {val_metrics['mortal']['ACC']:.4f} "
             f"AUC: {val_metrics['mortal']['AUROC']:.4f}")
        
        # 保存最佳模型
        current_auc = (val_metrics['readmit']['AUROC'] + val_metrics['mortal']['AUROC']) / 2
        if current_auc > best_auc:
            best_auc = current_auc
            torch.save(model.state_dict(), f"{args.save_dir}/best_model_combined.pth")
    
    # 最终测试
    model.load_state_dict(torch.load(f"{args.save_dir}/best_model_combined.pth"))
    test_metrics = evaluate(model, test_loader, device)
    print("\n=== Final Test Results ===")
    print(f"[Readmission] {test_metrics['readmit']}")
    print(f"[Mortality]   {test_metrics['mortal']}")

def main(args):
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 根据模式选择训练方法
    if args.mode == 'incremental':
        print("Starting incremental training...")
        incremental_training(args)
    else:
        print("Starting combined training...")
        combined_training(args)



if __name__ == "__main__":
    args = parse_args()
    main(args)
