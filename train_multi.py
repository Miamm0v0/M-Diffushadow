import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from numericalmodel import NumericalDiffusionModel
from Qmultimodel import QuantumMultimodalDModel_nseq
import json
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import random
import matplotlib.pyplot as plt
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Quantum Diffusion Model Training with Dual Modal Data")

    # 数据相关 - 两个文件路径
    parser.add_argument('--seq_data_path', type=str, 
                        default='/home/fyp26lyh/QuantumLLADA/data/TFI_train_seq.json',
                        help='Path to the sequence JSON data file')
    parser.add_argument('--phys_data_path', type=str,
                        default='/home/fyp26lyh/QuantumLLADA/data/TFI_train_phys.json',
                        help='Path to the physical features JSON data file')
    
    parser.add_argument('--seq_eval_data_path', type=str, 
                        default='/home/fyp26lyh/QuantumLLADA/data/TFI_train_seq.json',
                        help='Path to the sequence JSON data file')
    parser.add_argument('--phys_eval_data_path', type=str,
                        default='/home/fyp26lyh/QuantumLLADA/data/TFI_train_phys.json',
                        help='Path to the physical features JSON data file')
    parser.add_argument('--model_load_path', type=str, default='',
                        help='Path to load a pre-trained model ')
    
    parser.add_argument('--eval_loss', type=bool, default=False,
                        help='Whether to evaluate model on eval set during training')
    
    parser.add_argument('--train_sample_per_group', type=int, default=10000,
                        help='Number of samples per h group for training dataset')
    parser.add_argument('--eval_sample_per_group', type=int, default=1000,
                        help='Number of samples per h group for eval dataset')    
    parser.add_argument('--train_target', type=str, default='both',
                        choices=['b','c','both'],
                        help='the training target of model')
        
    
    # 训练超参数
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--head_count', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--eps_b', type=float, default=1e-3,
                        help='Minimum masking probability offset for b modal (for p_mask)')
    parser.add_argument('--eps_c', type=float, default=1e-3,
                        help='Minimum masking probability offset for c modal (for p_mask)')    
    parser.add_argument('--pairing_epochs', type=int, default=2,
                        help='Number of pairing epochs for dual modal data')

    # 学习率调度器
    parser.add_argument('--lr_scheduler', type=str, default='constant',
                        choices=['constant', 'cosine', 'step', 'linear'],
                        help='Learning rate scheduler: constant, cosine, step, linear')
    parser.add_argument('--step_size', type=int, default=5,
                        help='Step size for StepLR scheduler (only used if lr_scheduler=step)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Multiplicative factor for StepLR or Linear decay')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')     

    # 模型保存
    parser.add_argument('--model_save_path', type=str,
                        default='/home/fyp26lyh/QuantumLLADA/DshadowGPT/model_train/Quantum_diffusion_model.pth',
                        help='Path to save the trained model')
    
    parser.add_argument('--loss_save_path', type=str,
                        default='/home/fyp26lyh/QuantumLLADA/DshadowGPT/model_train/loss_fig/loss_multimodal.json',
                        help='Path to save the loss curve figure')

    # 调试选项
    parser.add_argument('--print_grad_per_step', action='store_true',
                        help='Print gradient norm for every batch')

    return parser.parse_args()


MASK_TOKEN_ID = -1.0

# def apply_numerical_mask(x, mask_ratio):
#     mask = torch.rand_like(x) < mask_ratio
#     masked_x = x.clone()
#     masked_x[mask] = MASK_TOKEN_ID  # 用0掩码（或自定义其他值）
#     return masked_x, mask

# def numerical_diffusion_loss(model, x, mask_ratio, device='cuda'):
#     # x: [B, L, D]
#     masked_x, mask = apply_numerical_mask(x, mask_ratio)
#     pred = model(masked_x.to(device), mask_ratio.to(device))
#     loss = F.mse_loss(pred[mask], x.to(device)[mask])  # 均方误差
#     return loss

def get_scheduler(optimizer, args, total_steps):
    if args.lr_scheduler == 'constant':
        return None  # 无调度器
    elif args.lr_scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif args.lr_scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'linear':
        # 线性衰减到 0
        return torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                lambda step: max(0.0, 1.0 - step / total_steps))
    else:
        raise ValueError(f"Unknown scheduler: {args.lr_scheduler}")



def forward_process_b(batch, eps=1e-3):
    """
    对状态序列部分（第1列到第20列）添加掩码噪声，使用 -1.0 填充
    保留第0列作为条件（不掩码）
    
    Args:
        batch: shape [B, 31], 第0列是条件，第1~11列是P_i，第12-21列是b_i，第22-31列是C_i
        eps: 最小掩码概率偏移
    
    Returns:
        noisy_batch: 掩码后的数据，被掩码位置为 -1.0
        masked_indices: 哪些位置被掩码了（只针对状态序列）
        p_mask: 每个样本的掩码概率 [B]
    """
    x = batch.float()  # [B, 21]
    b, l = x.shape
    device = x.device

    cond = x[:, :11]          # [B, 11]  条件列（不加噪声）
    seq = x[:, 11:]           # [B, 10] 状态序列（要加噪声）

    t = torch.rand(b, device=device)  # [B]
    p_mask = (1 - eps) * t + eps      # [B], 掩码概率 ∈ [eps, 1]

    p_mask_expanded = p_mask.unsqueeze(1).expand(-1, 10)  # [B, 10]
    rand = torch.rand((b, 10), device=device)           # [B, 10]
    masked_indices = rand < p_mask_expanded             # [B, 10], bool
    seq_masked = torch.where(masked_indices, 
                            seq.new_tensor(-1.0), 
                            seq)  # [B, 10]

    noisy_batch = torch.cat([cond, seq_masked], dim=1)  # [B, 21]

    return noisy_batch, masked_indices, p_mask



def forward_process_c(batch, eps=1e-3):
    """
    对状态序列部分（第1列到第20列）添加掩码噪声，使用 -1.0 填充
    保留第0列作为条件（不掩码）
    
    Args:
        batch: shape [B, 31], 第0列是条件，第1~11列是P_i，第12-21列是b_i，第22-31列是C_i
        eps: 最小掩码概率偏移
    
    Returns:
        noisy_batch: 掩码后的数据，被掩码位置为 -1.0
        masked_indices: 哪些位置被掩码了（只针对状态序列）
        p_mask: 每个样本的掩码概率 [B]
    """
    x = batch.float()  # [B, 21]
    b, l = x.shape
    device = x.device

    g = x[:, 0:1] 
    pair_data = x[:, 1:].reshape(b, 10, 3)
    r1_pair = pair_data[:, :, 0]  # [B, 10] - 第一个r
    r2_pair = pair_data[:, :, 1]  # [B, 10] - 第二个r
    p_pair = pair_data[:, :, 2]   # [B, 10] - 乘积结果

    # cond = x[:, :11]          # [B, 11]  条件列（不加噪声）
    # seq = x[:, 11:]           # [B, 10] 状态序列（要加噪声）

    t = torch.rand(b, device=device)  # [B]
    p_mask = (1 - eps) * t + eps      # [B], 掩码概率 ∈ [eps, 1]

    p_mask_expanded = p_mask.unsqueeze(1).expand(-1, 10)  # [B, 10]
    rand = torch.rand((b, 10), device=device)           # [B, 10]
    masked_indices = rand < p_mask_expanded             # [B, 10], bool
    seq_masked = torch.where(masked_indices, 
                            p_pair.new_tensor(-1.0), 
                            p_pair)  # [B, 10]
    pair_data = torch.stack([r1_pair, r2_pair, seq_masked], dim=2)  # [2,2,3]
    flat_part = pair_data.reshape(b, 30)

    noisy_batch = torch.cat([g, flat_part], dim=1)  # [B, 31]

    return noisy_batch, masked_indices, p_mask

# with open(r'/home/fyp26lyh/QuantumLLADA/data/TFI_train_data_nseq.json', 'r') as f:
#     raw_data = json.load(f)

import json
import torch
from torch.utils.data import Dataset
import random

import torch
from torch.utils.data import Dataset
import json
import random

class MultimodalQuantumDataset(Dataset):
    def __init__(self, single_json_file, pair_json_file, pairing_epochs=2, sample_per_group=10000, current_round=0):
        """
        双模态量子数据集
        Args:
            single_json_file: 单模态数据文件路径
            pair_json_file: 双模态数据文件路径
            pairing_epochs: 总配对轮次
            current_round: 当前配对轮次 (0, 1, ..., pairing_epochs-1)
        """
        # 1. 加载原始数据（不shuffle，保持原始分组）
        print(f"Loading single data from {single_json_file}...")
        with open(single_json_file, 'r') as f:
            self.single_data = json.load(f)
        print(f"Loading pair data from {pair_json_file}...")
        with open(pair_json_file, 'r') as f:
            self.pair_data = json.load(f)
        
        print(f"Single data: {len(self.single_data)} samples")
        print(f"Pair data: {len(self.pair_data)} samples")
        
        # 2. 按原始位置分组（假设每个h有10000个连续样本）
        self.single_by_h = self._group_by_original_position(self.single_data)
        self.pair_by_h = self._group_by_original_position(self.pair_data)
        
        print(f"Found {len(self.single_by_h)} h values in single data")
        print(f"Found {len(self.pair_by_h)} h values in pair data")
        
        # 3. 验证两个模态的h值一致
        # self._validate_h_consistency()
        
        # 4. 为当前轮次创建随机配对
        self.pairing_epochs = pairing_epochs
        self.current_round = current_round
        self.paired_samples = self._create_pairing(current_round)
        
        print(f"Created {len(self.paired_samples)} paired samples for round {current_round}")

    def _group_by_original_position(self, data):
        """按原始位置分组，假设每个h有10000个连续样本"""
        grouped = {}
        num_h = len(data) // 10000
        
        for i in range(num_h):
            start = i * 10000
            end = (i + 1) * 10000
            h_value = data[start][0]  # 取该组第一个样本的h值
            grouped[h_value] = data[start:end]
            print(f"h={h_value}: {len(grouped[h_value])} samples")
            
        return grouped

    def _validate_h_consistency(self):
        """验证两个模态的h值集合是否一致"""
        single_h_values = set(self.single_by_h.keys())
        pair_h_values = set(self.pair_by_h.keys())
        
        if single_h_values != pair_h_values:
            raise ValueError(f"h值不匹配! Single: {single_h_values}, Pair: {pair_h_values}")
        
        # 验证每个h的样本数一致
        for h_value in single_h_values:
            single_count = len(self.single_by_h[h_value])
            pair_count = len(self.pair_by_h[h_value])
            if single_count != pair_count:
                raise ValueError(f"h={h_value} 样本数不匹配: Single={single_count}, Pair={pair_count}")

    def _create_pairing(self, current_round):
        """为当前轮次创建随机配对"""
        paired_samples = []
        
        # 对每个h值独立创建配对
        for h_value in self.single_by_h.keys():
            single_samples = self.single_by_h[h_value]  # 该h的所有单模态样本
            pair_samples = self.pair_by_h[h_value]      # 该h的所有双模态样本
            
            # 设置固定随机种子确保可重现的配对
            seed = hash(str(h_value) + str(current_round)) % (2**32)
            random.seed(seed)
            
            # 创建pair样本的随机索引排列
            pair_indices = list(range(len(pair_samples)))
            random.shuffle(pair_indices)
            
            # 创建配对：single_samples[i] 配 pair_samples[pair_indices[i]]
            for i in range(len(single_samples)):
                pair_idx = pair_indices[i]
                paired_samples.append({
                    'single': single_samples[i],
                    'pair': pair_samples[pair_idx],
                    'h': h_value
                })
                
        return paired_samples

    def update_pairing(self, new_round):
        """更新到新的配对轮次"""
        if new_round < 0 or new_round >= self.pairing_epochs:
            raise ValueError(f"轮次必须在 [0, {self.pairing_epochs-1}] 范围内")
            
        if new_round != self.current_round:
            print(f"Updating pairing from round {self.current_round} to {new_round}")
            self.current_round = new_round
            self.paired_samples = self._create_pairing(new_round)
            print(f"Pairing updated, now {len(self.paired_samples)} samples")

    def __len__(self):
        return len(self.paired_samples)

    def __getitem__(self, idx):
        sample = self.paired_samples[idx]
        single_tensor = torch.tensor(sample['single'], dtype=torch.float32)
        pair_tensor = torch.tensor(sample['pair'], dtype=torch.float32)
        
        return single_tensor, pair_tensor

    def get_h_values(self):
        """返回所有h值"""
        return list(self.single_by_h.keys())

    def get_pairing_info(self):
        """返回配对信息"""
        return {
            'current_round': self.current_round,
            'total_rounds': self.pairing_epochs,
            'total_samples': len(self.paired_samples),
            'h_values': self.get_h_values()
        }



class MultimodalEvalDataset(Dataset):
    def __init__(self, single_json_file, pair_json_file):
        """
        验证集双模态数据集 - 处理不均衡分组
        Args:
            single_json_file: 单模态验证数据文件
            pair_json_file: 双模态验证数据文件  
            samples_per_h: 每个h抽样的样本数，None表示使用所有样本
        """
        print(f"Loading eval single data from {single_json_file}...")
        with open(single_json_file, 'r') as f:
            self.single_data = json.load(f)
        print(f"Loading eval pair data from {pair_json_file}...")
        with open(pair_json_file, 'r') as f:
            self.pair_data = json.load(f)
        
        print(f"Eval single data: {len(self.single_data)} samples")
        print(f"Eval pair data: {len(self.pair_data)} samples")
        
        # 按h值动态分组（不假设固定数量）
        self.single_by_h = self._group_by_h_dynamic(self.single_data)
        self.pair_by_h = self._group_by_h_dynamic(self.pair_data)
        
        print(f"Found {len(self.single_by_h)} h values in single eval data")
        print(f"Found {len(self.pair_by_h)} h values in pair eval data")
        
        # 验证h值一致性
        self._validate_h_consistency_eval()
        
        # 创建配对（验证集通常只用一种配对）
        self.paired_samples = self._create_eval_pairing()
        
        print(f"Created {len(self.paired_samples)} eval paired samples")

    def _group_by_h_dynamic(self, data):
        """动态按h值分组，不假设固定样本数"""
        grouped = {}
        for sample in data:
            h_value = sample[0]
            if h_value not in grouped:
                grouped[h_value] = []
            grouped[h_value].append(sample)
        return grouped

    def _validate_h_consistency_eval(self):
        """验证集h值一致性检查（更宽松）"""
        single_h_values = set(self.single_by_h.keys())
        pair_h_values = set(self.pair_by_h.keys())
        
        # 只检查两个模态共有的h值
        common_h = single_h_values & pair_h_values
        if len(common_h) == 0:
            raise ValueError("验证集没有共同的h值!")
        
        print(f"Common h values: {len(common_h)}")
        for h in common_h:
            single_count = len(self.single_by_h[h])
            pair_count = len(self.pair_by_h[h])
            print(f"h={h}: single={single_count}, pair={pair_count}")

    def _create_eval_pairing(self):
        """创建验证集配对 - 使用所有可配对的样本"""
        paired_samples = []
        
        # 只使用两个模态共有的h值
        common_h = set(self.single_by_h.keys()) & set(self.pair_by_h.keys())
        
        for h_value in sorted(common_h):
            single_samples = self.single_by_h[h_value]
            pair_samples = self.pair_by_h[h_value]
            
            # 使用两个模态中较小的样本数
            num_samples = min(len(single_samples), len(pair_samples))
            
            if num_samples == 0:
                print(f"Warning: h={h_value} 没有可配对的样本")
                continue
                
            print(f"h={h_value}: 使用 {num_samples} 个样本 (single有{len(single_samples)}, pair有{len(pair_samples)})")
            
            # 设置固定随机种子
            seed = hash(str(h_value)) % (2**32)
            random.seed(seed)
            
            # 从两个模态中独立随机抽样
            single_indices = random.sample(range(len(single_samples)), num_samples)
            pair_indices = random.sample(range(len(pair_samples)), num_samples)
            
            for i in range(num_samples):
                single_idx = single_indices[i]
                pair_idx = pair_indices[i]
                
                paired_samples.append({
                    'single': single_samples[single_idx],
                    'pair': pair_samples[pair_idx],
                    'h': h_value
                })
        
        return paired_samples

    def __len__(self):
        return len(self.paired_samples)

    def __getitem__(self, idx):
        sample = self.paired_samples[idx]
        single_tensor = torch.tensor(sample['single'], dtype=torch.float32)
        pair_tensor = torch.tensor(sample['pair'], dtype=torch.float32)
        return single_tensor, pair_tensor, sample['h']

    def get_distribution_info(self):
        """返回验证集分布信息"""
        distribution = {}
        for sample in self.paired_samples:
            h = sample['h']
            distribution[h] = distribution.get(h, 0) + 1
        
        info = {
            'total_samples': len(self.paired_samples),
            'h_distribution': distribution,
            'h_values': sorted(distribution.keys())
        }
        return info




def dual_collate_fn(batch):
    """
    处理双模态数据的collate函数
    """
    seq_batch = torch.stack([item[0] for item in batch])  # [B, 21]
    phys_batch = torch.stack([item[1] for item in batch]) # [B, 31]
    return seq_batch, phys_batch



@torch.no_grad()  # 不计算梯度，节省内存
def evaluate(model, qubit, dataloader_eval, loss_fn, args):
    """
    在验证集上评估模型的平均 loss
    """
    model.eval()  # 切换到 eval 模式（影响 dropout/batchnorm）
    total_loss = 0.0
    num_batches = 0


    if args.train_target == 'both':
        for seq_batch, phys_batch in dataloader_eval:
            seq_batch = seq_batch.to('cuda')      # [B, 21]
            phys_batch = phys_batch.to('cuda')    # [B, 31]
            B, L = phys_batch.shape  # L=2*qubit+1
            pair_data = phys_batch[:, 1:].reshape(B, 10, 3)
            c_pair = pair_data[:, :, 2]   # [B, 10] - 乘积结果


            # 加噪声
            noisy_data, masked_indices, p_mask = forward_process_b(batch=seq_batch, eps=args.eps_b)
            noisy_data_c, masked_indices_c, p_mask_c = forward_process_c(batch=phys_batch, eps=args.eps_c)
            pred_b, pred_c = model(x_single=noisy_data.unsqueeze(-1), x_pair=noisy_data_c.unsqueeze(-1), mask_indices=masked_indices, mask_indices_pair=masked_indices_c)  # [B, qubit, 2]

            # 目标
            target = seq_batch[:, 1+qubit:].round().long()  # [B, qubit]
            flat_logits = pred_b.reshape(-1, 2)           # [B*qubit, 2]
            flat_targets = target.reshape(-1)           # [B*qubit]
            flat_mask = masked_indices.reshape(-1)      # [B*qubit]

            target_c = c_pair.round().long()
            flat_logits_c = pred_c.reshape(-1, 2)           # [B*qubit, 2]
            flat_targets_c = target_c.reshape(-1)           # [B*qubit]
            flat_mask_c = masked_indices_c.reshape(-1)      # [B*qubit]

            # 只对被掩码的位置计算损失
            masked_logits = flat_logits[flat_mask]      # [M, 2]
            masked_targets = flat_targets[flat_mask]    # [M]
            loss = loss_fn(masked_logits, masked_targets)

            masked_logits_c = flat_logits_c[flat_mask_c]      # [M, 2]
            masked_targets_c = flat_targets_c[flat_mask_c]    # [M]
            loss_c = loss_fn(masked_logits_c, masked_targets_c)        

            total_loss += loss.item()
            total_loss += loss_c.item()
            num_batches += 1

        avg_loss = total_loss / num_batches / 2
    elif args.train_target == 'b':
        for seq_batch, phys_batch in dataloader_eval:
            seq_batch = seq_batch.to('cuda')      # [B, 21]
            phys_batch = phys_batch.to('cuda')    # [B, 31]
            # B, L = data.shape  # L=2*qubit+1

            # 加噪声
            noisy_data, masked_indices, p_mask = forward_process_b(batch=seq_batch, eps=args.eps_b)
            pred_b, pred_c = model(x_single=noisy_data.unsqueeze(-1), x_pair=phys_batch, mask_indices=masked_indices)  # [B, qubit, 2]

            # 目标
            target = seq_batch[:, 1+qubit:].round().long()  # [B, qubit]
            flat_logits = pred_b.reshape(-1, 2)           # [B*qubit, 2]
            flat_targets = target.reshape(-1)           # [B*qubit]
            flat_mask = masked_indices.reshape(-1)      # [B*qubit]

            # 只对被掩码的位置计算损失
            masked_logits = flat_logits[flat_mask]      # [M, 2]
            masked_targets = flat_targets[flat_mask]    # [M]
            loss = loss_fn(masked_logits, masked_targets)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches        
    else:
        for seq_batch, phys_batch in dataloader_eval:
            seq_batch = seq_batch.to('cuda')      # [B, 21]
            phys_batch = phys_batch.to('cuda')    # [B, 31]
            B, L = phys_batch.shape  # L=2*qubit+1
            pair_data = phys_batch[:, 1:].reshape(B, 10, 3)
            c_pair = pair_data[:, :, 2]   # [B, 10] - 乘积结果


            noisy_data_c, masked_indices_c, p_mask_c = forward_process_c(batch=phys_batch, eps=args.eps_c)
            pred_b, pred_c = model(x_single=seq_batch.unsqueeze(-1), x_pair=noisy_data_c.unsqueeze(-1), mask_indices_pair=masked_indices_c)  # [B, qubit, 2]


            target_c = c_pair.round().long()
            flat_logits_c = pred_c.reshape(-1, 2)           # [B*qubit, 2]
            flat_targets_c = target_c.reshape(-1)           # [B*qubit]
            flat_mask_c = masked_indices_c.reshape(-1)      # [B*qubit]

            masked_logits_c = flat_logits_c[flat_mask_c]      # [M, 2]
            masked_targets_c = flat_targets_c[flat_mask_c]    # [M]
            loss_c = loss_fn(masked_logits_c, masked_targets_c)       

            total_loss += loss_c.item()
            num_batches += 1 
        
        avg_loss = total_loss / num_batches     

    return avg_loss




def main():
    args = parse_args()
    print("🚀 Starting training with arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")    
    dataset_train = MultimodalQuantumDataset(
        single_json_file=args.seq_data_path,
        pair_json_file=args.phys_data_path,
        pairing_epochs=args.pairing_epochs,
        sample_per_group=args.train_sample_per_group,
        current_round=0
    )
    if args.eval_loss:
        dataset_eval = MultimodalEvalDataset(
            single_json_file=args.seq_eval_data_path,
            pair_json_file=args.phys_eval_data_path
            )
        dataloader_eval = DataLoader(
            dataset_eval,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=dual_collate_fn
        )
    # dataset_test = SimpleQuantumDataset(args.data_path.replace('train', 'test'))

    # print(f"\n✅ Total number of samples in training dataset: {len(dataset_train):,}")
    # print(f"\n✅ Total number of samples in testing dataset: {len(dataset_test):,}")

    dataloaders = []
    for round in range(dataset_train.pairing_epochs):
        dataset_train.update_pairing(round)
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            collate_fn=dual_collate_fn
        )
        dataloaders.append(dataloader_train)



    for i, (seq_batch, phys_batch) in enumerate(dataloaders[0]):
        if i == 0:
            print(f"Sequence batch shape: {seq_batch.shape}")  # [B, 21]
            print(f"Physical batch shape: {phys_batch.shape}")  # [B, 90]
            print(f"Sample 0 - g: {seq_batch[0, 0]}, P: {seq_batch[0, 1:11]}, b: {seq_batch[0, 11:21]}")
            break

    total_steps = args.epochs * len(dataloaders[0])

    # model = NumericalDiffusionModel().to('cuda')
    model = QuantumMultimodalDModel_nseq(
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        head_count=args.head_count
    ).to('cuda')


    if args.model_load_path != '':
        print(f"Loading pre-trained model from {args.model_load_path}...")
        model.load_state_dict(torch.load(args.model_load_path, map_location='cuda'))


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()


    scheduler = get_scheduler(optimizer, args, total_steps)

    grad_norms = []
    lrs = []
    epoch_losses = []
    test_losses = []

    from tqdm import tqdm


    if args.train_target == 'both':
        for epoch in range(args.epochs):
            total_loss = 0.0
            current_round = epoch % dataset_train.pairing_epochs
            current_dataloader = dataloaders[current_round]  # 🎯 直接取用
            model.train()
            for seq_batch, phys_batch in tqdm(current_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
                seq_batch = seq_batch.to('cuda')      # [B, 21]
                phys_batch = phys_batch.to('cuda')    # [B, 31]
                B, L = seq_batch.shape
                num_qubits = int((L - 1) / 2)


                pair_data = phys_batch[:, 1:].reshape(B, 10, 3)
                c_pair = pair_data[:, :, 2]   # [B, 10] - 乘积结果

                data_m1 = seq_batch
                # 加噪声
                noisy_data, masked_indices, p_mask = forward_process_b(batch=data_m1, eps=args.eps_b)
                noisy_data_c, masked_indices_c, p_mask_c = forward_process_c(batch=phys_batch, eps=args.eps_c)
                # print(f"noisy_data: {noisy_data}")
                # print(f"shape of noisy_data: {noisy_data.shape}")
                # print(f"noisy_data: {noisy_data_c}")
                # print(f"shape of noisy_data: {noisy_data_c.shape}")            
                # print(phys_batch)
                # print(f"Contains NaN in C: {torch.isnan(phys_batch).any().item()}")
                # print(f"Contains Inf in C: {torch.isinf(phys_batch).any().item()}")
                # print(f"Max value in C: {phys_batch.max().item()}, Min value: {phys_batch.min().item()}")
                pred_b, pred_c = model(x_single=noisy_data.unsqueeze(-1), x_pair=noisy_data_c.unsqueeze(-1), mask_indices = masked_indices, mask_indices_pair=masked_indices_c)
                # print(f"shape of pred: {pred.shape}")

                # 损失计算
                # pred_b loss
                target = seq_batch[:, 11:21]
                target = target.round().long()
                flat_logits = pred_b.reshape(-1, 2)           # [B*10, 2]
                flat_targets = target.reshape(-1)         # [B*10]
                flat_mask = masked_indices.reshape(-1)             # [B*10]

                # 只对被掩码的位置计算损失
                masked_logits = flat_logits[flat_mask]             # [M, 2]
                masked_targets = flat_targets[flat_mask]           # [M]

                # pred_c loss
                target_c = c_pair.round().long()
                flat_logits_c = pred_c.reshape(-1, 2)           # [B*10, 2]
                flat_targets_c = target_c.reshape(-1)         # [B*10]
                flat_mask_c = masked_indices_c.reshape(-1)             # [B*10]

                # 只对被掩码的位置计算损失
                masked_logits_c = flat_logits_c[flat_mask_c]             # [M, 2]
                masked_targets_c = flat_targets_c[flat_mask_c]           # [M]

                # 损失
                loss = loss_fn(masked_logits, masked_targets) + loss_fn(masked_logits_c, masked_targets_c)

                optimizer.zero_grad()
                loss.backward()

                total_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5  # L2 范数
                grad_norms.append(total_norm)

                if args.print_grad_per_step:
                    print(f"  Loss: {loss.item():.6f} | Grad Norm: {total_norm:.6f}")

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                    lrs.append(optimizer.param_groups[0]['lr'])

                total_loss += loss.item() / 2

            avg_loss = total_loss / len(current_dataloader)
            epoch_losses.append(avg_loss)
            if args.eval_loss:
                avg_eval_loss = evaluate(model, num_qubits, dataloader_eval, loss_fn, args)
                test_losses.append(avg_eval_loss)
                model.train()
            else:
                avg_eval_loss = None



            avg_grad_norm = sum(grad_norms[-len(current_dataloader):]) / len(current_dataloader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_loss:.8f} | Avg Grad Norm: {avg_grad_norm:.6f} | LR: {current_lr:.2e}")
            if avg_grad_norm < 1e-5:
                print("Gradient norm is too small, stopping training.")
                break

    elif args.train_target == 'b':
        for epoch in range(args.epochs):
            total_loss = 0.0
            current_round = epoch % dataset_train.pairing_epochs
            current_dataloader = dataloaders[current_round]  # 🎯 直接取用
            model.train()
            for seq_batch, phys_batch in tqdm(current_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
                seq_batch = seq_batch.to('cuda')      # [B, 21]
                phys_batch = phys_batch.to('cuda')    # [B, 31]
                B, L = seq_batch.shape
                num_qubits = int((L - 1) / 2)

                data_m1 = seq_batch
                # 加噪声
                noisy_data, masked_indices, p_mask = forward_process_b(batch=data_m1, eps=args.eps_b)
                # print(f"noisy_data: {noisy_data}")
                # print(f"shape of noisy_data: {noisy_data.unsqueeze(-1).shape}")
                # print(f"shape of phys_batch: ", phys_batch.shape)
                # print(f"Contains NaN in C: {torch.isnan(phys_batch).any().item()}")
                # print(f"Contains Inf in C: {torch.isinf(phys_batch).any().item()}")
                # print(f"Max value in C: {phys_batch.max().item()}, Min value: {phys_batch.min().item()}")
                pred_b, pred_c = model(x_single=noisy_data.unsqueeze(-1), x_pair=phys_batch, mask_indices = masked_indices)
                # print(f"shape of pred: {pred.shape}")

                # 损失计算
                target = seq_batch[:, 11:21]
                target = target.round().long()
                flat_logits = pred_b.reshape(-1, 2)           # [B*10, 2]
                flat_targets = target.reshape(-1)         # [B*10]
                flat_mask = masked_indices.reshape(-1)             # [B*10]

                # 只对被掩码的位置计算损失
                masked_logits = flat_logits[flat_mask]             # [M, 2]
                masked_targets = flat_targets[flat_mask]           # [M]

                # 损失
                loss = loss_fn(masked_logits, masked_targets)

                optimizer.zero_grad()
                loss.backward()

                total_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5  # L2 范数
                grad_norms.append(total_norm)

                if args.print_grad_per_step:
                    print(f"  Loss: {loss.item():.6f} | Grad Norm: {total_norm:.6f}")

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                    lrs.append(optimizer.param_groups[0]['lr'])

                total_loss += loss.item()

            avg_loss = total_loss / len(current_dataloader)
            epoch_losses.append(avg_loss)
            if args.eval_loss:
                avg_eval_loss = evaluate(model, num_qubits, dataloader_eval, loss_fn, args)
                test_losses.append(avg_eval_loss)
                model.train()
            else:
                avg_eval_loss = None



            avg_grad_norm = sum(grad_norms[-len(current_dataloader):]) / len(current_dataloader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_loss:.8f} | Avg Grad Norm: {avg_grad_norm:.6f} | LR: {current_lr:.2e}")
            if avg_grad_norm < 1e-5:
                print("Gradient norm is too small, stopping training.")
                break 

    else:
        for epoch in range(args.epochs):
            total_loss = 0.0
            current_round = epoch % dataset_train.pairing_epochs
            current_dataloader = dataloaders[current_round]  # 🎯 直接取用
            model.train()
            for seq_batch, phys_batch in tqdm(current_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
                seq_batch = seq_batch.to('cuda')      # [B, 21]
                phys_batch = phys_batch.to('cuda')    # [B, 31]
                B, L = seq_batch.shape
                num_qubits = int((L - 1) / 2)


                pair_data = phys_batch[:, 1:].reshape(B, 10, 3)
                c_pair = pair_data[:, :, 2]   # [B, 10] - 乘积结果

                noisy_data_c, masked_indices_c, p_mask_c = forward_process_c(batch=phys_batch, eps=args.eps_c)
                # print(f"shape of seq_batch: {seq_batch.shape}")
                # print(f"shape of noisy_data_c: {noisy_data_c.unsqueeze(-1).shape}")
                # print(f"noisy_data: {noisy_data}")
                # print(f"shape of noisy_data: {noisy_data.shape}")
                # print(f"noisy_data: {noisy_data_c}")
                # print(f"shape of noisy_data: {noisy_data_c.shape}")            
                # print(phys_batch)
                # print(f"Contains NaN in C: {torch.isnan(phys_batch).any().item()}")
                # print(f"Contains Inf in C: {torch.isinf(phys_batch).any().item()}")
                # print(f"Max value in C: {phys_batch.max().item()}, Min value: {phys_batch.min().item()}")
                pred_b, pred_c = model(x_single=seq_batch.unsqueeze(-1), x_pair=noisy_data_c.unsqueeze(-1), mask_indices_pair=masked_indices_c)

                target_c = c_pair.round().long()
                flat_logits_c = pred_c.reshape(-1, 2)           # [B*10, 2]
                flat_targets_c = target_c.reshape(-1)         # [B*10]
                flat_mask_c = masked_indices_c.reshape(-1)             # [B*10]

                # 只对被掩码的位置计算损失
                masked_logits_c = flat_logits_c[flat_mask_c]             # [M, 2]
                masked_targets_c = flat_targets_c[flat_mask_c]           # [M]


                loss = loss_fn(masked_logits_c, masked_targets_c)

                optimizer.zero_grad()
                loss.backward()

                total_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5  # L2 范数
                grad_norms.append(total_norm)

                if args.print_grad_per_step:
                    print(f"  Loss: {loss.item():.6f} | Grad Norm: {total_norm:.6f}")

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                    lrs.append(optimizer.param_groups[0]['lr'])

                total_loss += loss.item()

            avg_loss = total_loss / len(current_dataloader)
            epoch_losses.append(avg_loss)
            if args.eval_loss:
                avg_eval_loss = evaluate(model, num_qubits, dataloader_eval, loss_fn, args)
                test_losses.append(avg_eval_loss)
                model.train()
            else:
                avg_eval_loss = None



            avg_grad_norm = sum(grad_norms[-len(current_dataloader):]) / len(current_dataloader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_loss:.8f} | Avg Grad Norm: {avg_grad_norm:.6f} | LR: {current_lr:.2e}")
            if avg_grad_norm < 1e-5:
                print("Gradient norm is too small, stopping training.")
                break 


    # 保存模型
    torch.save(model.state_dict(), args.model_save_path)
    print("Model saved.")

    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss', color='blue', linewidth=2)
    # plt.xlabel('Epoch', fontsize=12)
    # plt.ylabel('Average Loss', fontsize=12)
    # plt.title('Training Loss Curve', fontsize=14)
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()

    # # 保存图像
    # loss_plot_path = args.loss_fig_save_path
    # plt.savefig(loss_plot_path)
    # print(f"📉 Loss curve saved to: {loss_plot_path}")

    train_log = {
        "args": vars(args),  # 保存本次训练的超参数
        "epoch_losses": epoch_losses,
        "test_losses": test_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "total_epochs": len(epoch_losses)
    }
    log_save_path = args.loss_save_path

    # 检查文件是否存在，如果存在则加载旧数据
    if os.path.exists(log_save_path):
        with open(log_save_path, 'r') as f:
            try:
                logs = json.load(f)  # 已有的训练日志列表
            except json.JSONDecodeError:
                logs = []  # 如果文件为空或损坏，从空列表开始
    else:
        logs = []  # 文件不存在，初始化为空列表

    # 追加本次训练日志
    logs.append(train_log)

    # 写回文件
    with open(log_save_path, 'w') as f:
        json.dump(logs, f, indent=4)

    print(f"✅ Training log appended to: {log_save_path}")


if __name__ == "__main__":
    main()