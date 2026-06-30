import argparse
import json
import torch
import numpy as np
import random
from Qmultimodel import QuantumMultimodalDModel_nseq
from test_1 import generate_nseq_both
from torch.utils.data import DataLoader
from train_multi import MultimodalEvalDataset, dual_collate_fn, forward_process_b, forward_process_c
import os
from tqdm import tqdm

### save_path 默认在 /data/fyp26lyh/M-Diffushadow/output


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FM phase consistency across g values")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_qubits', type=int, default=10)
    parser.add_argument('--mask_ratio_b', type=float, default=0.5, help='fraction of b positions to mask')
    parser.add_argument('--mask_ratio_c', type=float, default=0.5, help='fraction of c positions to mask')
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--num_samples_per_g', type=int, default=100, help='Number of samples per g value')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--save_path', type=str, default='./fm_phase_diagram_results.npz')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--b_value', type=int, default=1, choices=[0, 1], help='b value for FM phase (0 or 1)')
    return parser.parse_args()



def build_fm_z_basis_sample(g_value, num_qubits, mask_ratio_b, mask_ratio_c, b_value=1, device='cuda'):
    """
    构造铁磁相、Z基下的理想样本
    - 所有测量基设为Z(4)
    - 所有b值为b_value（全0或全1）
    - 所有c值为1（完全关联）
    """
    
    # ============= 1. 固定所有测量基为Z =============
    P_seq = [4.0] * num_qubits  # 单模态：全是Z基
    Pr_seq = [4.0] * (2 * num_qubits)  # 双模态：r1和r2全是Z基
    
    # ============= 2. 理想铁磁相的真实值 =============
    b_true = [float(b_value)] * num_qubits  # 全0或全1
    c_true = [1.0] * num_qubits  # 全1（完全关联）
    
    # ============= 3. 构建单模态prompt =============
    L = 1 + 2 * num_qubits  # 21
    prompt = torch.full((1, L, 1), -1.0, device=device)
    prompt[0, 0, 0] = float(g_value)  # g值
    prompt[0, 1:1+num_qubits, 0] = torch.tensor(P_seq, device=device)  # 全Z
    
    # b值随机掩码
    b_mask = np.zeros(num_qubits, dtype=bool)
    k_b = int(np.round(mask_ratio_b * num_qubits))
    if k_b > 0:
        masked_idx = np.random.choice(num_qubits, size=k_b, replace=False)
        b_mask[masked_idx] = True
    
    # 填充b值：未掩码的位置填真实值，掩码的位置填-1
    for i in range(num_qubits):
        if not b_mask[i]:
            prompt[0, 1+num_qubits + i, 0] = b_true[i]
        else:
            prompt[0, 1+num_qubits + i, 0] = -1.0
    
    # ============= 4. 构建双模态prompt_c =============
    Lc = 1 + 3 * num_qubits  # 31
    prompt_c = torch.full((1, Lc, 1), -1.0, device=device)
    prompt_c[0, 0, 0] = float(g_value)  # g值
    
    # 填充r1和r2：全是Z(4)
    for i in range(num_qubits):
        start_idx = 1 + i * 3
        prompt_c[0, start_idx, 0] = 4.0    # r1 = Z
        prompt_c[0, start_idx+1, 0] = 4.0  # r2 = Z
    
    # c值随机掩码
    c_mask = np.zeros(num_qubits, dtype=bool)
    k_c = int(np.round(mask_ratio_c * num_qubits))
    if k_c > 0:
        masked_idx_c = np.random.choice(num_qubits, size=k_c, replace=False)
        c_mask[masked_idx_c] = True
    
    # 填充c值：未掩码的位置填真实值(1)，掩码的位置填-1
    for i in range(num_qubits):
        p_pos = 1 + i * 3 + 2
        if not c_mask[i]:
            prompt_c[0, p_pos, 0] = c_true[i]
        else:
            prompt_c[0, p_pos, 0] = -1.0
    
    return prompt, prompt_c, b_true, c_true, b_mask, c_mask


@torch.no_grad()
def evaluate_g_value(model, g_value, args, device):
    """
    评估单个g值下的铁磁相一致性
    """
    model.eval()
    
    # 存储该g值的所有结果
    all_b_consistency = []
    all_c_consistency = []
    all_b_correct_rate = []
    all_c_correct_rate = []
    
    for sample_idx in range(args.num_samples_per_g):
        # 随机选择b_value（0或1），模拟铁磁相的两种可能基态
        b_value = random.choice([0, 1]) if args.b_value is None else args.b_value
        
        # 构造样本
        prompt, prompt_c, b_true, c_true, b_mask, c_mask = build_fm_z_basis_sample(
            g_value=g_value,
            num_qubits=args.num_qubits,
            mask_ratio_b=args.mask_ratio_b,
            mask_ratio_c=args.mask_ratio_c,
            b_value=b_value,
            device=device
        )
        # print(prompt)
        # print(f"prompt c: {prompt_c.cpu().numpy().flatten()}")
        # print(f"prompt: {prompt.cpu().numpy().flatten()}")
        
        # 模型预测
        completed_b, completed_c = generate_nseq_both(
            model=model,
            prompt=prompt,
            prompt_c=prompt_c,
            steps=args.steps,
            mask_id=-1.0,
            num_qubits=args.num_qubits,
            temperature=args.temperature,
            use_sampling=False
        )
        
        # 转换为numpy
        b_pred_full = completed_b[0, :, 0].cpu().numpy()  # [21]
        b_pred = b_pred_full[11:11+args.num_qubits]  # [10] 只取b值
        
        # completed_c: [1, 31, 1] -> 取c值部分（每组第三个）
        c_pred_full = completed_c[0, :, 0].cpu().numpy()  # [31]
        # 方法1：直接索引
        c_pred = c_pred_full[3:31:3]  # [10] 位置3,6,9,...,30
        
        # ============= 一致性指标 =============
        # b值是否全部相同？
        b_unique = np.unique(b_pred)
        b_is_consistent = len(b_unique) == 1
        all_b_consistency.append(b_is_consistent)
        
        # c值是否全部为1？
        c_is_all_one = np.all(c_pred == 1)
        all_c_consistency.append(c_is_all_one)

        # print(b_pred)
        # print(c_pred)
        
        # ============= 准确率指标 =============
        # b值正确率（只计算被掩码的位置）
        masked_b_pred = b_pred[b_mask]
        masked_b_true = np.array(b_true)[b_mask]
        b_correct = np.mean(masked_b_pred == masked_b_true) if len(masked_b_pred) > 0 else 1.0
        all_b_correct_rate.append(b_correct)
        
        # c值正确率（只计算被掩码的位置）
        masked_c_pred = c_pred[c_mask]
        masked_c_true = np.array(c_true)[c_mask]
        c_correct = np.mean(masked_c_pred == masked_c_true) if len(masked_c_pred) > 0 else 1.0
        all_c_correct_rate.append(c_correct)
    
    # 返回该g值的统计结果
    return {
        'g_value': g_value,
        'b_consistency_rate': np.mean(all_b_consistency) * 100,
        'c_consistency_rate': np.mean(all_c_consistency) * 100,
        'b_accuracy': np.mean(all_b_correct_rate) * 100,
        'c_accuracy': np.mean(all_c_correct_rate) * 100,
        'b_accuracy_std': np.std(all_b_correct_rate) * 100,
        'c_accuracy_std': np.std(all_c_correct_rate) * 100,
        'b_consistency_std': np.std(all_b_consistency) * 100,
        'c_consistency_std': np.std(all_c_consistency) * 100,
        'num_samples': args.num_samples_per_g
    }


def main():
    args = parse_args()
    
    # ============= 硬编码g值列表 =============
    G_VALUES = np.linspace(0.0, 0.5, 21).tolist()
    # 更密的采样
    # G_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
    #             1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # 只评估铁磁相
    # G_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8]
    # =========================================
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"🔬 Evaluating FM phase consistency across g values")
    print(f"   g values: {G_VALUES}")
    print(f"   num_qubits = {args.num_qubits}")
    print(f"   mask_ratio_b = {args.mask_ratio_b}, mask_ratio_c = {args.mask_ratio_c}")
    print(f"   num_samples_per_g = {args.num_samples_per_g}")
    print(f"   b_value = {args.b_value} (random if None)")
    print("-" * 60)
    
    # 1. 加载模型
    print(f"Loading model from {args.model_path}...")
    model = QuantumMultimodalDModel_nseq(
        hidden_dim=128,
        num_layers=3,
        head_count=4
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully")
    
    # 2. 遍历所有g值进行评估
    print("\n🚀 Starting evaluation across g values...")
    all_results = []
    
    for g_value in tqdm(G_VALUES, desc="Evaluating g values"):
        results = evaluate_g_value(model, g_value, args, device)
        all_results.append(results)
        
        # 实时打印进度
        print(f"\n   g={g_value:.1f}: B_consistency={results['b_consistency_rate']:.1f}%, "
              f"C_consistency={results['c_consistency_rate']:.1f}%, "
              f"B_acc={results['b_accuracy']:.1f}%, C_acc={results['c_accuracy']:.1f}%")
    
    # 3. 整理最终结果
    final_results = {
        'args': vars(args),
        'g_values': G_VALUES,
        'results': all_results,
        'summary': {
            'g_values': G_VALUES,
            'b_consistency_rates': [r['b_consistency_rate'] for r in all_results],
            'c_consistency_rates': [r['c_consistency_rate'] for r in all_results],
            'b_accuracies': [r['b_accuracy'] for r in all_results],
            'c_accuracies': [r['c_accuracy'] for r in all_results],
            'b_accuracy_stds': [r['b_accuracy_std'] for r in all_results],
            'c_accuracy_stds': [r['c_accuracy_std'] for r in all_results]
        }
    }
    
    # 4. 只保存文件，不输出到终端（除了上面的进度打印）
    base_dir = os.path.dirname(args.save_path)
    base_filename = os.path.basename(args.save_path)
    filename_without_ext = os.path.splitext(base_filename)[0]
    ext = os.path.splitext(base_filename)[1]
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    new_filename = f"{filename_without_ext}_{model_name}{ext}"
    save_path = os.path.join(base_dir, new_filename)
    np.savez(save_path, results=final_results)
    print(f"\n✅ Results saved to {save_path}")
    print(f"   Run 'python -c \"import numpy as np; data = np.load(\\\"{save_path}\\\", allow_pickle=True); print(data.files); print(data[\\'results\\'].item().keys())\"' to inspect")


if __name__ == '__main__':
    main()