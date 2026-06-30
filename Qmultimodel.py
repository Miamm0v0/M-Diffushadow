import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumMultimodalDModel_nseq(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=3, head_count=4, qubit=10, length_single=21, length_pair=31):
        super().__init__()
        self.qubit = qubit
        self.length_single = length_single  # 单模态长度: 1 + 2*10 = 21
        self.length_pair = length_pair      # 双模态长度: 1 + 3*10 = 31
        
        # 位置嵌入
        self.position_embedding = nn.Embedding(41, hidden_dim)  # 1 + 10 + 10 + 10 + 10 = 41
        
        # 类型嵌入
        self.type_embedding = nn.Embedding(4, hidden_dim)  # 0:g, 1:r, 2:b, 3:pair_product
        
        self.g_proj = nn.Linear(1, hidden_dim)
        self.base_P_embedding = nn.Embedding(3, hidden_dim)  # 基础X,Y,Z嵌入: 0,1,2
        self.b_embedding = nn.Embedding(3, hidden_dim)  # b: 0,1,2(mask)
        self.product_embedding = nn.Embedding(2, hidden_dim)  # 乘积: 0,1
        
        # 组合投影层：将两个Pauli嵌入组合成二维表示
        self.pair_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, head_count) for _ in range(num_layers)
        ])
        self.output_layer_b = nn.Linear(hidden_dim, 2)  
        self.output_layer_c = nn.Linear(hidden_dim, 2)
        self.MASK_TOKEN_ID = -1.0

    def forward(self, x_single, x_pair, mask_indices=None, mask_indices_pair=None, return_attn_weights=False):
        batch_size = x_single.shape[0]

        # print("x_pair:", x_pair)
        # print("the shape of x_pair:", x_pair.shape)
        # print("the shape of x_single:", x_single.shape)
        
        # 处理单模态输入
        g_single = x_single[:, :1]  # [B, 1, 1]
        r_single = x_single[:, 1:1+self.qubit].squeeze(-1)  # [B, 10]
        b_single = x_single[:, 1+self.qubit:self.length_single].squeeze(-1)  # [B, 10]

        # print("g_single:", g_single)
        # print("r_single:", g_single)
        # print("g_single:", g_single)

        # 单模态r: 使用基础Pauli嵌入
        r_single_tokens = (r_single - 2).long()  # 2->0, 3->1, 4->2
        b_single_tokens = b_single.long().clone()

        b_single_tokens[mask_indices] = 2
        
        # print("r_single_tokens:", r_single_tokens)
        # print("b_single_tokens:", b_single_tokens)

        
        x_g = self.g_proj(g_single)  # [B, 1, D]
        x_r_single = self.base_P_embedding(r_single_tokens)  # [B, 10, D] - 基础嵌入
        
        x_b_single = self.b_embedding(b_single_tokens)  # [B, 10, D]
        # print("the shape of x_pair:", x_pair.shape)
        # 处理双模态输入
        g_pair = x_pair[:, :1]  # [B, 1, 1]
        # print("g_pair:", g_pair)
        pair_data = x_pair[:, 1:].reshape(batch_size, self.qubit, 3)  # [B, 10, 3]
        # print("pair_data:", pair_data)
        r1_pair = pair_data[:, :, 0]  # [B, 10] - 第一个r
        r2_pair = pair_data[:, :, 1]  # [B, 10] - 第二个r
        p_pair = pair_data[:, :, 2]   # [B, 10] - 乘积结果
        # print("r1_pair:", r1_pair)
        # print("r2_pair:", r2_pair)
        # 双模态r: 使用相同的基础Pauli嵌入 + 组合投影
        r1_tokens = (r1_pair - 2).long()  # 2->0, 3->1, 4->2
        r2_tokens = (r2_pair - 2).long()  # 2->0, 3->1, 4->2
        
        r1_emb = self.base_P_embedding(r1_tokens)  # [B, 10, D] - 相同的基础嵌入
        r2_emb = self.base_P_embedding(r2_tokens)  # [B, 10, D] - 相同的基础嵌入
        
        # 组合两个Pauli嵌入
        x_r_pair = self.pair_proj(torch.cat([r1_emb, r2_emb], dim=-1))  # [B, 10, D]
        
        p_tokens = p_pair.long()  # [B, 10] - 0,1

        p_tokens[mask_indices_pair] = 2
        x_c = self.b_embedding(p_tokens)  # [B, 10, D]
        
        # 构建最终序列
        x = torch.cat([
            x_g,                    # g [B, 1, D]
            x_r_single,             # 单模态r [B, 10, D]
            x_b_single,             # 单模态b [B, 10, D]
            x_r_pair,               # 双模态Pauli组合 [B, 10, D]
            x_c                     # 双模态乘积结果 [B, 10, D]
        ], dim=1)  # [B, 41, D]
        
        # 位置嵌入
        pos_ids = torch.arange(41, device=x.device).expand(batch_size, -1)
        x = x + self.position_embedding(pos_ids)
        
        # 类型嵌入
        type_ids = torch.zeros(41, dtype=torch.long, device=x.device)
        type_ids[0] = 0              # g
        type_ids[1:11] = 1           # 单模态r
        type_ids[11:21] = 2          # 单模态b  
        type_ids[21:31] = 1          # 双模态Pauli组合(复用r类型)
        type_ids[31:41] = 3          # 双模态乘积结果
        x = x + self.type_embedding(type_ids).unsqueeze(0)

        # Transformer层
        if return_attn_weights:
            attn_maps = []
            for layer in self.layers:
                x = layer(x)
                attn_maps.append(layer.attention_weights)
        else:
            for layer in self.layers:
                x = layer(x)

        if return_attn_weights:
            return attn_maps
        
        if mask_indices is not None:
            output_b = self.output_layer_b(x[:, 11:21])
        else:
            output_b = None
        if mask_indices_pair is not None:
            output_c = self.output_layer_c(x[:, 31:41])
        else:
            output_c = None
        
        # 输出b的预测
        return output_b, output_c # 预测单模态的b部分(11:21)和多模态部分(31:41) [B, 10, 2]
    

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, head_count, dropout=0.0, ffn_multiplier=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Self-Attention部分
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, head_count, 
            batch_first=True,
            dropout=dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-Forward部分
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_multiplier * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_multiplier * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # Self-Attention + Residual
        residual = x
        x = self.norm1(x)
        attn_out, attn_weights = self.attention(
            x, x, x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False
        )
        # print("attn_out shape:", attn_out.shape)
        # print("attn_weights shape:", attn_weights.shape)
        self.attention_weights = attn_weights.detach().cpu()
        x = residual + self.dropout1(attn_out)
        
        # Feed-Forward + Residual
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout2(ffn_out)
        
        return x