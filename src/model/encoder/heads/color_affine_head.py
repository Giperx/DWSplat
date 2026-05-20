import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

class ColorAffineHead(nn.Module):
    def __init__(self, in_dim=3072, num_dino_layers=24):
        super().__init__()

        # ==========================================
        # 创新点：可学习的 DINO 层级融合权重
        # ==========================================
        # 初始化 24 层的权重为 0，经过 softmax 后就是均等的 1/24
        self.dino_layer_weights = nn.Parameter(torch.zeros(num_dino_layers))

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 12) # 输出 12 维
        )
        # 保持零初始化，保证初始输出恒等变换
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # 定义物理边界超参数
        self.scale_range = 0.5
        self.shift_range = 0.2

    def forward(self, aggregated_tokens_list: List[torch.Tensor], dino_token_list: List[torch.Tensor]):

        # 1. Camera Token: 依然取聚合器最深层，保留最强的空间与全局注意力上下文
        camera_token = aggregated_tokens_list[-1][:, :, 0, :]  # (B, S, 2*embed_dim)

        # ==========================================
        # 2. DINO Tokens: 自适应多层加权融合
        # ==========================================
        # 计算每一层的 softmax 权重 (维度: [24])
        layer_weights = F.softmax(self.dino_layer_weights, dim=0)

        # 提取每一层的 patch tokens 并堆叠 -> [24, B, S, 1369, dino_dim]
        # 注意：dino_token_list 中的 tensor shape 为 [B, S, 1374, dino_dim]
        all_layers_patch_tokens = torch.stack([l[:, :, 5:, :] for l in dino_token_list], dim=0)

        # 将权重调整为可广播的形状 [24, 1, 1, 1, 1]
        weight_view = layer_weights.view(-1, 1, 1, 1, 1)

        # 加权求和，得到融合后的特征 -> [B, S, 1369, dino_dim]
        fused_patch_tokens = (all_layers_patch_tokens * weight_view).sum(dim=0)

        # 3. 对融合后的 DINO 特征进行全局测光 (GAP)
        global_light_feat = fused_patch_tokens.mean(dim=2)  # (B, S, dino_dim)

        # 4. 拼接显式测光特征与隐式相机上下文
        x = torch.cat([global_light_feat, camera_token], dim=-1)  # (B, S, 3072)

        B, S, _ = x.shape
        raw_output = self.mlp(x) # [B, S, 12]
        
        # 分离出 W_delta 和 b_delta
        W_delta_raw = raw_output[..., :9].view(B, S, 3, 3)
        b_delta_raw = raw_output[..., 9:].view(B, S, 3)
        
        # 生成单位矩阵 I
        I = torch.eye(3, device=x.device).view(1, 1, 3, 3)
        
        # 边界约束
        W = I + self.scale_range * torch.tanh(W_delta_raw)
        b = self.shift_range * torch.tanh(b_delta_raw)
        
        return W, b


# 输出	Shape	含义
# W	(B, S, 3, 3)	每个视角一个 3×3 颜色仿射矩阵（近似单位阵，元素在 I ± 0.5 内）
# b	(B, S, 3)	每个视角一个 3 维颜色偏移向量（范围 -0.2 ~ 0.2）
# 应用时对每个像素的 RGB 做 color_out = W @ color_in + b，即逐视角的线性颜色校正。

# ----- 在渲染管线中的使用 -----
# W shape: [B, 3, 3, 3], b shape: [B, 3, 3]
# 假设你的 GS 基础颜色 C_base shape: [B, 3, N, 3] (N为高斯点数量或像素数量)

# 进行仿射变换 (这里写成逐像素操作的维度演示)
# C_pred = torch.einsum('bsij, bsnj -> bsni', W, C_base) + b.unsqueeze(2)

# 最后，为了绝对安全，还可以对最终颜色做一次 Sigmoid 或 Clamp
# C_pred = torch.clamp(C_pred, 0.0, 1.0)

# color: (B, S, 3, H, W)
# W:     (B, S, 3, 3)
# b:     (B, S, 3)
# output.color = torch.einsum("bsij, bsjhw -> bsihw", affine_w, color) + affine_b.unsqueeze(-1).unsqueeze(-1)