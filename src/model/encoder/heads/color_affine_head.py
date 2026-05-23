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
        # self.scale_range = 0.5
        # self.shift_range = 0.2
        
        self.scale_range = 1.0  # 允许 0.0 到 2.0 倍的缩放
        self.shift_range = 0.5  # 允许更极端的偏置
        
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


# ------------------------------

class SpatialColorAffineHead(nn.Module):
    def __init__(self, in_dim=1024):
        super().__init__()
        # 注意：这里只用 DINO 的 patch_tokens，不需要 concat 全局 token 了
        # 我们要预测一个 37x37 的网格，每个网格点有自己的 12 维仿射参数
        self.conv_mlp = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 12, kernel_size=1) # 输出每 patch 12 维参数
        )
        
        # 零初始化依然极其重要
        nn.init.zeros_(self.conv_mlp[-1].weight)
        nn.init.zeros_(self.conv_mlp[-1].bias)
        
        # self.scale_range = 0.5
        # self.shift_range = 0.2
        
        self.scale_range = 1.0  # 允许 0.0 到 2.0 倍的缩放
        self.shift_range = 0.5  # 允许更极端的偏置
        
    def forward(self, aggregated_tokens_list: List[torch.Tensor], dino_token_list: List[torch.Tensor], target_H=294, target_W=518):
        # patch_tokens shape: [B, S, 1369, 1024]
        patch_tokens = dino_token_list[-1][:, :, 5:, :]  # 取最后一层的 patch tokens
        B, S, N, C = patch_tokens.shape
        
        # 1. 恢复空间网格维度
        H_patch, W_patch = target_H // 14, target_W // 14
        x = patch_tokens.transpose(-1, -2).view(B * S, C, H_patch, W_patch) 
        # x shape: [B*S, 1024, 37, 37]
        
        # 2. 预测局部的 12 维参数
        raw_output = self.conv_mlp(x) # [B*S, 12, 37, 37]
        
        # 3. 将 37x37 的极低分辨率参数图，平滑放大到真实的图像分辨率 (如 518x294)
        # 这一步极其关键：双线性插值保证了光度变换在整张图上是丝滑过渡的！
        raw_output_up = F.interpolate(raw_output, size=(target_H, target_W), mode='bilinear', align_corners=False)
        # raw_output_up shape: [B*S, 12, H, W]
        
        # 4. 分离 W 和 b 并应用物理边界
        W_delta_raw = raw_output_up[:, :9, :, :].view(B*S, 3, 3, target_H, target_W)
        b_delta_raw = raw_output_up[:, 9:, :, :].view(B*S, 3, target_H, target_W)
        
        I = torch.eye(3, device=x.device).view(1, 3, 3, 1, 1)
        
        W_grid = I + self.scale_range * torch.tanh(W_delta_raw)
        b_grid = self.shift_range * torch.tanh(b_delta_raw)
        
        return W_grid, b_grid

# =================在渲染管线中的使用=================
# W_grid shape: [B*S, 3, 3, H, W]
# b_grid shape: [B*S, 3, H, W]
# C_render shape: [B*S, 3, H, W]

# 因为 W 已经是逐像素的了，我们需要用 einsum 执行逐像素的 3x3 矩阵乘法
# b c i h w -> batch, channel_out, channel_in, height, width
# C_pred = torch.einsum('b c i h w, b i h w -> b c h w', W_grid, C_render) + b_grid

# 最后严格 clamp
# C_pred = C_pred.clamp(0.0, 1.0)


# ------------------------------

class ConditionedSpatialAffineHead(nn.Module):
    # in_dim = 1024 (Patch) + 2048 (Camera Token) = 3072
    def __init__(self, in_dim=3072): 
        super().__init__()
        
        # 使用 1x1 卷积处理拼接后的特征
        self.conv_mlp = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 12, kernel_size=1) 
        )
        
        # 严格的零初始化
        nn.init.zeros_(self.conv_mlp[-1].weight)
        nn.init.zeros_(self.conv_mlp[-1].bias)
        
        # self.scale_range = 0.5
        # self.shift_range = 0.2
        
        self.scale_range = 1.0  # 允许 0.0 到 2.0 倍的缩放
        self.shift_range = 0.5  # 允许更极端的偏置

    def forward(self, aggregated_tokens_list: List[torch.Tensor], dino_token_list: List[torch.Tensor], target_H=294, target_W=518):
        # patch_tokens shape: [B, S, 1369, 1024]
        # camera_token shape: [B, S, 2048]
        patch_tokens = dino_token_list[-1][:, :, 5:, :]  # 取最后一层的 patch tokens
        camera_token = aggregated_tokens_list[-1][:, :, 0, :]  # 取最后一层的 camera token
        B, S, N, C_patch = patch_tokens.shape
        _, _, C_cam = camera_token.shape
        
        # ==========================================
        # 核心神级操作：全局上下文广播与拼接
        # ==========================================
        # 将 [B, S, 2048] 扩展为 [B, S, 1369, 2048]
        camera_token_expanded = camera_token.unsqueeze(2).expand(-1, -1, N, -1)
        
        # 在通道维度拼接 -> [B, S, 1369, 3072]
        conditioned_tokens = torch.cat([patch_tokens, camera_token_expanded], dim=-1)
        
        # ==========================================
        # 后续逻辑与之前完全一致
        # ==========================================
        # 恢复空间网格维度
        H_patch, W_patch = target_H // 14, target_W // 14
        x = conditioned_tokens.transpose(-1, -2).view(B * S, C_patch + C_cam, H_patch, W_patch) 
        # x shape: [B*S, 3072, 37, 37]
        
        raw_output = self.conv_mlp(x) # [B*S, 12, 37, 37]
        
        # 双线性上采样到原图分辨率
        raw_output_up = F.interpolate(raw_output, size=(target_H, target_W), mode='bilinear', align_corners=False)
        
        # 分离 W 和 b 并应用物理边界
        W_delta_raw = raw_output_up[:, :9, :, :].view(B*S, 3, 3, target_H, target_W)
        b_delta_raw = raw_output_up[:, 9:, :, :].view(B*S, 3, target_H, target_W)
        
        I = torch.eye(3, device=x.device).view(1, 3, 3, 1, 1)
        
        W_grid = I + self.scale_range * torch.tanh(W_delta_raw)
        b_grid = self.shift_range * torch.tanh(b_delta_raw)
        
        return W_grid, b_grid