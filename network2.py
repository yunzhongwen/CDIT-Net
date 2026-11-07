import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

# 自动选择设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_group_norm(channels, max_groups=8):
    """自适应 GroupNorm 工具函数，防止组数不整除通道数，彻底替代 InstanceNorm 以保留焦点方差特征"""
    groups = max_groups
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


# =========================================================================
# 基础增强模块：深度残差块 (ResBlock)
# =========================================================================
class ResBlock(nn.Module):
    """带 GroupNorm 的增强型残差模块，提升局部高频与梯度特征表征能力"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.norm1 = get_group_norm(out_channels)
        self.act = nn.GELU()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.norm2 = get_group_norm(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride
                ),
                get_group_norm(out_channels),
            )

    def forward(self, x):
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + self.shortcut(x))


# =========================================================================
# 核心模块 1：扩张卷积投影 (Dilated Convolutional Projection, DCP)
# =========================================================================
class DilatedConvProjection(nn.Module):
    """扩张卷积投影 (DCP)：替代传统 Transformer 中的线性映射，注入局部空间与多感受野信息"""

    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=embed_dim,
            ),
            # 替换为 GroupNorm 以保留跨图像域的焦点对比度与强度方差
            get_group_norm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)


# =========================================================================
# 核心模块 2：优化版高频极值感知局部注意力
# =========================================================================
class LocalAttention(nn.Module):
    """优化版局部注意力：利用空间维度的 Max/Avg 极值突显焦点高频边缘与显著梯度"""

    def __init__(self, channels):
        super().__init__()
        # 利用大核卷积在空间维度上融合极值图
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        # 深度可分离卷积进行细节增强
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, groups=channels
            ),
            get_group_norm(channels),
            nn.GELU(),
        )

    def forward(self, x):
        # 提取空间维度的最大值与平均值（最大值响应可极大地突显聚焦高频边缘）
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_map = torch.cat([max_out, avg_out], dim=1)

        attn = self.spatial_gate(spatial_map)
        x_attn = x * attn
        return x + self.detail_enhance(x_attn)


class MixFFN(nn.Module):
    """混合前馈网络 (Mix-FFN)"""

    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x_spatial = x.transpose(1, 2).view(B, -1, H, W)
        x_spatial = self.dwconv(x_spatial)
        x = x_spatial.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class InteractiveTransformerBlock(nn.Module):
    """瓶颈层交互 Transformer 模块 (融入 DCP 投影与 SRA 空间降维保护)"""

    def __init__(self, embed_dim, num_heads, reduction_size=32, dropout=0.1):
        super().__init__()
        self.reduction_size = reduction_size

        self.dcp_q = DilatedConvProjection(embed_dim)
        self.dcp_kv = DilatedConvProjection(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = MixFFN(embed_dim, embed_dim * 4, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.pool = nn.AdaptiveAvgPool2d((reduction_size, reduction_size))

    def forward(self, query_spatial, kv_spatial):
        B, C, H, W = query_spatial.shape

        query_spatial = self.dcp_q(query_spatial)
        kv_spatial = self.dcp_kv(kv_spatial)

        if H > self.reduction_size or W > self.reduction_size:
            q_reduced = self.pool(query_spatial)
            kv_reduced = self.pool(kv_spatial)

            q_flat = q_reduced.flatten(2).transpose(1, 2)
            kv_flat = kv_reduced.flatten(2).transpose(1, 2)

            attn_output, _ = self.cross_attn(q_flat, kv_flat, kv_flat)
            out1 = self.norm1(q_flat + self.dropout(attn_output))
            out2 = self.norm2(
                out1
                + self.ffn(
                    out1, self.reduction_size, self.reduction_size
                )
            )

            out2_spatial = out2.transpose(1, 2).view(
                B, C, self.reduction_size, self.reduction_size
            )
            return F.interpolate(
                out2_spatial, size=(H, W), mode="bilinear", align_corners=False
            )
        else:
            q_flat = query_spatial.flatten(2).transpose(1, 2)
            kv_flat = kv_spatial.flatten(2).transpose(1, 2)

            attn_output, _ = self.cross_attn(q_flat, kv_flat, kv_flat)
            out1 = self.norm1(q_flat + self.dropout(attn_output))
            out2 = self.norm2(out1 + self.ffn(out1, H, W))

            return out2.transpose(1, 2).view(B, C, H, W)


class GlobalLocalAttention(nn.Module):
    """全局-局部注意力 (GLA)：加入 SRA 空间降维保护，彻底解决解码器大图推理 OOM"""

    def __init__(self, channels, num_heads=4, reduction_size=32, dropout=0.1):
        super().__init__()
        self.reduction_size = reduction_size
        self.local_attn = LocalAttention(channels)
        self.global_attn = nn.MultiheadAttention(
            channels, num_heads, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(channels)
        self.pool = nn.AdaptiveAvgPool2d((reduction_size, reduction_size))
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            get_group_norm(channels),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape

        local_feat = self.local_attn(x)

        if H > self.reduction_size or W > self.reduction_size:
            x_reduced = self.pool(x)
            q_flat = x_reduced.flatten(2).transpose(1, 2)

            global_feat_out, _ = self.global_attn(q_flat, q_flat, q_flat)
            global_feat_out = self.norm(q_flat + self.dropout(global_feat_out))

            global_feat_small = global_feat_out.transpose(1, 2).view(
                B, C, self.reduction_size, self.reduction_size
            )
            global_feat = F.interpolate(
                global_feat_small,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
        else:
            q_flat = x.flatten(2).transpose(1, 2)
            global_feat_out, _ = self.global_attn(q_flat, q_flat, q_flat)
            global_feat_out = self.norm(q_flat + self.dropout(global_feat_out))
            global_feat = global_feat_out.transpose(1, 2).view(B, C, H, W)

        concat_feat = torch.cat([local_feat, global_feat], dim=1)
        weight_mask = self.gate(concat_feat)
        return weight_mask * local_feat + (1 - weight_mask) * global_feat


class ChannelAttention(nn.Module):
    """通道注意力机制"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class Output(nn.Module):
    """深度残差细化模块"""

    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        mid_channels = int(in_channels * 1.5)
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            get_group_norm(mid_channels),
            nn.GELU(),
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
            get_group_norm(out_channels),
            nn.GELU(),
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, kernel_size=1),
            get_group_norm(in_channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        return self.act(x + residual)


# =========================================================================
# 核心模块 3：跨域交互 Transformer (CDIT)
# =========================================================================
class CDIT(nn.Module):
    """跨域交互 Transformer：实现判别分支特征与生成分支特征的软共享交互"""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            get_group_norm(channels),
            nn.Sigmoid(),
        )

    def forward(self, dec_feat, gen_feat):
        concat_feat = torch.cat([dec_feat, gen_feat], dim=1)
        gate_val = self.gate(concat_feat)
        dec_enhanced = dec_feat + gate_val * gen_feat
        gen_enhanced = gen_feat + (1.0 - gate_val) * dec_feat
        return dec_enhanced, gen_enhanced


# =========================================================================
# 核心模块 4：焦点/离焦边界 (FDB) 提取器
# =========================================================================
class FDBBoundaryExtractor(nn.Module):
    """焦点/离焦边界 (FDB) 提取器：利用形态学膨胀与腐蚀提取决策图边缘"""

    def __init__(self, kernel_size=5):
        super().__init__()
        self.pad = kernel_size // 2
        self.kernel_size = kernel_size

    def forward(self, decision_map):
        dilation = F.max_pool2d(
            decision_map,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.pad,
        )
        erosion = -F.max_pool2d(
            -decision_map,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.pad,
        )
        boundary_mask = dilation - erosion
        return boundary_mask


# =========================================================================
# 主网络：融合三大方案的 PDR-MultiFocusFusionNet Pro (精准优化版)
# =========================================================================
class PDRMultiFocusFusionNet(nn.Module):
    """渐进式决策修复多聚焦图像融合网络 (突破 93% 准确率上限精修版)"""

    def __init__(
        self, in_channels=3, out_channels=64, embed_dim=64, reduction_size=32
    ):
        super().__init__()

        # ---------------- 1. 深度残差编码器 (提升高频与梯度表征能力) ----------------
        self.enc1 = ResBlock(in_channels, embed_dim // 2, stride=2)  # H/2, C/2
        self.enc2 = ResBlock(embed_dim // 2, embed_dim, stride=2)  # H/4, C

        # ---------------- 2. 瓶颈层与 DCP 交互 Transformer ----------------
        self.interactive_A_to_B = InteractiveTransformerBlock(
            embed_dim, num_heads=4, reduction_size=reduction_size
        )
        self.interactive_B_to_A = InteractiveTransformerBlock(
            embed_dim, num_heads=4, reduction_size=reduction_size
        )

        self.fusion_bottleneck = nn.Sequential(
            ChannelAttention(embed_dim * 2),
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1),
            get_group_norm(embed_dim),
            nn.GELU(),
        )

        # 粗决策预测头 (1/4 分辨率)
        self.coarse_head = nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1)

        # ---------------- 3. 解码 Level 1 (1/2 分辨率) ----------------
        # [判别分支]
        self.dec_up1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            get_group_norm(embed_dim),
            nn.GELU(),
        )
        self.gla_level1 = GlobalLocalAttention(
            embed_dim, reduction_size=reduction_size
        )
        self.dec_fuse1 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1),
            get_group_norm(embed_dim),
            nn.GELU(),
        )
        self.refine_head1 = nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1)

        # [生成分支 Level 1]
        self.gen_up1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            get_group_norm(embed_dim),
            nn.GELU(),
        )

        # [跨域交互模块 CDIT 1]
        self.cdit1 = CDIT(embed_dim)

        # ---------------- 4. 解码 Level 2 (全分辨率 1x) ----------------
        # [判别分支]
        self.dec_up2 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            get_group_norm(out_channels),
            nn.GELU(),
        )
        self.raw_feat_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            get_group_norm(out_channels),
            nn.GELU(),
        )
        self.gla_level2 = GlobalLocalAttention(
            out_channels, reduction_size=reduction_size
        )
        self.out_block = Output(
            in_channels=out_channels, out_channels=out_channels
        )
        self.refine_head2 = nn.Conv2d(
            out_channels, 1, kernel_size=3, padding=1
        )

        # [生成分支 Level 2 与预测头]
        self.gen_up2 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            get_group_norm(out_channels),
            nn.GELU(),
        )
        self.gen_head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

        # ---------------- 5. FDB 边界提取器 ----------------
        self.fdb_extractor = FDBBoundaryExtractor(kernel_size=5)

    def forward(self, img_A, img_B, return_multi_scale=False):
        # ---------------- Step 1: 深度残差编码提取 ----------------
        enc1_A = self.enc1(img_A)
        enc1_B = self.enc1(img_B)

        feat_A = self.enc2(enc1_A)
        feat_B = self.enc2(enc1_B)

        # ---------------- Step 2: DCP 瓶颈交互 ----------------
        fused_A = self.interactive_A_to_B(feat_A, feat_B)
        fused_B = self.interactive_B_to_A(feat_B, feat_A)

        concat_deep_feat = torch.cat([fused_A, fused_B], dim=1)
        bottleneck_feat = self.fusion_bottleneck(concat_deep_feat)

        logit_0 = self.coarse_head(bottleneck_feat)
        map_0 = torch.sigmoid(logit_0)

        # ---------------- Step 3: Level 1 渐进细化与 CDIT 交互 ----------------
        dec_up1 = self.dec_up1(bottleneck_feat)
        gen_up1 = self.gen_up1(bottleneck_feat)

        if dec_up1.shape[2:] != enc1_A.shape[2:]:
            dec_up1 = F.interpolate(
                dec_up1,
                size=enc1_A.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            gen_up1 = F.interpolate(
                gen_up1,
                size=enc1_A.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        dec_up1, gen_up1 = self.cdit1(dec_up1, gen_up1)

        skip_feat = torch.cat([enc1_A, enc1_B], dim=1)
        skip_feat_fused = self.gla_level1(skip_feat)
        fused_up1 = self.dec_fuse1(
            torch.cat([dec_up1, skip_feat_fused], dim=1)
        )

        delta_logit_1 = self.refine_head1(fused_up1)
        logit_0_up = F.interpolate(
            logit_0,
            size=fused_up1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        logit_1 = logit_0_up + delta_logit_1
        map_1 = torch.sigmoid(logit_1)

        # ---------------- Step 4: Level 2 解码 (双分支输出) ----------------
        dec_up2 = self.dec_up2(fused_up1)
        gen_up2 = self.gen_up2(gen_up1)

        # [判别分支]
        raw_concat = torch.cat([img_A, img_B], dim=1)
        raw_feat = self.raw_feat_conv(raw_concat)
        raw_gla = self.gla_level2(raw_feat)

        out_feat = self.out_block(dec_up2 + raw_gla)
        delta_logit_2 = self.refine_head2(out_feat)
        logit_1_up = F.interpolate(
            logit_1, size=img_A.shape[2:], mode="bilinear", align_corners=False
        )
        logit_final = logit_1_up + delta_logit_2
        map_final = torch.sigmoid(logit_final)

        if map_final.shape[2:] != img_A.shape[2:]:
            map_final = F.interpolate(
                map_final,
                size=img_A.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # [生成分支：生成全彩融合图像]
        img_gen = self.gen_head(gen_up2 + raw_gla)
        if img_gen.shape[2:] != img_A.shape[2:]:
            img_gen = F.interpolate(
                img_gen,
                size=img_A.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # ---------------- Step 5: FDB 复合融合策略 ----------------
        map_detached = map_final.detach()

        img_hard = img_A * map_detached + img_B * (1.0 - map_detached)
        m_edge = self.fdb_extractor(map_detached)
        img_fused = (1.0 - m_edge) * img_hard + m_edge * img_gen

        # ---------------- Step 6: 模式输出 ----------------
        if self.training or return_multi_scale:
            return img_fused, (map_final, map_1, map_0), img_gen
        else:
            return img_fused, map_final


# =========================================================================
# 测试验证脚本
# =========================================================================
if __name__ == "__main__":
    print(
        "正在构建精修版 PDR 多聚焦图像融合网络 (突破 93% 准确率上限)... \n"
    )
    model = PDRMultiFocusFusionNet().to(DEVICE)
    model.eval()

    # ---------------------------------------------------------
    # 测试 1: 常规分辨率 (计算标准参数量与 FLOPs)
    # ---------------------------------------------------------
    print("=== 常规测试 (输入 256x256) ===")
    test_tensor_A = torch.rand((1, 3, 256, 256)).to(DEVICE)
    test_tensor_B = torch.rand((1, 3, 256, 256)).to(DEVICE)

    flops, params = profile(
        model, inputs=(test_tensor_A, test_tensor_B), verbose=False
    )
    print("✅ 计算完成！")
    print(f"📌 FLOPs (计算量): {flops / 1e9:.3f} G")
    print(f"📌 Params (参数量): {params / 1e6:.3f} M")

    # ---------------------------------------------------------
    # 测试 2: 多任务训练模式输出测试 (深监督 + 生成损失)
    # ---------------------------------------------------------
    print("\n=== 多任务训练模式测试 (Training Mode) ===")
    model.train()
    img_fused, dec_maps, img_gen = model(test_tensor_A, test_tensor_B)
    print("✅ 成功生成多任务输出：")
    print(f"   - 最终 FDB 复合融合图形状:   {img_fused.shape}")
    print(f"   - 1x 判别决策图 M_final 形状:  {dec_maps[0].shape}")
    print(f"   - 1/2 阶中间决策图 M_1 形状:   {dec_maps[1].shape}")
    print(f"   - 1/4 阶粗略决策图 M_0 形状:   {dec_maps[2].shape}")
    print(f"   - 生成分支全彩图像 I_gen 形状: {img_gen.shape}")

    # ---------------------------------------------------------
    # 测试 3: 高分辨率极限压力测试 (1024x1024)
    # ---------------------------------------------------------
    print("\n=== 高分辨率极限压力测试 (输入 1024x1024) ===")
    model.eval()
    high_res_A = torch.rand((1, 3, 1024, 1024)).to(DEVICE)
    high_res_B = torch.rand((1, 3, 1024, 1024)).to(DEVICE)

    start = time.time()
    with torch.no_grad():
        final_fused, final_map = model(high_res_A, high_res_B)
    end = time.time()

    print("✅ 高分辨率测试成功！(显存安全，无 OOM 卡死)")
    print(f"📌 单次大图推理耗时: {end - start:.3f} 秒")
    print(f"📌 输出复合融合图形状: {final_fused.shape}")
    print(f"📌 输出决策图形状:     {final_map.shape}")