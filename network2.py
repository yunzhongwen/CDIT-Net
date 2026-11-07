import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_group_norm(channels, max_groups=8):
    groups = max_groups
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
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


class DilatedConvProjection(nn.Module):
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
            get_group_norm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)


class LocalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, groups=channels
            ),
            get_group_norm(channels),
            nn.GELU(),
        )

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_map = torch.cat([max_out, avg_out], dim=1)

        attn = self.spatial_gate(spatial_map)
        x_attn = x * attn
        return x + self.detail_enhance(x_attn)


class MixFFN(nn.Module):
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


class CDIT(nn.Module):
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


class FDBBoundaryExtractor(nn.Module):
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


class PDRMultiFocusFusionNet(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=64, embed_dim=64, reduction_size=32
    ):
        super().__init__()

        self.enc1 = ResBlock(in_channels, embed_dim // 2, stride=2)
        self.enc2 = ResBlock(embed_dim // 2, embed_dim, stride=2)

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

        self.coarse_head = nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1)

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

        self.gen_up1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            get_group_norm(embed_dim),
            nn.GELU(),
        )

        self.cdit1 = CDIT(embed_dim)

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

        self.fdb_extractor = FDBBoundaryExtractor(kernel_size=5)

    def forward(self, img_A, img_B, return_multi_scale=False):
        enc1_A = self.enc1(img_A)
        enc1_B = self.enc1(img_B)

        feat_A = self.enc2(enc1_A)
        feat_B = self.enc2(enc1_B)

        fused_A = self.interactive_A_to_B(feat_A, feat_B)
        fused_B = self.interactive_B_to_A(feat_B, feat_A)

        concat_deep_feat = torch.cat([fused_A, fused_B], dim=1)
        bottleneck_feat = self.fusion_bottleneck(concat_deep_feat)

        logit_0 = self.coarse_head(bottleneck_feat)
        map_0 = torch.sigmoid(logit_0)

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

        dec_up2 = self.dec_up2(fused_up1)
        gen_up2 = self.gen_up2(gen_up1)

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

        img_gen = self.gen_head(gen_up2 + raw_gla)
        if img_gen.shape[2:] != img_A.shape[2:]:
            img_gen = F.interpolate(
                img_gen,
                size=img_A.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        map_detached = map_final.detach()

        img_hard = img_A * map_detached + img_B * (1.0 - map_detached)

        m_edge = self.fdb_extractor(map_detached)
        img_fused = (1.0 - m_edge) * img_hard + m_edge * img_gen

        if self.training or return_multi_scale:
            return img_fused, (map_final, map_1, map_0), img_gen
        else:
            return img_fused, map_final


if __name__ == "__main__":
    model = PDRMultiFocusFusionNet().to(DEVICE)
    model.eval()

    print("=== Standard Test (Input: 256x256) ===")
    test_tensor_A = torch.rand((1, 3, 256, 256)).to(DEVICE)
    test_tensor_B = torch.rand((1, 3, 256, 256)).to(DEVICE)

    flops, params = profile(
        model, inputs=(test_tensor_A, test_tensor_B), verbose=False
    )
    print(f"FLOPs: {flops / 1e9:.3f} G")
    print(f"Params: {params / 1e6:.3f} M")

    print("\n=== Multi-Task Training Mode Test ===")
    model.train()
    img_fused, dec_maps, img_gen = model(test_tensor_A, test_tensor_B)
    print(f"Fused Image Shape: {img_fused.shape}")
    print(f"M_final Shape:     {dec_maps[0].shape}")
    print(f"M_1 Shape:         {dec_maps[1].shape}")
    print(f"M_0 Shape:         {dec_maps[2].shape}")
    print(f"I_gen Shape:       {img_gen.shape}")

    print("\n=== High-Resolution Stress Test (Input: 1024x1024) ===")
    model.eval()
    high_res_A = torch.rand((1, 3, 1024, 1024)).to(DEVICE)
    high_res_B = torch.rand((1, 3, 1024, 1024)).to(DEVICE)

    start = time.time()
    with torch.no_grad():
        final_fused, final_map = model(high_res_A, high_res_B)
    end = time.time()

    print(f"Inference Time:    {end - start:.3f} s")
    print(f"Fused Image Shape: {final_fused.shape}")
    print(f"Decision Map Shape: {final_map.shape}")
