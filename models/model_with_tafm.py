import torch
import torch.nn as nn
import torch.nn.functional as F


# LWGANet backbone import
try:
    from . import LWGANet as RealLWGANet

    print("Successfully imported the LWGANet backbone.")


    class LWGANet:
        @staticmethod
        def LWGANet_L2_1242_e96_k11_RELU(pretrained=True):
            return RealLWGANet.LWGANet_L2_1242_e96_k11_RELU(pretrained=pretrained)

except (ImportError, ModuleNotFoundError) as e:
    print(f"警告: 无法导入真实的 LWGANet (错误: {e})。将使用占位符。")


    class MockLWGANet(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Sequential(nn.Conv2d(3, 96, 4, 4))
            self.c2 = nn.Sequential(nn.Conv2d(96, 192, 2, 2))
            self.c3 = nn.Sequential(nn.Conv2d(192, 384, 2, 2))
            self.c4 = nn.Sequential(nn.Conv2d(384, 768, 2, 2))

        def forward(self, x):
            c1 = self.c1(x)
            c2 = self.c2(c1)
            c3 = self.c3(c2)
            c4 = self.c4(c3)
            return c1, c2, c3, c4


    class LWGANet:
        @staticmethod
        def LWGANet_L2_1242_e96_k11_RELU(pretrained=True): return MockLWGANet()


# ====================== TAFM 和解码器模块 ======================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__();
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False));
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(F.adaptive_avg_pool2d(x, 1));
        max_out = self.fc(F.adaptive_max_pool2d(x, 1));
        return self.sigmoid(avg_out + max_out)


class EnhancedTFFModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(EnhancedTFFModule, self).__init__()
        self.conv_fusion = nn.Sequential(nn.Conv2d(in_d * 3, out_d, 3, 1, 1, bias=False), nn.BatchNorm2d(out_d),
                                         nn.ReLU(True))
        self.channel_att = ChannelAttention(out_d)
        self.out_conv = nn.Sequential(nn.Conv2d(out_d, out_d, 3, 1, 1, bias=False), nn.BatchNorm2d(out_d),
                                      nn.ReLU(True))

    def forward(self, x1, x2):
        x_fused = self.conv_fusion(torch.cat([x1, x2, torch.abs(x1 - x2)], dim=1))
        x_refined = x_fused * self.channel_att(x_fused);
        return self.out_conv(x_refined) + x_fused

#对应TAFM模块
class TemporalFusionModuleWrapper(nn.Module):
    def __init__(self, in_d=64, out_d=64):
        super(TemporalFusionModuleWrapper, self).__init__()
        self.tffm_x2, self.tffm_x3, self.tffm_x4, self.tffm_x5 = [EnhancedTFFModule(in_d, out_d) for _ in range(4)]

    def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):
        c2, c3, c4, c5 = self.tffm_x2(x1_2, x2_2), self.tffm_x3(x1_3, x2_3), self.tffm_x4(x1_4, x2_4), self.tffm_x5(
            x1_5, x2_5)
        return c2, c3, c4, c5


class GuidedRefinementModule(nn.Module):
    def __init__(self, in_d):
        super(GuidedRefinementModule, self).__init__()
        self.cls = nn.Conv2d(in_d, 1, kernel_size=1)
        self.refine_conv = nn.Sequential(nn.Conv2d(in_d, in_d, 3, 1, 1, bias=False), nn.BatchNorm2d(in_d),
                                         nn.ReLU(True))

    def forward(self, x):
        mask_logits = self.cls(x)
        x_out = self.refine_conv(x * torch.sigmoid(mask_logits))
        return x_out, mask_logits

#对应论文HAGD模块
class UnifiedDecoder(nn.Module):
    def __init__(self, mid_d=64):
        super(UnifiedDecoder, self).__init__()
        self.refine_p5, self.refine_p4, self.refine_p3 = [GuidedRefinementModule(mid_d) for _ in range(3)]
        self.fuse_p4 = nn.Sequential(nn.Conv2d(mid_d + 1, mid_d, 3, 1, 1, bias=False), nn.BatchNorm2d(mid_d),
                                     nn.ReLU(True))
        self.fuse_p3 = nn.Sequential(nn.Conv2d(mid_d + 1, mid_d, 3, 1, 1, bias=False), nn.BatchNorm2d(mid_d),
                                     nn.ReLU(True))
        self.fuse_p2 = nn.Sequential(nn.Conv2d(mid_d + 1, mid_d, 3, 1, 1, bias=False), nn.BatchNorm2d(mid_d),
                                     nn.ReLU(True))
        self.cls_p2 = nn.Conv2d(mid_d, 1, kernel_size=1)

    def forward(self, d2, d3, d4, d5):
        p5_ref, m5 = self.refine_p5(d5);
        p5_up = F.interpolate(p5_ref, scale_factor=2, mode='bilinear', align_corners=False)
        p4_comb = self.fuse_p4(
            torch.cat([d4, F.interpolate(torch.sigmoid(m5), scale_factor=2, mode='bilinear', align_corners=False)],
                      1)) + p5_up
        p4_ref, m4 = self.refine_p4(p4_comb);
        p4_up = F.interpolate(p4_ref, scale_factor=2, mode='bilinear', align_corners=False)
        p3_comb = self.fuse_p3(
            torch.cat([d3, F.interpolate(torch.sigmoid(m4), scale_factor=2, mode='bilinear', align_corners=False)],
                      1)) + p4_up
        p3_ref, m3 = self.refine_p3(p3_comb);
        p3_up = F.interpolate(p3_ref, scale_factor=2, mode='bilinear', align_corners=False)
        p2_comb = self.fuse_p2(
            torch.cat([d2, F.interpolate(torch.sigmoid(m3), scale_factor=2, mode='bilinear', align_corners=False)],
                      1)) + p3_up
        m2 = self.cls_p2(p2_comb)
        return m2, m3, m4, m5


# ====================== 最终模型 ======================
class ModelToVisualize(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = LWGANet.LWGANet_L2_1242_e96_k11_RELU(pretrained=pretrained)
        backbone_channels = [96, 192, 384, 768]
        self.mid_d = 64
        self.proj_2 = nn.Conv2d(backbone_channels[0], self.mid_d, kernel_size=1)
        self.proj_3 = nn.Conv2d(backbone_channels[1], self.mid_d, kernel_size=1)
        self.proj_4 = nn.Conv2d(backbone_channels[2], self.mid_d, kernel_size=1)
        self.proj_5 = nn.Conv2d(backbone_channels[3], self.mid_d, kernel_size=1)
        self.tfm = TemporalFusionModuleWrapper(in_d=self.mid_d, out_d=self.mid_d)
        self.decoder = UnifiedDecoder(self.mid_d)
        print(f"模型 'ModelToVisualize'初始化完成。")

    def forward(self, x1, x2, domain_label_t1=None, domain_label_t2=None):
        x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)
        s1_2, s1_3, s1_4, s1_5 = self.proj_2(x1_2), self.proj_3(x1_3), self.proj_4(x1_4), self.proj_5(x1_5)
        s2_2, s2_3, s2_4, s2_5 = self.proj_2(x2_2), self.proj_3(x2_3), self.proj_4(x2_4), self.proj_5(x2_5)
        c2, c3, c4, c5 = self.tfm(s1_2, s1_3, s1_4, s1_5, s2_2, s2_3, s2_4, s2_5)
        m2, m3, m4, m5 = self.decoder(c2, c3, c4, c5)
        masks = [torch.sigmoid(F.interpolate(m, scale_factor=s, mode='bilinear', align_corners=False)) for m, s in
                 zip([m2, m3, m4, m5], [4, 8, 16, 32])]
        return masks

