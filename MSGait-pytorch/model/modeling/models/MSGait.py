import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks


class LocalMask_H(nn.Module):
    def __init__(self, in_channels, out_channels, H_dropping_ratio=0.2, num=5, mask_size=3, kernel_size=(3, 3, 3),
                 stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(LocalMask_H, self).__init__()
        self.H = H_dropping_ratio
        self.mask_size = mask_size
        self.num = num
        self.local_h = BasicConv3d(in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        n, c, s, h, w = x.size()
        Mp = torch.zeros(n, c, s, h, w).cuda()  # 初始化为0
        Mq = torch.ones(n, c, s, h, w).cuda()  # 初始化为1

        # 得到高上的mask条纹
        num_h = int(self.H * h)
        temp_h = range(0, h)
        subset_h = random.sample(temp_h, num_h)

        # 得到每个高的mask条纹上的mask区域
        border_h = range(0, w - self.mask_size)
        sample_h = random.sample(border_h, self.num)

        for r in subset_h:
            for k in sample_h:
                Mp[:, :, :, r, k:k + self.mask_size] = 1  # 将指定区域的Mp置为1
                Mq[:, :, :, r, k:k + self.mask_size] = 0  # 将指定区域的Mq置为0

        x_Mp = x * Mp
        x_Mq = x * Mq
        feature_Mp = self.local_h(x_Mp)
        feature_Mq = self.local_h(x_Mq)
        feature_local = feature_Mp + feature_Mq
        return feature_local


class LocalMask_V(nn.Module):
    def __init__(self, in_channels, out_channels, V_dropping_ratio=0.2, num=5, mask_size=3, kernel_size=(3, 3, 3),
                 stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(LocalMask_V, self).__init__()
        self.V = V_dropping_ratio
        self.mask_size = mask_size
        self.num = num
        self.local_v = BasicConv3d(in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        n, c, s, h, w = x.size()
        Mp = torch.zeros(n, c, s, h, w).cuda()  # 初始化为0
        Mq = torch.ones(n, c, s, h, w).cuda()  # 初始化为1

        # 得到宽上的mask条纹
        num_v = int(self.V * w)
        temp_v = range(0, w)
        subset_v = random.sample(temp_v, num_v)

        # 得到每个宽的mask条纹上的mask区域
        border_v = range(0, h - self.mask_size)
        sample_v = random.sample(border_v, self.num)

        for m in subset_v:
            for n in sample_v:
                Mp[:, :, :, n:n + self.mask_size, m] = 1  # 将指定区域的Mp置为1
                Mq[:, :, :, n:n + self.mask_size, m] = 0  # 将指定区域的Mq置为0

        x_Mp = x * Mp
        x_Mq = x * Mq
        feature_Mp = self.local_v(x_Mp)
        feature_Mq = self.local_v(x_Mq)
        feature_local = feature_Mp + feature_Mq
        return feature_local


class GLMask3d(nn.Module):
    def __init__(self, in_channels, out_channels, H_dropping_ratio=0.2, V_dropping_ratio=0.2, num=5, mask_size=3,
                 fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(GLMask3d, self).__init__()
        self.mask_size = mask_size
        self.num = num
        self.fm_sign = fm_sign

        self.local_3d_h = LocalMask_H(
            in_channels=in_channels, out_channels=out_channels, H_dropping_ratio=H_dropping_ratio, num=num,
            mask_size=mask_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, **kwargs)

        self.local_3d_v = LocalMask_V(
            in_channels=in_channels, out_channels=out_channels, V_dropping_ratio=V_dropping_ratio, num=num,
            mask_size=mask_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, **kwargs)

        self.bn = nn.BatchNorm3d(out_channels)

        self.global_3d = BasicConv3d(in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        feature_local_h = self.local_3d_h(x)
        feature_local_v = self.local_3d_v(x)
        feature_local = feature_local_h + feature_local_v

        feature_global = self.global_3d(x)

        if not self.fm_sign:
            feat = F.leaky_relu(feature_local) + F.leaky_relu(feature_global)
        else:
            feat = F.leaky_relu(torch.cat([feature_local, feature_global], dim=3))
        return self.bn(feat)



class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1) * p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p]
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class MSGait(BaseModel):

    def __init__(self, *args, **kargs):
        super(MSGait, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        dropping_ratio = model_cfg['dropping_ratio']

        # For CASIA-B or other unstated datasets.
        self.conv3d = nn.Sequential(
            BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.GLMaskA0 = GLMask3d(
            in_c[0], in_c[1], H_dropping_ratio=dropping_ratio[0], V_dropping_ratio=dropping_ratio[1], num=10,
            mask_size=3, fm_sign=False, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        self.MaxPool0 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.GLMaskA1 = GLMask3d(
            in_c[1], in_c[2], H_dropping_ratio=dropping_ratio[0], V_dropping_ratio=dropping_ratio[1], num=5,
            mask_size=3, fm_sign=False, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        self.GLMaskB2 = GLMask3d(
            in_c[2], in_c[2], H_dropping_ratio=dropping_ratio[0], V_dropping_ratio=dropping_ratio[1], num=5,
            mask_size=3, fm_sign=True, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = GeMHPP()

        self.Head0 = SeparateFCs(64, in_c[-1], in_c[-1])

        if 'SeparateBNNecks' in model_cfg.keys():
            self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
            self.Bn_head = False
        else:
            self.Bn = nn.BatchNorm1d(in_c[-1])
            self.Head1 = SeparateFCs(64, in_c[-1], class_num)
            self.Bn_head = True

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)  # [n, 1, s, h, w]
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d(sils)  # [n, c0, s1, h ,w]

        outs = self.GLMaskA0(outs)  # [n, c1, s1, h, w]

        outs = self.GLMaskA1(outs)  # [n, c2, s0, h, w]
        outs = self.MaxPool0(outs)  # spatial pooling: [n, c2, s0, h, w]  ->  [n, c2, s0, h/2, w/2]

        outs = self.GLMaskB2(outs)  # [n, c2, s0, h, w/2]

        outs = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c2, h, w/2]
        outs = self.HPP(outs)  # [n, c, p]

        gait = self.Head0(outs)  # [n, c, p]

        if self.Bn_head:  # Original GaitGL Head
            bnft = self.Bn(gait)  # [n, c, p]
            logi = self.Head1(bnft)  # [n, c, p]
            embed = bnft
        else:  # BNNechk as Head
            bnft, logi = self.BNNecks(gait)  # [n, c, p]
            embed = gait

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n * s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': bnft
            }
        }
        return retval
