import torch
from torch import nn
import torch.nn.functional as F
import math
from math import ceil
from lib_swin import SwinTransformer
from lib_resnet import ResNet_50, ResNet_101, ResNet_152


class FRModel(nn.Module):
    def __init__(self, in_channel, n_class, img_size, n_feats=96,
                 depth=2, pretrained=None, backbone='Swin', use_arc_face=False):
        super().__init__()
        self.depth = depth
        self.backbone = backbone
        self.use_arc_face = use_arc_face

        if self.backbone == 'Swin':
            n_blocks = [2, 2, 2, 2][:depth]
            n_heads = [3, 6, 12, 24][:depth]

            self.encoder = SwinTransformer(
                in_chans=in_channel,
                embed_dim=n_feats,
                depths=n_blocks,
                num_heads=n_heads,
                window_size=7,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                out_indices=[self.depth - 1],
            )
            if pretrained is None:
                self.encoder.init_weights()
            else:
                checkpoint = torch.load(pretrained)
                self.encoder.load_state_dict(checkpoint['model'], False)

            self.flatten_size = ceil(img_size / 4 / (2 ** (self.depth - 1)))
            self.flatten_size = self.flatten_size ** 2 * n_feats * (2 ** (self.depth - 1))
            # self.embedding_linear = nn.Linear(self.flatten_size, 512)

        elif self.backbone == 'ResNet-50':
            self.encoder = ResNet_50(img_size, in_channel=in_channel)
            self.flatten_size = 512
        elif self.backbone == 'ResNet-101':
            self.encoder = ResNet_101(img_size, in_channel=in_channel)
            self.flatten_size = 512
        elif self.backbone == 'ResNet-152':
            self.encoder = ResNet_152(img_size, in_channel=in_channel)
            self.flatten_size = 512

        if self.use_arc_face:
            self.arc_face = ArcFace(self.flatten_size, n_class)

        self.linear_prob = nn.Sequential(
            nn.Linear(self.flatten_size, n_class)
        )

    def forward(self, x):
        # x: (N, C, H, W)
        if self.backbone == 'Swin':
            feat = self.encoder(x)[-1].view(x.shape[0], -1)
            # feat = self.embedding_linear(feat)
        else:
            feat = self.encoder(x)

        if self.use_arc_face:
            output = self.arc_face(feat)
        else:
            output = self.linear_prob(feat)

        return feat, output

    def device(self):
        return next(self.parameters()).device


class ArcFace(nn.Module):
    def __init__(self, embedding_size, class_num, s=30.0, m=0.50):
        super().__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # drop to CosFace
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        return output * self.s


if __name__ == '__main__':
    a = torch.ones(10)
