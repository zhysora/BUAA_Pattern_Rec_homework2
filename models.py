import torch
from torch import nn
from math import ceil
from lib import SwinTransformer


class SwinFR(nn.Module):
    def __init__(self, in_channel, n_class, img_size, n_feats=96,
                 n_blocks=None, n_heads=None):
        super().__init__()
        n_blocks = n_blocks or [2, 2, 2, 2]
        n_heads = n_heads or [3, 6, 12, 24]
        self.depth = len(n_blocks)

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
        self.encoder.init_weights()

        self.flatten_size = ceil(img_size / 4 / (2 ** (self.depth - 1)))
        self.flatten_size = self.flatten_size ** 2 * n_feats * (2 ** (self.depth - 1))

        self.linear_prob = nn.Sequential(
            nn.Linear(self.flatten_size, n_class),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: (N, C, H, W)
        feat = self.encoder(x)[-1]
        return self.linear_prob(feat.view(x.shape[0], -1))


if __name__ == '__main__':
    x = torch.ones([1, 3, 200, 200])
    model = SwinFR(in_channel=3, img_size=200, n_class=100)
    y = model(x)
    print(y.shape)
    print(y)
