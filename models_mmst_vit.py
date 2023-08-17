import torch
from torch import nn
from einops import rearrange, repeat

from attention import SpatialTransformer, TemporalTransformer

from models_pvt_simclr import PVTSimCLR


class MMST_ViT(nn.Module):
    def __init__(self, out_dim=2, num_grid=64, num_short_term_seq=6, num_long_term_seq=12, num_year=5,
                 pvt_backbone=None, context_dim=9, dim=192, batch_size=64, depth=4, heads=3, pool='cls', dim_head=64,
                 dropout=0., emb_dropout=0., scale_dim=4, ):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.batch_size = batch_size
        self.pvt_backbone = pvt_backbone

        self.proj_context = nn.Linear(num_year * num_long_term_seq * context_dim, num_short_term_seq * dim)
        # self.proj_context = nn.Linear(num_year * num_long_term_seq * context_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_short_term_seq, num_grid, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = SpatialTransformer(dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = TemporalTransformer(dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.norm1 = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )

    def forward_features(self, x, ys):
        x = rearrange(x, 'b t g c h w -> (b t g) c h w')
        ys = rearrange(ys, 'b t g n d -> (b t g) n d')

        # prevent the number of grids from being too large to cause out of memory
        B = x.shape[0]
        n = B // self.batch_size if B % self.batch_size == 0 else B // self.batch_size + 1

        x_hat = torch.empty(0).to(x.device)
        for i in range(n):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            x_tmp = x[start:end]
            ys_tmp = ys[start:end]

            x_hat_tmp = self.pvt_backbone(x_tmp, context=ys_tmp)
            x_hat = torch.cat([x_hat, x_hat_tmp], dim=0)

        return x_hat

    def forward(self, x, ys=None, yl=None):
        b, t, g, _, _, _ = x.shape
        x = self.forward_features(x, ys)
        x = rearrange(x, '(b t g) d -> b t g d', b=b, t=t, g=g)

        cls_space_tokens = repeat(self.space_token, '() g d -> b t g d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(g + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t g d -> (b t) g d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() t d -> b t d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        # concatenate parameters in different months
        yl = rearrange(yl, 'b y m d -> b (y m d)')
        yl = self.proj_context(yl)
        yl = rearrange(yl, 'b (t d) -> b t d', t=t)
        # yl = repeat(yl, '() d -> b t d', b=b, t=t)

        yl = torch.cat((cls_temporal_tokens, yl), dim=1)
        yl = self.norm1(yl)

        x = self.temporal_transformer(x, yl)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)


if __name__ == "__main__":
    # x.shape = B, T, G, C, H, W
    x = torch.randn((1, 6, 10, 3, 224, 224))
    # ys.shape = B, T, G, N1, d
    ys = torch.randn((1, 6, 10, 28, 9))
    # yl.shape = B, T, N2, d
    yl = torch.randn((1, 5, 12, 9))

    pvt = PVTSimCLR("pvt_tiny", out_dim=512, context_dim=9)
    model = MMST_ViT(out_dim=4, pvt_backbone=pvt, dim=512)

    # print(model)

    z = model(x, ys=ys, yl=yl)
    print(z)
    print(z.shape)
