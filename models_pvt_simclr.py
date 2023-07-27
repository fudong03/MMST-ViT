import models_pvt
from attention import MultiModalTransformer
from torch import nn


class PVTSimCLR(nn.Module):

    def __init__(self, base_model, out_dim=512, context_dim=9, num_head=8, mm_depth=2, dropout=0., pretrained=True, gated_ff=True):
        super(PVTSimCLR, self).__init__()

        self.backbone = models_pvt.__dict__[base_model](pretrained=pretrained)
        num_ftrs = self.backbone.head.in_features

        self.proj = nn.Linear(num_ftrs, out_dim)

        self.proj_context = nn.Linear(context_dim, out_dim)

        # attention
        dim_head = out_dim // num_head
        self.mm_transformer = MultiModalTransformer(out_dim, mm_depth, num_head, dim_head, context_dim=out_dim, dropout=dropout)

        self.norm1 = nn.LayerNorm(context_dim)

    def forward(self, x, context=None):
        h = self.backbone.forward_features(x)  # shape = B, N, D
        h = h.squeeze()

        # project to targeted dim
        x = self.proj(h)
        context = self.proj_context(self.norm1(context))

        # multi-modal attention
        x = self.mm_transformer(x, context=context)

        # return the classification token
        return x[:, 0]
