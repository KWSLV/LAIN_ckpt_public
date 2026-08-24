import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)

        nn.init.kaiming_normal_(self.A.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.B.weight, mode="fan_in", nonlinearity="relu")
        self.B.weight.data *= 0.01

    def forward(self, x):
        return self.B(self.A(x)) * self.scale


class TextSemanticAdapter(nn.Module):
    def __init__(
        self,
        dim=512,
        bottleneck=64,
        rank=4,
        alpha=16,
        residual_scale=0.05,
        dropout=0.1,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.lowrank = LowRankLinear(bottleneck, bottleneck, rank, alpha)
        self.up = nn.Linear(bottleneck, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_normal_(self.down.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.up.weight, mode="fan_in", nonlinearity="relu")
        self.down.weight.data *= 0.01
        self.up.weight.data *= 0.01

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.act(x)
        x = self.lowrank(x)
        x = self.up(x)
        x = self.dropout(x)
        return residual + x * self.residual_scale


class ObjectConditionedAdapter(nn.Module):
    def __init__(self, dim=512, rank=4):
        super().__init__()
        self.lora_down = nn.Linear(dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, dim, bias=False)
        self.condition_net = nn.Sequential(
            nn.Linear(dim, rank * 2),
            nn.ReLU(),
            nn.Linear(rank * 2, rank),
            nn.Sigmoid(),
        )

        nn.init.kaiming_normal_(self.lora_down.weight, mode="fan_in", nonlinearity="relu")
        self.lora_down.weight.data *= 0.01
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, verb_feat, obj_feat):
        obj_feat = obj_feat.unsqueeze(0).expand(verb_feat.shape[0], -1)
        modulation = self.condition_net(obj_feat)
        latent = self.lora_down(verb_feat) * modulation
        return verb_feat + self.lora_up(latent)


def get_adapter_parameters(clip_model):
    if hasattr(clip_model, "text_encoder") and hasattr(clip_model.text_encoder, "text_adapter"):
        return list(clip_model.text_encoder.text_adapter.parameters())
    return []
