"""Original pair-relative SceneGate V5 from Scenegateresult-master.

This module intentionally preserves the collapsing V5 formulation as a
separate experiment path. The stabilized implementation remains in
scene_gate_v5.py and is selected with scene_gate_version="v5".
"""

import torch
import torch.nn as nn


class SceneGatePairV5Legacy(nn.Module):
    def __init__(
        self,
        dim=512,
        hidden_dim=128,
        dropout=0.1,
        alpha=0.1,
        detach_gate_input=True,
        detach_scene_residual=True,
    ):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.detach_gate_input = detach_gate_input
        self.detach_scene_residual = detach_scene_residual
        self.norm_ho = nn.LayerNorm(dim)
        self.norm_cls = nn.LayerNorm(dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )
        self._diag_enabled = False
        self.reset_diagnostics()
        self._init_weights()

    def reset_diagnostics(self):
        self._diag = {
            "raw": [],
            "u": [],
            "u_rel": [],
            "ratio": [],
        }

    def enable_diagnostics(self, enabled=True):
        self._diag_enabled = enabled
        if enabled:
            self.reset_diagnostics()

    @staticmethod
    def _summarize(values):
        if not values:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p90": 0.0,
            }
        x = torch.cat(values).float()
        return {
            "mean": x.mean().item(),
            "std": x.std(unbiased=False).item() if x.numel() > 1 else 0.0,
            "min": x.min().item(),
            "max": x.max().item(),
            "p50": x.quantile(0.5).item(),
            "p90": x.quantile(0.9).item(),
        }

    def diagnostics(self):
        return {key: self._summarize(value) for key, value in self._diag.items()}

    def _init_weights(self):
        for module in self.gate_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        last = self.gate_mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        ho_tokens,
        cls_token,
        override_gate=None,
        return_gate=False,
    ):
        _, n_pairs, _ = ho_tokens.shape
        cls_expanded = cls_token.unsqueeze(1).expand(-1, n_pairs, -1)

        gate_ho = ho_tokens.detach() if self.detach_gate_input else ho_tokens
        gate_cls = cls_expanded.detach() if self.detach_gate_input else cls_expanded
        gate_input = torch.cat(
            [self.norm_ho(gate_ho), self.norm_cls(gate_cls)],
            dim=-1,
        )
        raw = self.gate_mlp(gate_input)
        u = torch.tanh(raw)
        u_rel = u - u.mean(dim=1, keepdim=True)
        gate = self.alpha * u_rel
        if override_gate is not None:
            gate = override_gate.to(device=gate.device, dtype=gate.dtype)
            if gate.dim() == 1:
                gate = gate.view(1, -1, 1)
            elif gate.dim() == 2:
                gate = gate.unsqueeze(-1)

        residual_cls = cls_expanded.detach() if self.detach_scene_residual else cls_expanded
        if self._diag_enabled:
            with torch.no_grad():
                residual = gate * residual_cls
                ratio = residual.norm(dim=-1) / (ho_tokens.norm(dim=-1) + 1e-8)
                self._diag["raw"].append(raw.detach().flatten().cpu())
                self._diag["u"].append(u.detach().flatten().cpu())
                self._diag["u_rel"].append(u_rel.detach().flatten().cpu())
                self._diag["ratio"].append(ratio.detach().flatten().cpu())
        fused = ho_tokens + gate * residual_cls
        if return_gate:
            return fused, {
                "gate": gate,
                "gate_raw": raw,
                "gate_unit": u,
                "gate_relative": u_rel,
                "alpha": torch.as_tensor(
                    self.alpha, device=ho_tokens.device, dtype=ho_tokens.dtype
                ),
            }
        return fused


def build_scene_gate_v5_legacy(
    gate_type="pair",
    dim=512,
    hidden_dim=128,
    dropout=0.1,
    alpha=0.1,
):
    if gate_type == "pair":
        return SceneGatePairV5Legacy(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            alpha=alpha,
        )
    raise ValueError(
        f"Legacy SceneGate V5 only supports pair gate_type, got: {gate_type}"
    )
