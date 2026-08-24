import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneGate(nn.Module):
    def __init__(self, dim=512, hidden_dim=128):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.gate_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.gate_mlp[-2].weight)
        nn.init.zeros_(self.gate_mlp[-2].bias)

    def forward(self, ho_tokens, cls_token, override_gate=None, return_gate=False):
        gate_pred = self.gate_mlp(cls_token)
        gate = override_gate if override_gate is not None else gate_pred
        if gate.dim() == 2:
            gate = gate.unsqueeze(-1)
        fused = ho_tokens + gate * cls_token.unsqueeze(1)
        if return_gate:
            return fused, {"gate": gate_pred}
        return fused


class SceneGatePair(nn.Module):
    def __init__(self, dim=512, hidden_dim=256):
        super().__init__()
        self.pair_gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.pair_gate_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.pair_gate_mlp[-2].weight)
        nn.init.zeros_(self.pair_gate_mlp[-2].bias)

    def forward(self, ho_tokens, cls_token, override_gate=None, return_gate=False):
        _, n_pairs, _ = ho_tokens.shape
        cls_expanded = cls_token.unsqueeze(1).expand(-1, n_pairs, -1)
        gate_pred = self.pair_gate_mlp(torch.cat([ho_tokens, cls_expanded], dim=-1))
        gate = override_gate if override_gate is not None else gate_pred
        if gate.dim() == 1:
            gate = gate.view(1, -1, 1)
        elif gate.dim() == 2:
            gate = gate.unsqueeze(-1)
        fused = ho_tokens + gate * cls_expanded
        if return_gate:
            return fused, {"gate": gate_pred}
        return fused


def build_scene_gate(gate_type="image", dim=512, hidden_dim=128):
    if gate_type == "image":
        return SceneGate(dim=dim, hidden_dim=hidden_dim)
    if gate_type == "pair":
        return SceneGatePair(dim=dim, hidden_dim=hidden_dim * 2)
    raise ValueError(f"Unknown scene gate type: {gate_type}")


class SceneGateV2(nn.Module):
    def __init__(self, dim=512, rank=16, dropout=0.0):
        super().__init__()
        self.down_proj = nn.Linear(dim, rank)
        self.up_proj = nn.Linear(rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, ho_tokens, cls_token, override_gate=None, return_gate=False):
        hidden = self.dropout(F.relu(self.down_proj(cls_token)))
        gate_pred = torch.tanh(self.up_proj(hidden))
        gate = override_gate if override_gate is not None else gate_pred
        if gate.dim() == 2:
            gate = gate.unsqueeze(-1)
        fused = ho_tokens + gate * cls_token.unsqueeze(1)
        if return_gate:
            return fused, {"gate": gate_pred}
        return fused


class SceneGatePairV2(nn.Module):
    def __init__(self, dim=512, rank=16, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down_proj = nn.Linear(dim * 2, rank)
        self.up_proj = nn.Linear(rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, ho_tokens, cls_token, override_gate=None, return_gate=False):
        _, n_pairs, _ = ho_tokens.shape
        cls_expanded = cls_token.unsqueeze(1).expand(-1, n_pairs, -1)
        gate_input = torch.cat([self.norm(ho_tokens), cls_expanded], dim=-1)
        gate_pred = torch.tanh(self.up_proj(self.dropout(F.relu(self.down_proj(gate_input)))))
        gate = override_gate if override_gate is not None else gate_pred
        if gate.dim() == 1:
            gate = gate.view(1, -1, 1)
        elif gate.dim() == 2:
            gate = gate.unsqueeze(-1)
        fused = ho_tokens + gate * cls_expanded
        if return_gate:
            return fused, {"gate": gate_pred}
        return fused


def build_scene_gate_v2(gate_type="image", dim=512, rank=16, dropout=0.1):
    if gate_type == "image":
        return SceneGateV2(dim=dim, rank=rank, dropout=dropout)
    if gate_type == "pair":
        return SceneGatePairV2(dim=dim, rank=rank, dropout=dropout)
    raise ValueError(f"Unknown scene gate type: {gate_type}")


class SceneGatePairV3(nn.Module):
    def __init__(self, dim=512, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.norm_ho = nn.LayerNorm(dim)
        self.norm_cls = nn.LayerNorm(dim)
        self.gate_body = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.gate_out = nn.Linear(hidden_dim, 1)
        self.residual_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_limit = None
        self._init_weights()

    def _init_weights(self):
        for module in list(self.gate_body.modules()) + [self.gate_out]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)
        for module in self.residual_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    def forward(self, ho_tokens, cls_token, override_gate=None, return_gate=False):
        _, n_pairs, _ = ho_tokens.shape
        cls_expanded = cls_token.unsqueeze(1).expand(-1, n_pairs, -1)
        gate_input = torch.cat([self.norm_ho(ho_tokens), self.norm_cls(cls_expanded)], dim=-1)
        gate_base_raw = self.gate_out(self.gate_body(gate_input))
        gate_base = torch.tanh(gate_base_raw)
        gate_delta_raw = self.residual_mlp(gate_input)
        if self.residual_limit is None:
            gate_delta = gate_delta_raw
        else:
            gate_delta = self.residual_limit * torch.tanh(gate_delta_raw / self.residual_limit)
        gate_pred = torch.tanh(gate_base + gate_delta)
        gate = override_gate if override_gate is not None else gate_pred
        if gate.dim() == 1:
            gate = gate.view(1, -1, 1)
        elif gate.dim() == 2:
            gate = gate.unsqueeze(-1)
        fused = ho_tokens + gate * cls_expanded
        if return_gate:
            return fused, {"gate": gate_pred, "gate_raw": gate_base + gate_delta}
        return fused


def build_scene_gate_v3(gate_type="pair", dim=512, hidden_dim=128, dropout=0.1, rank=16):
    if gate_type == "pair":
        return SceneGatePairV3(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
    if gate_type == "image":
        return SceneGateV2(dim=dim, rank=rank, dropout=dropout)
    raise ValueError(f"Unknown scene gate type: {gate_type}")


class SceneGatePairV5(nn.Module):
    def __init__(
        self,
        dim=512,
        hidden_dim=128,
        dropout=0.1,
        alpha=0.1,
        activation="tanh",
        init_std=0.0,
        detach_gate_input=True,
        detach_scene_residual=True,
    ):
        super().__init__()
        self.alpha = alpha
        if activation not in {"tanh", "centered_tanh", "standardized_tanh"}:
            raise ValueError(f"Unknown SceneGate V5 activation: {activation}")
        self.activation = activation
        if init_std < 0:
            raise ValueError(f"SceneGate init_std must be non-negative, got {init_std}")
        self.init_std = float(init_std)
        self.detach_gate_input = detach_gate_input
        self.detach_scene_residual = detach_scene_residual
        self.ablation_mode = "pair"
        self.ablation_seed = 0
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

    def _init_weights(self):
        for module in self.gate_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.init_std > 0:
            nn.init.normal_(self.gate_mlp[-1].weight, std=self.init_std)
        else:
            nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def reset_diagnostics(self):
        self._diag = {"raw": [], "u": [], "u_rel": [], "ratio": []}

    def enable_diagnostics(self, enabled=True):
        self._diag_enabled = enabled
        if enabled:
            self.reset_diagnostics()

    def set_ablation_mode(self, mode="pair", seed=0):
        if mode not in {"pair", "zero", "shuffle", "random"}:
            raise ValueError(f"Unknown SceneGate V5 ablation mode: {mode}")
        self.ablation_mode = mode
        self.ablation_seed = int(seed)

    @staticmethod
    def _summarize(values):
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0}
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

    def forward(self, ho_tokens, cls_token, override_gate=None, return_gate=False):
        _, n_pairs, _ = ho_tokens.shape
        cls_expanded = cls_token.unsqueeze(1).expand(-1, n_pairs, -1)
        gate_ho = ho_tokens.detach() if self.detach_gate_input else ho_tokens
        gate_cls = cls_expanded.detach() if self.detach_gate_input else cls_expanded
        gate_input = torch.cat([self.norm_ho(gate_ho), self.norm_cls(gate_cls)], dim=-1)
        raw = self.gate_mlp(gate_input)
        activation_input = raw
        if self.activation in {"centered_tanh", "standardized_tanh"}:
            activation_input = raw - raw.mean(dim=1, keepdim=True)
        if self.activation == "standardized_tanh":
            scale = activation_input.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
            activation_input = activation_input / scale
        unit = torch.tanh(activation_input)
        relative = unit - unit.mean(dim=1, keepdim=True)
        applied_relative = relative
        if self.ablation_mode == "zero":
            applied_relative = torch.zeros_like(relative)
        elif self.ablation_mode == "shuffle" and n_pairs > 1:
            generator = torch.Generator(device=relative.device)
            generator.manual_seed(self.ablation_seed)
            permutation = torch.randperm(n_pairs, generator=generator, device=relative.device)
            applied_relative = relative[:, permutation, :]
        elif self.ablation_mode == "random" and n_pairs > 1:
            generator = torch.Generator(device=relative.device)
            generator.manual_seed(self.ablation_seed)
            random_relative = torch.randn(
                relative.shape,
                generator=generator,
                device=relative.device,
                dtype=relative.dtype,
            )
            random_relative = random_relative - random_relative.mean(dim=1, keepdim=True)
            random_std = random_relative.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-8)
            target_std = relative.std(dim=1, keepdim=True, unbiased=False)
            applied_relative = random_relative / random_std * target_std
        gate = self.alpha * applied_relative
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
                # Synchronize once during epoch-end summarization, not once per image.
                self._diag["raw"].append(raw.detach().flatten())
                self._diag["u"].append(unit.detach().flatten())
                self._diag["u_rel"].append(relative.detach().flatten())
                self._diag["ratio"].append(ratio.detach().flatten())
        fused = ho_tokens + gate * residual_cls
        if return_gate:
            return fused, {
                "gate": gate,
                "gate_raw": raw,
                "gate_unit": unit,
                "gate_relative": relative,
                "gate_relative_applied": applied_relative,
                "alpha": torch.as_tensor(self.alpha, device=ho_tokens.device, dtype=ho_tokens.dtype),
            }
        return fused


class SceneGatePairV5LowRank(SceneGatePairV5):
    """V5 pair-relative gate with a real rank-controlled bottleneck.

    The activation, pair centering, detach policy, residual scaling, ablation
    modes, and diagnostics are inherited unchanged from SceneGatePairV5. Only
    the gate predictor capacity changes from 2*dim -> hidden_dim -> 1 to
    2*dim -> rank -> 1, so ``scene_gate_rank`` is an effective parameter.
    """

    def __init__(
        self,
        dim=512,
        rank=16,
        dropout=0.1,
        alpha=0.1,
        activation="tanh",
        init_std=0.0,
        detach_gate_input=True,
        detach_scene_residual=True,
    ):
        if rank < 1:
            raise ValueError(f"SceneGate V5-lowrank rank must be positive, got {rank}")
        super().__init__(
            dim=dim,
            hidden_dim=rank,
            dropout=dropout,
            alpha=alpha,
            activation=activation,
            init_std=init_std,
            detach_gate_input=detach_gate_input,
            detach_scene_residual=detach_scene_residual,
        )
        self.rank = int(rank)


def build_scene_gate_v5(
    gate_type="pair", dim=512, hidden_dim=128, dropout=0.1, rank=16,
    alpha=0.1, activation="tanh", init_std=0.0
):
    if gate_type != "pair":
        raise ValueError(f"SceneGate V5 only supports pair gate_type, got: {gate_type}")
    return SceneGatePairV5(
        dim=dim, hidden_dim=hidden_dim, dropout=dropout,
        alpha=alpha, activation=activation, init_std=init_std
    )


def build_scene_gate_v5_lowrank(
    gate_type="pair", dim=512, rank=16, dropout=0.1,
    alpha=0.1, activation="tanh", init_std=0.0
):
    if gate_type != "pair":
        raise ValueError(f"SceneGate V5-lowrank only supports pair gate_type, got: {gate_type}")
    return SceneGatePairV5LowRank(
        dim=dim, rank=rank, dropout=dropout,
        alpha=alpha, activation=activation, init_std=init_std
    )
