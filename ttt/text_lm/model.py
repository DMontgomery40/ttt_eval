from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from ttt.core.backbone import BackboneType, create_backbone


@dataclass(frozen=True)
class TinyLmConfig:
    vocab_size: int = 4096
    d_model: int = 256
    backbone: BackboneType = "gru"


class TinyLm(nn.Module):
    def __init__(self, cfg: TinyLmConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.backbone = create_backbone(cfg.backbone, cfg.d_model)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        h = self.backbone(x)
        h = self.ln(h)
        return self.head(h)

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.cfg)

