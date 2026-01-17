from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ttt.text_lm.model import TinyLm, TinyLmConfig
from ttt.text_lm.store import TextModelStore, _read_json
from ttt.tokenization.bpe import BpeTokenizer


def _device_from_env() -> torch.device:
    d = os.environ.get("TEXT_LM_DEVICE", "auto").strip().lower()
    if d == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


@dataclass
class LoadedTextModel:
    model_id: str
    tokenizer: BpeTokenizer
    model: TinyLm
    device: torch.device


class TextLmService:
    def __init__(self, *, store: TextModelStore) -> None:
        self.store = store
        self._cache: Dict[str, LoadedTextModel] = {}
        self._device = _device_from_env()

    def list_models(self) -> list:
        return self.store.list_models()

    def _load_model(self, model_id: str) -> LoadedTextModel:
        cached = self._cache.get(model_id)
        if cached is not None:
            return cached

        tok = BpeTokenizer.load(self.store.paths.tokenizer_json(model_id))
        cfg_raw = _read_json(self.store.paths.config_json(model_id))
        if not isinstance(cfg_raw, dict):
            raise ValueError("Invalid config.json for model")

        cfg = TinyLmConfig(
            vocab_size=int(cfg_raw.get("vocab_size", tok.vocab_size)),
            d_model=int(cfg_raw.get("d_model", 256)),
            backbone=str(cfg_raw.get("backbone", "gru")),  # type: ignore[arg-type]
        )

        ckpt = torch.load(self.store.paths.checkpoint_pt(model_id), map_location="cpu")
        model = TinyLm(cfg)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(self._device).eval()

        loaded = LoadedTextModel(model_id=model_id, tokenizer=tok, model=model, device=self._device)
        self._cache[model_id] = loaded
        return loaded

    @torch.no_grad()
    def generate(
        self,
        *,
        prompt: str,
        model_id: Optional[str],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> dict:
        chosen = model_id or self.store.latest_model_id()
        if not chosen:
            raise FileNotFoundError("No trained text model found in artifacts/text_models")

        m = self._load_model(chosen)
        ids = m.tokenizer.encode(prompt, add_bos=True, add_eos=False)

        for _ in range(int(max_new_tokens)):
            x = torch.tensor([ids], dtype=torch.long, device=m.device)
            logits = m.model(x)[0, -1]
            logits = logits / max(1e-6, float(temperature))
            if top_k > 0 and top_k < logits.numel():
                vals, _ = torch.topk(logits, int(top_k))
                cutoff = vals[-1]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            ids.append(next_id)
            if next_id == m.tokenizer.eos_id:
                break

        text = m.tokenizer.decode(ids, skip_special=True)
        return {"model_id": chosen, "prompt": prompt, "text": text}

