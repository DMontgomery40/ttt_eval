from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from ttt.text_lm.context import ContextConfig, create_context_net
from ttt.text_lm.session_store import TextSessionStore
from ttt.text_lm.ttt_chat import adapt_context_on_tokens, generate_with_context
from ttt.core.model import DEFAULT_CANARY_TEXT

from .text_lm_service import LoadedTextModel, TextLmService


def _to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def _device_from_env() -> torch.device:
    d = os.environ.get("TEXT_LM_DEVICE", "auto").strip().lower()
    if d == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


@dataclass(frozen=True)
class ChatResponse:
    session_id: str
    model_id: str
    prompt: str
    completion: str
    text: str
    update_events: List[Dict[str, Any]]
    updated_at_unix: int


class TextChatService:
    def __init__(
        self,
        *,
        lm: TextLmService,
        session_store: TextSessionStore,
    ) -> None:
        self.lm = lm
        self.sessions = session_store
        self._device = _device_from_env()

    def list_sessions(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        return self.sessions.list_sessions(limit=limit)

    def _choose_model_id(self, model_id: Optional[str]) -> str:
        if model_id:
            return model_id
        # Prefer the latest usable (has tokenizer+checkpoint) model.
        for rec in self.lm.list_models():
            mid = str((rec or {}).get("model_id") or "").strip()
            if not mid:
                continue
            if str((rec or {}).get("status") or "").strip().lower() == "failed":
                continue
            if not os.path.exists(self.lm.store.paths.checkpoint_pt(mid)):
                continue
            if not os.path.exists(self.lm.store.paths.tokenizer_json(mid)):
                continue
            return mid
        raise FileNotFoundError(
            "No usable text model found in artifacts/text_models. Train one in the Train tab."
        )

    def create_session(
        self,
        *,
        model_id: Optional[str],
        context_cfg: ContextConfig,
    ) -> Dict[str, Any]:
        chosen = self._choose_model_id(model_id)
        loaded = self.lm.load_model(chosen)
        d_model = int(loaded.model.cfg.d_model)

        ctx = create_context_net(d_model=d_model, cfg=context_cfg)
        # Initialize optimizer state by doing a no-op load of a fresh optimizer state dict.
        empty_opt_state: Dict[str, Any] = {}

        sid = self.sessions.new_session_id(prefix="chat")
        return self.sessions.create_session(
            session_id=sid,
            model_id=chosen,
            context_cfg=context_cfg,
            context_state=_to_cpu(ctx.state_dict()),
            optim_state=_to_cpu(empty_opt_state),
        )

    def reset_session(self, session_id: str) -> Dict[str, Any]:
        meta = self.sessions.load_meta(session_id)
        chosen = str(meta.get("model_id") or "").strip()
        if not chosen:
            raise ValueError("Session missing model_id")
        loaded = self.lm.load_model(chosen)
        d_model = int(loaded.model.cfg.d_model)

        cfg_raw = meta.get("context", {})
        try:
            context_cfg = ContextConfig(**cfg_raw) if isinstance(cfg_raw, dict) else ContextConfig()
        except Exception:
            context_cfg = ContextConfig()

        ctx = create_context_net(d_model=d_model, cfg=context_cfg)
        self.sessions.save_state(
            session_id,
            context_state=_to_cpu(ctx.state_dict()),
            optim_state={},
        )
        return self.sessions.load_meta(session_id)

    def chat(
        self,
        *,
        session_id: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> ChatResponse:
        meta = self.sessions.load_meta(session_id)
        model_id = str(meta.get("model_id") or "").strip()
        if not model_id:
            raise ValueError("Session missing model_id")

        cfg_raw = meta.get("context", {})
        try:
            context_cfg = ContextConfig(**cfg_raw) if isinstance(cfg_raw, dict) else ContextConfig()
        except Exception:
            context_cfg = ContextConfig()

        loaded: LoadedTextModel = self.lm.load_model(model_id)
        tok = loaded.tokenizer
        model = loaded.model.to(self._device)
        model.eval()

        st = self.sessions.load_state(session_id)
        ctx_state = st.get("context_state", {})
        opt_state = st.get("optim_state", {})

        ctx = create_context_net(d_model=int(model.cfg.d_model), cfg=context_cfg)
        if isinstance(ctx_state, dict):
            ctx.load_state_dict(ctx_state, strict=False)
        ctx = ctx.to(self._device)

        ids = tok.encode(prompt, add_bos=True, add_eos=False)
        canary_token_ids_list = None
        if bool(getattr(context_cfg, "spfw_enabled", False)):
            texts = list(getattr(context_cfg, "canary_texts", []) or [])
            if not texts:
                texts = [DEFAULT_CANARY_TEXT]
            canary_token_ids_list = [tok.encode(t, add_bos=True, add_eos=True) for t in texts]

        update_events, new_opt_state = adapt_context_on_tokens(
            model=model,
            context=ctx,
            token_ids=ids,
            cfg=context_cfg,
            device=self._device,
            optimizer_state=(opt_state if isinstance(opt_state, dict) else None),
            canary_token_ids_list=canary_token_ids_list,
        )

        out_ids = generate_with_context(
            model=model,
            context=ctx,
            prompt_ids=ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=self._device,
            eos_id=tok.eos_id,
        )

        full_text = tok.decode(out_ids, skip_special=True)
        prompt_text = tok.decode(ids, skip_special=True)
        completion = full_text[len(prompt_text) :] if full_text.startswith(prompt_text) else full_text
        completion = completion.lstrip("\n")

        # Persist updated fast weights + optimizer state (on CPU).
        self.sessions.save_state(
            session_id,
            context_state=_to_cpu(ctx.state_dict()),
            optim_state=_to_cpu(new_opt_state),
        )

        self.sessions.append_chat_event(
            session_id,
            prompt=prompt,
            response_preview=completion,
            update_events=[e.__dict__ for e in update_events],
        )

        return ChatResponse(
            session_id=str(session_id),
            model_id=str(model_id),
            prompt=prompt,
            completion=completion,
            text=full_text,
            update_events=[e.__dict__ for e in update_events],
            updated_at_unix=int(time.time()),
        )
