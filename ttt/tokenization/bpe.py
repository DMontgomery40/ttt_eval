from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


_SEGMENT_RE = re.compile(r"\\s+|[^\\s]+", re.UNICODE)


DEFAULT_SPECIAL_TOKENS: Tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<mask>")


def _iter_segments(text: str) -> Iterator[str]:
    for seg in _SEGMENT_RE.findall(text):
        if seg:
            yield seg


def _iter_lines(paths: Sequence[str], *, max_lines: Optional[int] = None) -> Iterator[str]:
    seen = 0
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line.rstrip("\\n")
                seen += 1
                if max_lines is not None and seen >= max_lines:
                    return


def _atomic_write_json(path: str, data: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


@dataclass(frozen=True)
class BpeTrainingConfig:
    vocab_size: int = 4096
    min_pair_freq: int = 2
    max_lines: Optional[int] = None
    special_tokens: Tuple[str, ...] = DEFAULT_SPECIAL_TOKENS


class BpeTokenizer:
    """
    Simple byte-level BPE.

    - Pretokenization preserves whitespace using `\\s+|[^\\s]+`
    - Base vocabulary is raw bytes (0..255), offset by `len(special_tokens)`
    - BPE merges are learned over byte IDs and merge-produced IDs

    This is intentionally dependency-free (no sentencepiece/tokenizers).
    """

    def __init__(
        self,
        *,
        merges: Sequence[Tuple[int, int]],
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> None:
        self.special_tokens: Tuple[str, ...] = tuple(special_tokens)
        self.special_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.special_tokens)}
        self.id_to_special: Dict[int, str] = {i: t for t, i in self.special_to_id.items()}

        self.byte_offset = len(self.special_tokens)
        self.base_byte_vocab_size = 256
        self.merge_id_offset = self.byte_offset + self.base_byte_vocab_size

        self.merges: List[Tuple[int, int]] = [tuple(map(int, m)) for m in merges]
        self.pair_rank: Dict[Tuple[int, int], int] = {pair: i for i, pair in enumerate(self.merges)}
        self.pair_to_id: Dict[Tuple[int, int], int] = {
            pair: self.merge_id_offset + i for i, pair in enumerate(self.merges)
        }

        # Build id->bytes table
        id_to_bytes: List[Optional[bytes]] = [None] * self.byte_offset
        id_to_bytes.extend([bytes([b]) for b in range(256)])

        for i, (a, b) in enumerate(self.merges):
            new_id = self.merge_id_offset + i
            if a >= new_id or b >= new_id:
                raise ValueError(f"Invalid merge refers to future id: {(a, b)} at {new_id}")
            a_bytes = id_to_bytes[a]
            b_bytes = id_to_bytes[b]
            if a_bytes is None or b_bytes is None:
                raise ValueError(f"Merge references special token ids: {(a, b)}")
            id_to_bytes.append(a_bytes + b_bytes)

        self._id_to_bytes = id_to_bytes
        self.vocab_size = len(self._id_to_bytes)

    @property
    def pad_id(self) -> int:
        return self.special_to_id.get("<pad>", 0)

    @property
    def bos_id(self) -> int:
        return self.special_to_id.get("<bos>", 1)

    @property
    def eos_id(self) -> int:
        return self.special_to_id.get("<eos>", 2)

    @property
    def mask_id(self) -> int:
        return self.special_to_id.get("<mask>", 3)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)

        for seg in _iter_segments(text):
            b = seg.encode("utf-8", errors="replace")
            seg_ids = [self.byte_offset + x for x in b]
            ids.extend(self._apply_bpe(seg_ids))

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str:
        out = bytearray()
        for tid in ids:
            if tid in self.id_to_special:
                if skip_special:
                    continue
                token = self.id_to_special[tid]
                out.extend(token.encode("utf-8"))
                continue

            if tid < 0 or tid >= len(self._id_to_bytes):
                continue
            piece = self._id_to_bytes[tid]
            if piece:
                out.extend(piece)
        return out.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        payload = {
            "schema_version": 1,
            "created_at_unix": int(time.time()),
            "special_tokens": list(self.special_tokens),
            "merges": [[int(a), int(b)] for (a, b) in self.merges],
        }
        _atomic_write_json(path, payload)

    @classmethod
    def load(cls, path: str) -> "BpeTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Invalid tokenizer JSON (expected object)")
        merges = payload.get("merges", [])
        if not isinstance(merges, list):
            raise ValueError("Invalid merges (expected list)")
        merges_t: List[Tuple[int, int]] = []
        for m in merges:
            if not isinstance(m, list) or len(m) != 2:
                continue
            merges_t.append((int(m[0]), int(m[1])))
        special = payload.get("special_tokens", list(DEFAULT_SPECIAL_TOKENS))
        if not isinstance(special, list) or not all(isinstance(x, str) for x in special):
            special = list(DEFAULT_SPECIAL_TOKENS)
        return cls(merges=merges_t, special_tokens=special)

    def _apply_bpe(self, ids: List[int]) -> List[int]:
        # Merge best-ranked pair repeatedly until no applicable merges remain.
        if len(ids) < 2 or not self.pair_rank:
            return list(ids)

        out = list(ids)
        while True:
            best_rank: Optional[int] = None
            best_pair: Optional[Tuple[int, int]] = None
            for i in range(len(out) - 1):
                pair = (out[i], out[i + 1])
                rank = self.pair_rank.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            merged_id = self.pair_to_id[best_pair]
            new: List[int] = []
            i = 0
            while i < len(out):
                if i < len(out) - 1 and (out[i], out[i + 1]) == best_pair:
                    new.append(merged_id)
                    i += 2
                else:
                    new.append(out[i])
                    i += 1
            out = new

        return out


def train_bpe(
    lines: Iterable[str],
    *,
    config: BpeTrainingConfig,
    progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_every_merges: int = 50,
) -> BpeTokenizer:
    byte_offset = len(config.special_tokens)
    base_ids = byte_offset + 256

    vocab: Dict[Tuple[int, ...], int] = {}
    t0 = time.time()
    line_count = 0
    for line in lines:
        line_count += 1
        for seg in _iter_segments(line):
            b = seg.encode("utf-8", errors="replace")
            if not b:
                continue
            seq = tuple(byte_offset + x for x in b)
            vocab[seq] = vocab.get(seq, 0) + 1

        if progress_hook and (line_count == 1 or line_count % 2000 == 0):
            progress_hook(
                {
                    "stage": "tokenizer_vocab",
                    "lines": int(line_count),
                    "unique_segments": int(len(vocab)),
                    "seconds": float(time.time() - t0),
                }
            )

    if progress_hook:
        progress_hook(
            {
                "stage": "tokenizer_vocab_built",
                "lines": int(line_count),
                "unique_segments": int(len(vocab)),
                "seconds": float(time.time() - t0),
            }
        )

    merges: List[Tuple[int, int]] = []
    next_id = base_ids
    max_merges = max(0, int(config.vocab_size) - base_ids)

    for merge_i in range(max_merges):
        pair_counts: Counter[Tuple[int, int]] = Counter()
        for seq, freq in vocab.items():
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += freq

        if not pair_counts:
            break

        (a, b), count = pair_counts.most_common(1)[0]
        if int(count) < int(config.min_pair_freq):
            break

        merges.append((int(a), int(b)))
        new_id = next_id
        next_id += 1

        new_vocab: Dict[Tuple[int, ...], int] = {}
        for seq, freq in vocab.items():
            out: List[int] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            out_t = tuple(out)
            new_vocab[out_t] = new_vocab.get(out_t, 0) + freq
        vocab = new_vocab

        if progress_hook and (merge_i == 0 or (merge_i + 1) % max(1, int(progress_every_merges)) == 0):
            progress_hook(
                {
                    "stage": "tokenizer_merge",
                    "merge": int(merge_i + 1),
                    "max_merges": int(max_merges),
                    "best_pair_count": int(count),
                    "vocab_size": int(next_id),
                    "seconds": float(time.time() - t0),
                }
            )

    return BpeTokenizer(merges=merges, special_tokens=config.special_tokens)


def train_bpe_from_files(
    paths: Sequence[str],
    *,
    vocab_size: int = 4096,
    min_pair_freq: int = 2,
    max_lines: Optional[int] = None,
    special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_every_merges: int = 50,
) -> BpeTokenizer:
    cfg = BpeTrainingConfig(
        vocab_size=int(vocab_size),
        min_pair_freq=int(min_pair_freq),
        max_lines=max_lines,
        special_tokens=tuple(special_tokens),
    )
    return train_bpe(
        _iter_lines(paths, max_lines=max_lines),
        config=cfg,
        progress_hook=progress_hook,
        progress_every_merges=progress_every_merges,
    )


def _cmd_train(args: argparse.Namespace) -> None:
    tok = train_bpe_from_files(
        args.input,
        vocab_size=args.vocab_size,
        min_pair_freq=args.min_pair_freq,
        max_lines=args.max_lines,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
    )
    tok.save(args.output)
    print(f"Wrote tokenizer: {args.output}")
    print(f"Vocab size: {tok.vocab_size}")


def _cmd_encode(args: argparse.Namespace) -> None:
    tok = BpeTokenizer.load(args.tokenizer)
    ids = tok.encode(args.text, add_bos=args.add_bos, add_eos=args.add_eos)
    print(" ".join(str(i) for i in ids))


def _cmd_decode(args: argparse.Namespace) -> None:
    tok = BpeTokenizer.load(args.tokenizer)
    ids = [int(x) for x in args.ids]
    print(tok.decode(ids, skip_special=not args.keep_special))


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Dependency-free byte-level BPE")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a BPE tokenizer from text files")
    p_train.add_argument("--input", nargs="+", required=True, help="Text files")
    p_train.add_argument("--output", required=True, help="Output tokenizer JSON")
    p_train.add_argument("--vocab_size", type=int, default=4096)
    p_train.add_argument("--min_pair_freq", type=int, default=2)
    p_train.add_argument("--max_lines", type=int, default=None)
    p_train.set_defaults(fn=_cmd_train)

    p_encode = sub.add_parser("encode", help="Encode text -> token ids")
    p_encode.add_argument("--tokenizer", required=True)
    p_encode.add_argument("--text", required=True)
    p_encode.add_argument("--add_bos", action="store_true")
    p_encode.add_argument("--add_eos", action="store_true")
    p_encode.set_defaults(fn=_cmd_encode)

    p_decode = sub.add_parser("decode", help="Decode token ids -> text")
    p_decode.add_argument("--tokenizer", required=True)
    p_decode.add_argument("ids", nargs="+")
    p_decode.add_argument("--keep_special", action="store_true")
    p_decode.set_defaults(fn=_cmd_decode)

    args = p.parse_args(list(argv) if argv is not None else None)
    args.fn(args)


if __name__ == "__main__":
    main()
