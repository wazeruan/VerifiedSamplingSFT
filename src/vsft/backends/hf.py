"""Hugging Face transformers backend (CPU / MPS / CUDA)."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from vsft.backends.base import Backend, Generation

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        return _DTYPES[name]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_tokenizer(name_or_path: str, trust_remote_code: bool = False):
    tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def eos_ids(model, tok) -> set[int]:
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, list) else [ids] if ids is not None else [])
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


class HFBackend(Backend):
    def __init__(self, model_cfg, backend_cfg, model=None, tokenizer=None):
        self.device = resolve_device(model_cfg.device)
        self.dtype = resolve_dtype(model_cfg.dtype, self.device)
        self.tokenizer = tokenizer or load_tokenizer(
            model_cfg.name_or_path, model_cfg.trust_remote_code
        )
        if model is None:
            # device_map loads weights straight onto the GPU (no full CPU copy of a 7B
            # model); on MPS/CPU it is slow or hangs, so load then move.
            on_cuda = self.device.type == "cuda"
            model = AutoModelForCausalLM.from_pretrained(
                model_cfg.name_or_path,
                dtype=self.dtype,
                device_map=str(self.device) if on_cuda else None,
                trust_remote_code=model_cfg.trust_remote_code,
            )
            if not on_cuda:
                model = model.to(self.device)
        self.model = model
        self.model.eval()
        self.max_len = model_cfg.max_model_len
        self.gen_bs = backend_cfg.gen_batch_size
        self.score_bs = backend_cfg.score_batch_size
        self.eos = eos_ids(self.model, self.tokenizer)
        torch.manual_seed(backend_cfg.seed)

    def _gen_config(self, n, temperature, top_p, budget) -> GenerationConfig:
        # A fresh config so the checkpoint's generation_config.json (e.g. Qwen's top_k=20,
        # top_p=0.8, repetition_penalty>1) does not silently change the sampler; this
        # keeps HF and vLLM sampling the same distribution.
        common = dict(
            max_new_tokens=budget,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=sorted(self.eos),
            repetition_penalty=1.0,
            top_k=0,
        )
        if temperature > 0:
            return GenerationConfig(do_sample=True, temperature=temperature, top_p=top_p, **common)
        return GenerationConfig(do_sample=False, **common)

    @torch.no_grad()
    def generate(self, prompts, n, temperature, top_p, max_tokens):
        tok = self.tokenizer
        tok.padding_side = "left"
        out: list[list[Generation] | None] = [None] * len(prompts)
        # Prompts that leave no room to generate get empty, unfinished generations
        # instead of failing the whole batch.
        lengths = [self.count_tokens(p) for p in prompts]
        ok = [i for i, n_tok in enumerate(lengths) if n_tok < self.max_len]
        for i in set(range(len(prompts))) - set(ok):
            out[i] = [Generation("", False) for _ in range(n)]
        for c in range(0, len(ok), self.gen_bs):
            idx = ok[c : c + self.gen_bs]
            chunk = [prompts[i] for i in idx]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            width = enc["input_ids"].shape[1]
            budget = max(1, min(max_tokens, self.max_len - max(lengths[i] for i in idx)))
            seqs = self.model.generate(
                **enc, generation_config=self._gen_config(n, temperature, top_p, budget)
            )
            new = seqs[:, width:].tolist()
            tails = enc["input_ids"][:, -1].tolist()  # last real prompt token (left padded)
            for j, i in enumerate(idx):
                gens = []
                for ids in new[j * n : (j + 1) * n]:
                    cut = next((k for k, t in enumerate(ids) if t in self.eos), None)
                    finished = cut is not None
                    ids = ids[:cut] if finished else ids
                    gens.append(Generation(self._decode_continuation(tails[j], ids), finished))
                out[i] = gens
        return out

    def _decode_continuation(self, tail_id: int, ids: list[int]) -> str:
        """Decode new ids in context so SentencePiece tokenizers keep the leading space."""
        tok = self.tokenizer
        head = tok.decode([tail_id], skip_special_tokens=False)
        full = tok.decode([tail_id, *ids], skip_special_tokens=False)
        text = full[len(head) :] if full.startswith(head) else tok.decode(ids)
        # Strip any special tokens that slipped through (e.g. pad inside the sequence).
        for special in tok.all_special_tokens:
            text = text.replace(special, "")
        return text

    @torch.no_grad()
    def token_logprobs(self, prompts, responses):
        tok = self.tokenizer
        pairs = []
        for p, r in zip(prompts, responses, strict=True):
            p_ids = tok(p, add_special_tokens=False)["input_ids"]
            r_ids = tok(r, add_special_tokens=False)["input_ids"]
            r_ids = r_ids[: max(0, self.max_len - len(p_ids))]
            pairs.append((p_ids, r_ids))
        results: list[list[float]] = [[] for _ in pairs]
        todo = [i for i, (p, r) in enumerate(pairs) if r and p]
        for c in range(0, len(todo), self.score_bs):
            idx = todo[c : c + self.score_bs]
            seqs = [pairs[i][0] + pairs[i][1] for i in idx]
            width = max(len(s) for s in seqs)
            max_r = max(len(pairs[i][1]) for i in idx)
            # Left padding: every response ends at the right edge, so only the last
            # max_r + 1 positions need logits (avoids full-vocab logits for the prompt).
            ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for k, s in enumerate(seqs):
                ids[k, width - len(s) :] = torch.tensor(s)
                mask[k, width - len(s) :] = 1
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            logits = self.model(
                input_ids=ids.to(self.device),
                attention_mask=mask.to(self.device),
                position_ids=pos.to(self.device),
                logits_to_keep=max_r + 1,
            ).logits  # [B, max_r + 1, V]; column j sits at position width - max_r - 1 + j
            for k, i in enumerate(idx):
                r_ids = pairs[i][1]
                # Response token at position q is predicted by the logits at q - 1.
                lo = max_r - len(r_ids)
                lp = torch.log_softmax(logits[k, lo : lo + len(r_ids)].float(), dim=-1)
                tgt = torch.tensor(r_ids, device=lp.device)
                results[i] = lp.gather(1, tgt[:, None]).squeeze(1).tolist()
            del logits
        return results
