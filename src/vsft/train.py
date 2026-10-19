"""Plain SFT (cross-entropy on completion tokens) with Accelerate.

Works single-device (CUDA / MPS / CPU) and multi-GPU via `accelerate launch` (DDP or FSDP;
for FSDP1 + LoRA set `fsdp_use_orig_params: true`). The objective is unchanged from standard
SFT, matching the paper: only the training data differs (boosted responses vs raw expert
traces).

Loss normalization is token-level over each full accumulation window across all ranks:
every micro-batch loss is `sum(token NLL) / N_window`, where N_window is the all-reduced
count of completion tokens in the window, so short and long completions weigh per token.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from vsft.backends.base import render_chat
from vsft.io import read_jsonl, write_json_atomic
from vsft.prompts import student_messages

log = logging.getLogger(__name__)

IGNORE = -100


def load_sft_pairs(path: str | Path, source: str, include_fallbacks: bool) -> list[tuple]:
    """(question, completion) pairs from a boosted file (or any id/question/expert file)."""
    pairs = []
    for row in read_jsonl(path):
        if source == "expert":
            text = row.get("expert")
        else:
            status = row.get("status") or "accepted"
            if status == "dropped" or (status.startswith("fallback") and not include_fallbacks):
                continue
            text = row.get("response")
        if isinstance(text, str) and text.strip():
            pairs.append((row["question"], text))
    if not pairs:
        raise ValueError(f"no usable training rows in {path} (source={source})")
    return pairs


def completion_text(tok, question: str, completion: str) -> tuple[str, str]:
    """(prompt, completion-with-end-of-turn) exactly as the chat template renders them.

    Rendering the whole conversation and taking the suffix after the generation prompt
    gives the template's own end-of-turn token (e.g. <|im_end|> for Qwen, not the base
    model's <|endoftext|>), so the model learns to stop the way it will be prompted.
    """
    msgs = student_messages(question)
    prompt = render_chat(tok, msgs)
    if getattr(tok, "chat_template", None):
        full = tok.apply_chat_template(
            [*msgs, {"role": "assistant", "content": completion}],
            tokenize=False,
            enable_thinking=False,
        )
        if full.startswith(prompt) and completion in full[len(prompt) :]:
            return prompt, full[len(prompt) :]
    eos = tok.eos_token or ""
    return prompt, completion + eos


def tokenize_pair(tok, question: str, completion: str, max_len: int) -> dict | None:
    prompt, comp = completion_text(tok, question, completion)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    comp_ids = tok(comp, add_special_tokens=False)["input_ids"]
    full_len = len(prompt_ids) + len(comp_ids)
    ids = (prompt_ids + comp_ids)[:max_len]
    labels = ([IGNORE] * len(prompt_ids) + comp_ids)[:max_len]
    if all(x == IGNORE for x in labels[1:]):
        return None  # prompt alone fills the context; nothing to learn
    return {"input_ids": ids, "labels": labels, "truncated": full_len > max_len}


class SFTDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def _mixed_precision(mode: str) -> str:
    if mode != "auto":
        return mode
    if torch.cuda.is_available():
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "no"


def _param_dtype(name: str, lora: bool) -> torch.dtype:
    if name == "auto":
        # LoRA: the frozen base can live in bf16 (adapters stay fp32); full FT keeps fp32
        # master weights so small updates are not rounded away.
        cuda_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        return torch.bfloat16 if (lora and cuda_bf16) else torch.float32
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _windows(it, size: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


def train(cfg, data_path: str | Path, out_dir: str | Path, resume_from: str | None = None):
    from accelerate import Accelerator
    from accelerate.utils import DataLoaderConfiguration, DistributedType, set_seed
    from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

    from vsft.backends.hf import load_tokenizer

    t = cfg.train
    out = Path(out_dir)
    lora = t.lora_r > 0
    mp = _mixed_precision(t.mixed_precision)
    # Accumulation is handled manually (token-level normalization over the window).
    acc = Accelerator(
        mixed_precision=mp,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=t.seed),
    )
    set_seed(t.seed)
    from vsft.hpc import enable_a100_math

    enable_a100_math()
    tok = load_tokenizer(cfg.model.name_or_path, cfg.model.trust_remote_code)

    pairs = load_sft_pairs(data_path, t.source, t.include_fallbacks)
    items = [tokenize_pair(tok, q, c, t.max_seq_len) for q, c in pairs]
    items = [x for x in items if x is not None]
    if not items:
        raise ValueError("every training example was empty after tokenization")
    truncated = sum(x.pop("truncated") for x in items)
    loader = DataLoader(
        SFTDataset(items),
        batch_size=t.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )

    param_dtype = _param_dtype(t.param_dtype, lora)
    if not lora and param_dtype == torch.bfloat16:
        log.warning("full finetuning with bf16 master weights: small updates may round away")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path, dtype=param_dtype, trust_remote_code=cfg.model.trust_remote_code
    )
    if t.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    lora_cfg = None
    if lora:
        from peft import LoraConfig, get_peft_model

        if t.gradient_checkpointing:
            model.enable_input_require_grads()
        lora_cfg = LoraConfig(
            r=t.lora_r,
            lora_alpha=t.lora_alpha,
            lora_dropout=t.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    model, loader = acc.prepare(model, loader)
    from vsft.hpc import notify_started, record_stopped, stop_requested

    started_notified = False
    # Optimizer is created on the prepared (possibly FSDP-flattened) parameters.
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = acc.prepare_optimizer(
        torch.optim.AdamW(
            trainable,
            lr=t.learning_rate,
            weight_decay=t.weight_decay,
            fused=torch.cuda.is_available(),  # fewer temporaries than foreach on GPU
        )
    )
    steps_per_epoch = math.ceil(len(loader) / t.grad_accum_steps)  # len(loader) is per rank
    total_steps = max(1, steps_per_epoch * t.num_epochs)
    # The scheduler is stepped exactly once per optimizer step on every rank (not wrapped by
    # accelerate, which would step it num_processes times per step).
    sched = get_cosine_schedule_with_warmup(opt, int(t.warmup_ratio * total_steps), total_steps)
    acc.register_for_checkpointing(sched)

    geometry = {
        "num_processes": acc.num_processes,
        "per_device_batch_size": t.per_device_batch_size,
        "grad_accum_steps": t.grad_accum_steps,
        "num_examples": len(items),
        "data_sha256": _file_sha256(Path(data_path)),
        "total_steps": total_steps,
        "seed": t.seed,
    }
    manifest = {
        "data": str(data_path),
        "num_truncated": truncated,
        "geometry": geometry,
        "precision": {
            "configured_param_dtype": t.param_dtype,
            "effective_param_dtype": str(param_dtype),
            "configured_mixed_precision": t.mixed_precision,
            "effective_mixed_precision": mp,
            "save_dtype": t.save_dtype,
        },
        "config": cfg.to_dict(),
    }
    if acc.is_main_process:
        out.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out / "train_manifest.json", manifest)

    step, start_epoch, skip_batches = 0, 0, 0
    if resume_from:
        state = json.loads((Path(resume_from) / "trainer_state.json").read_text())
        mismatched = {
            k: (state.get("geometry", {}).get(k), v)
            for k, v in geometry.items()
            if state.get("geometry", {}).get(k) != v
        }
        if mismatched:
            raise ValueError(
                f"cannot resume: run geometry changed (saved, now): {mismatched}. Resume needs "
                "the same data, batch geometry and num_epochs (the LR schedule depends on the "
                "total step count); start a new run to train longer."
            )
        acc.load_state(str(resume_from))
        step, start_epoch, skip_batches = state["step"], state["epoch"], state["batches_in_epoch"]
        log.info("resumed from %s at step %d (epoch %d)", resume_from, step, start_epoch)
        hist = out / "train_history.jsonl"
        if acc.is_main_process and hist.exists():
            # Drop records logged after the checkpoint (they will be logged again).
            kept = [
                line
                for line in hist.read_text().splitlines()
                if line.strip() and json.loads(line).get("step", 0) <= step
            ]
            tmp_hist = hist.with_suffix(".jsonl.tmp")
            tmp_hist.write_text("".join(x + "\n" for x in kept))
            tmp_hist.replace(hist)
        acc.wait_for_everyone()

    history_path = out / "train_history.jsonl"
    save_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[t.save_dtype]

    def save_state(epoch: int, batches_in_epoch: int, tag: str) -> None:
        d = out / f"state-{tag}"
        acc.save_state(str(d))
        if acc.is_main_process:
            write_json_atomic(
                d / "trainer_state.json",
                {
                    "step": step,
                    "epoch": epoch,
                    "batches_in_epoch": batches_in_epoch,
                    "geometry": geometry,
                },
            )
            _rotate_states(out, t.keep_last_states, keep=d)
        acc.wait_for_everyone()

    def _clear(path: Path) -> None:
        if acc.is_main_process and path.exists():
            shutil.rmtree(path)
        acc.wait_for_everyone()

    def save_model(d: Path, merge_lora: bool = False) -> None:
        tmp = d.with_name(d.name + ".tmp")
        _clear(tmp)  # never mix shards with a previous crashed save
        state_dict = acc.get_state_dict(model)  # collective under FSDP: all ranks call it
        if acc.is_main_process:
            if lora and merge_lora:
                # Rebuild on CPU from the gathered full state dict, then merge: never runs
                # collectives on sharded params, so it is safe under FSDP.
                from peft import get_peft_model

                base = AutoModelForCausalLM.from_pretrained(
                    cfg.model.name_or_path,
                    dtype=torch.float32,
                    trust_remote_code=cfg.model.trust_remote_code,
                )
                fresh = get_peft_model(base, lora_cfg)
                fresh.load_state_dict({k: v.cpu() for k, v in state_dict.items()}, strict=True)
                merged = fresh.merge_and_unload().to(save_dtype)
                merged.save_pretrained(str(tmp))
            else:
                acc.unwrap_model(model).save_pretrained(
                    str(tmp),
                    state_dict={
                        k: (v.to(save_dtype) if v.is_floating_point() else v)
                        for k, v in state_dict.items()
                    },
                )
            tok.save_pretrained(str(tmp))
            if d.exists():
                shutil.rmtree(d)
            tmp.rename(d)
        acc.wait_for_everyone()

    model.train()
    t0 = time.time()
    world = acc.num_processes
    use_no_sync = acc.distributed_type == DistributedType.MULTI_GPU
    fp16 = mp == "fp16"
    for epoch in range(start_epoch, t.num_epochs):
        loader.set_epoch(epoch)  # seedable sampler: same permutation on resume
        epoch_loader = acc.skip_first_batches(loader, skip_batches) if skip_batches else loader
        if epoch_loader is not loader and hasattr(epoch_loader, "set_epoch"):
            epoch_loader.set_epoch(epoch)  # the skipping wrapper must use the same permutation
        batches_done = skip_batches
        skip_batches = 0
        window_losses = []
        for window in _windows(epoch_loader, t.grad_accum_steps):
            local_tokens = sum(int((b["labels"][:, 1:] != IGNORE).sum()) for b in window)
            n_tok = acc.reduce(
                torch.tensor(local_tokens, device=acc.device, dtype=torch.long), reduction="sum"
            ).item()
            n_tok = max(1, n_tok)
            loss_sum = torch.zeros((), device=acc.device)
            for i, batch in enumerate(window):
                last = i == len(window) - 1
                # Skip the gradient all-reduce on non-final micro-batches under DDP only:
                # FSDP's no_sync would hold full unsharded grads on every rank.
                ctx = acc.no_sync(model) if (use_no_sync and not last) else _nullcontext()
                with ctx:
                    # sum(token NLL) / N_window; x world cancels DDP/FSDP gradient averaging.
                    loss = model(**batch, num_items_in_batch=n_tok).loss * world
                    acc.backward(loss)
                loss_sum += loss.detach().float()
            batches_done += len(window)
            # Collective checks so every rank raises together (no hang on one-rank errors).
            mean_loss = (acc.reduce(loss_sum, reduction="sum") / world).item()
            gnorm = acc.clip_grad_norm_(model.parameters(), t.max_grad_norm)
            gnorm = float(gnorm) if gnorm is not None else float("nan")
            if not math.isfinite(mean_loss):
                raise FloatingPointError(f"non-finite loss {mean_loss} at step {step}")
            if not fp16 and not math.isfinite(gnorm):
                raise FloatingPointError(f"non-finite grad norm {gnorm} at step {step}")
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if not started_notified:
                # STARTED only after a full optimizer step: model load, first forward/backward
                # (where OOMs show up) and the optimizer step all succeeded.
                started_notified = True
                if acc.is_main_process:
                    notify_started(num_examples=len(items), num_processes=acc.num_processes)
            window_losses.append(mean_loss)
            if step % t.log_every_steps == 0:
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "loss": sum(window_losses) / len(window_losses),
                    "grad_norm": gnorm,
                    "lr": sched.get_last_lr()[0],
                    "tokens": n_tok,
                    "elapsed_s": round(time.time() - t0, 1),
                }
                window_losses = []
                if acc.is_main_process:
                    with open(history_path, "a") as f:  # appended, so resume keeps history
                        f.write(json.dumps(rec) + "\n")
                acc.print(json.dumps(rec))
            if t.save_every_steps and step % t.save_every_steps == 0:
                save_state(epoch, batches_done, f"step{step}")
            # Graceful stop: any rank seeing the stop file makes all ranks checkpoint + exit.
            flag = torch.tensor(float(stop_requested()), device=acc.device)
            if acc.reduce(flag, reduction="sum").item() > 0:
                tag = f"preempt-step{step}"
                save_state(epoch, batches_done, tag)
                if acc.is_main_process:
                    record_stopped(
                        out / "STOPPED.json", step=step, state_dir=str(out / f"state-{tag}")
                    )
                raise Stopped(out / f"state-{tag}")
        save_state(epoch + 1, 0, f"epoch{epoch + 1}")
        save_model(out / f"checkpoint-epoch{epoch + 1}")  # LoRA: adapter only
    save_model(out / "final", merge_lora=True)
    if acc.is_main_process:
        (out / "STOPPED.json").unlink(missing_ok=True)  # a completed run is no longer stopped
        write_json_atomic(
            out / "train_result.json",
            {"final_step": step, "elapsed_s": round(time.time() - t0, 1), **asdict(cfg.train)},
        )
    return out / "final"


class Stopped(Exception):
    """Training stopped on request after saving a resumable state."""

    def __init__(self, state_dir: Path):
        super().__init__(str(state_dir))
        self.state_dir = state_dir


def _rotate_states(out: Path, keep_last: int, keep: Path) -> None:
    """Delete all but the newest `keep_last` state-* dirs (by saved step); 0 keeps all."""
    if keep_last <= 0:
        return
    states = []
    for d in out.glob("state-*"):
        f = d / "trainer_state.json"
        if d.is_dir() and f.is_file():
            states.append((json.loads(f.read_text())["step"], d.stat().st_mtime, d))
    states.sort(reverse=True)
    for _, _, d in states[keep_last:]:
        if d != keep:
            shutil.rmtree(d, ignore_errors=True)


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
