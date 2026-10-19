# VerifiedSamplingSFT

A variant of *Finetuning with Sampling: SFT Learns Better Than You Think* (Karan, Chen, Du,
arXiv:2610.02140). The paper rewrites expert traces so they sit closer to the student's own
distribution while carrying the same information, then runs ordinary SFT on the rewritten
traces. The paper reaches this with a block-wise Metropolis–Hastings random walk. This repo
replaces the random walk with a **score → verify → regenerate** loop.

## Algorithm

Each example has a question $q$, an expert trace $\tau$ and, optionally, a reference answer $a$.
The **student** $p_\theta$ is the model being finetuned. It generates and scores the candidates. A separate, larger **verifier** grades them.

```
prefix ← ""
for round r = 0 … max_rounds-1:
    # 1. Regenerate: K continuations of the verified prefix (proposal prompt sees q, τ, prefix)
    x_k = prefix + c_k,      c_k ~ p_θ( · | proposal(q, τ), prefix),   k = 1..K
    # 2. Score: the student scores every token of each candidate under the plain task prompt
    s_k = (1/|x_k|) Σ_t log p_θ(x_{k,t} | q, x_{k,<t})
    # 3. Select: the largest average log-prob wins; candidates within ε of it are tied,
    #    and ties go to the candidate with the highest token-F1 overlap with τ
    winner = argmax_k s_k   (ε-band tie-break)
    # 4. Verify: a larger verifier LLM (Qwen3.5-9B; the student for laptop runs), given τ and a,
    #    checks the winner split into numbered steps
    #    and returns {"correct": bool, "first_error_step": j}
    if correct (and the extracted final answer matches a):  emit winner; stop
    prefix ← winner[: end of step j-1]        # regenerate from the first wrong step's position
on exhaustion: emit τ (on_failure=expert) | drop | last winner
```

Training is plain SFT (cross-entropy on completion tokens) on the emitted traces, as in the paper.

| | Paper | This repo |
|---|---|---|
| Candidate generation | random resample point $m\sim U\{1..(k+1)B\}$, block-wise | the full continuation from the verifier-chosen position |
| Acceptance | MH ratio (in practice: accept if likelihood is higher) | argmax of mean token log-prob, ε-band tie-break |
| Information preservation | the proposal is conditioned on $\tau$ | the same, **plus** a verifier gate that localizes errors, and an answer check |

Implementation notes:
- Candidates are scored under the *plain* student prompt (question only), which is the paper's $p(x)$. The proposal prompt is only used to sample.
- `boost.score_scope: continuation` averages only the newly generated tokens, so a long shared prefix doesn't dilute ε.
- Restarting at the *end* of step $j-1$, rather than the start of step $j$, means the model generates the separator itself and the token boundary stays in-distribution.
- If the verifier output can't be parsed, the loop backs off exponentially from the end ($2, 4, 8, \dots$ steps).
- Every round is logged (scores, winner, tie set, verdict, error step, restart token) in the output JSONL.

## Layout

```
src/vsft/
  prompts.py      student / proposal / verifier prompts, step segmentation with exact offsets
  backends/       HF (CPU/MPS/CUDA) and vLLM backends: render, generate, token_logprobs
  selection.py    ε-band argmax winner selection
  verifier.py     verifier prompt → (correct, first wrong step) → restart offset
  boost.py        batched score→verify→regenerate loop
  train.py        Accelerate SFT (full or LoRA; DDP/FSDP via accelerate launch; resumable)
  evaluate.py     single-shot accuracy
  data.py         MATH L3–5 (8230/1024 split like the paper), MATH-500, GSM8K
  hpc.py          in-allocation preflight (H100/bf16/memory), check-train, config-get
configs/          smoke.yaml (laptop); h100/*.yaml (Slurm H100); accelerate/fsdp_h100.yaml
slurm/            submit.sh (wrapper), jobs/*.sbatch, status.sh, results.sh, remote.sh (SSH)
```

## Usage

```bash
uv sync                      # macOS / CPU / MPS
uv sync --extra vllm         # Linux + CUDA
```

```bash
uv run vsft prepare math math500 gsm8k --out-dir data/processed
```

```bash
uv run vsft boost --config configs/h100/math_qwen35.yaml --input data/processed/math_l345_train.jsonl --output data/boosted/math_qwen35.jsonl
```

```bash
uv run vsft train --config configs/h100/math_qwen35.yaml --data data/boosted/math_qwen35.jsonl --output-dir runs/sft-sampling
```

```bash
uv run vsft eval --config configs/h100/math_qwen35.yaml --model runs/sft-sampling/final --data data/processed/math_l345_test.jsonl data/processed/math500_test.jsonl data/processed/gsm8k_test.jsonl --output-dir runs/sft-sampling/eval
```

- **Baseline SFT on the raw expert traces:** add `--set train.source=expert`. See [Comparison with plain SFT](#comparison-with-plain-sft) for the full side-by-side.
- **Resume interrupted runs:** `boost` resumes automatically by skipping ids already in `--output`. For `train`, pass `--resume-from runs/<run>/state-epochN`.
- **Override any config value:** `--set section.key=value`.

Memory: Qwen3.5's 248k vocabulary makes the logits and loss the biggest training cost, so keep `per_device_batch_size` at 1–2 and raise `grad_accum_steps` instead. Full fine-tuning of Qwen3.5-4B (fp32 master weights + AdamW, about 67 GB) does not fit one H100; use FSDP on 2× H100-80GB (`configs/h100/math_qwen35_full_fsdp.yaml`).

## Comparison with plain SFT

The paper's central claim is that SFT on boosted traces beats SFT on the raw expert traces. The comparison has three arms. All three use the same config, the same training hyperparameters and the same test sets:

| Arm | Model | Training data |
|---|---|---|
| **Initial model** | the untrained student (`model.name_or_path`) | none |
| **Plain SFT** | student + SFT | the raw expert traces of the boosted rows (`train.source=expert`) |
| **Sampling SFT** | student + SFT | the boosted traces of the same rows (`train.source=response`) |

Both SFT arms read the same `boosted.jsonl` with the same row filter: rows dropped by `on_failure: drop`, or fallbacks excluded by `include_fallbacks: false`, are removed from both. So the two arms train on exactly the same questions, example count and LR schedule; the only difference is the text. `slurm/results.sh` prints each arm's example count and step count, so you can check this.

"Initial model" is the starting checkpoint. `Qwen/Qwen3.5-4B` is already post-trained. To match the paper's base-model setting exactly, point `model.name_or_path` at `Qwen/Qwen3.5-4B-Base` instead.

On Slurm, one command runs all three arms. Plain SFT waits for boosting, because it trains on the same rows; the initial-model eval runs as soon as the data is ready:

```bash
slurm/submit.sh compare --account ACCOUNT --run-dir /scratch/$USER/vsft/cmp --cache-root /scratch/$USER/vsft-cache --min-gpu-mem-gb 79
```

```
setup -> prepare -> boost -> train     -> eval        (Sampling SFT)
                          -> train_sft -> eval_sft    (plain SFT)
                 -> eval_base                         (initial model)
```

After `fetch_online.sh`, drop the network stages:

```bash
slurm/submit.sh compare --account ACCOUNT --resume RUN_DIR --stages boost,train,eval,train_sft,eval_sft,eval_base --cache-root /scratch/$USER/vsft-cache --min-gpu-mem-gb 79
```

When the evals finish, print the table. Each eval writes `RUN_DIR/eval/<base|sft|sampling>-JOBID/eval_summary.json`:

```bash
slurm/results.sh RUN_DIR
```

It looks like this (numbers illustrative):

```
| Method | math_l345_test | math500_test | gsm8k_test | Avg |
| --- | --- | --- | --- | --- |
| Initial model | … | … | … | … |
| Plain SFT (expert traces) | … | … | … | … |
| Sampling SFT | … | … | … | … |
| Sampling − Plain SFT (pts) | +… | +… | +… | +… |
```

Each stage can also be submitted on its own, e.g. `slurm/submit.sh train_sft --account ACCOUNT --resume RUN_DIR`.

Without Slurm, run the same three arms locally:

```bash
uv run vsft train --config CONFIG --data data/boosted/BOOSTED.jsonl --output-dir runs/sft-baseline --set train.source=expert
```

```bash
uv run vsft eval --config CONFIG --model runs/sft-baseline/final --label sft --data TEST_FILES --output-dir runs/eval/sft
```

```bash
uv run vsft eval --config CONFIG --label base --data TEST_FILES --output-dir runs/eval/base
```

```bash
uv run vsft compare runs/eval
```

Use `--label sampling` for the Sampling SFT model's eval.

## Slurm / SSH on H100

Every stage is a checked-in batch script under `slurm/jobs/`:

| Stage | Runs on | Needs network |
|---|---|---|
| setup | CPU | yes |
| prepare | CPU | yes |
| boost | H100 | no |
| train | H100 | no |
| eval | H100 | no |
| gpucheck | H100 | no |

You never call `sbatch` directly: `slurm/submit.sh` is a noninteractive wrapper (`--help`). The scripts are portable to bash 3.2 or newer.

**Before submitting**, the wrapper:

- Validates everything cheaply and requires `--account`, which it forwards to every stage.
- Reconciles `--depend-on` jobs.
- Atomically claims a fresh run directory (`runs/<name>`, then `-001`, `-002`, …). It never overwrites an existing one.
- Snapshots the code (`src/`, `slurm/`, `uv.lock` and any accelerate config) into `RUN_DIR/code`. Queued jobs run this snapshot, so later edits or `remote.sh` syncs never change a run.
- Uses one venv per lockfile, `CACHE_ROOT/venvs/<uv.lock sha>`. Runs on different locks never rebuild each other's environment. If the run needs setup first, the wrapper says so on the login node, before the job waits in the queue.
- Checks free disk space at the run directory and the cache root.
- Refuses to submit a second copy of a stage that is already queued or running for the same run.

**At submission**, it:

- Uses `sbatch --parsable`, with the stdout/stderr parsing and job-ID validation your HPC rules require.
- Reconciles any ambiguous response through a unique `--comment` token instead of resubmitting.
- Writes one durable receipt per submission under `receipts/`, recording the exact command, the token, every `VSFT_*` variable, logs and the latest state.
- Names logs like `logs/train-20261019T081500Z-123456.out`.
- Reconciles every predecessor right before chaining it with `afterok`.

**Status markers** live under `status/` and never move backwards for a job:

1. `SUBMITTED` — the wrapper has submitted the job.
2. `PREFLIGHT_OK` — the in-allocation preflight passed: H100 with enough memory, CUDA and bf16 verified by a matmul, inputs present, the model available in the offline HF cache, and the venv built from the snapshot's `uv.lock`.
3. `STARTED` — written by the application itself once the vLLM engine or model has loaded.
4. `SUCCEEDED` or `FAILED`.

**Monitoring** is bounded. The wrapper reports one evidence-based outcome:

| Outcome | Exit code |
|---|---|
| `STARTED` | 0 |
| `PENDING` (with the scheduler's reason) | 10 |
| `FAILED` (with a log tail, including jobs that started then died) | 1 |
| `UNKNOWN/UNVERIFIED` | 3 |
| `SUBMISSION_REJECTED` | 4 |

**Preemption and time limits.** `#SBATCH --signal=B:USR1@600` gives advance warning. A direct `SIGTERM` from `scancel`, preemption or the time limit is handled the same way. Either one creates a stop file, which the workload polls:
- `train` saves a resumable `state-preempt-stepN` and exits 75. The log prints the `--train-resume` command to continue.
- `boost` stops between verification rounds and discards the unfinished batch. Resubmitting it resumes by example id.
- Anything that doesn't stop within `--preempt-grace` gets TERM.
- Non-finite loss or gradient norms fail the job.
- `eval` evaluates only a model whose `SUCCEEDED.json` was written by a successful train job, and only after `check-train` passes.
- `--preempt-grace` defaults to 500 s for boost/train and 240 s for eval. It is always capped at each stage's warning lead minus 60 s.
- Only the USR1 warning path reliably checkpoints large training states. A TERM-only stop (`scancel`, preemption) saves state only if the save finishes within the site's `KillWait`/`GraceTime`, usually about 30 s. Eval and prepare stop promptly on TERM.

**Storage.** Point `--cache-root` (HF, uv, vLLM and Triton caches) and `--run-dir` at scratch storage, not `/home` or `/project`. Training states rotate (`train.keep_last_states`, default 2). Full fine-tuning keeps a large optimizer state per checkpoint, so with `math_qwen35_full_fsdp.yaml` pass `--min-free-gb 200`. Copy the final models you want to keep to project space.

### Typical Alliance-style cluster (compute nodes have no internet)

1. Where your site allows downloads (a data-transfer node, or a login node if permitted), run:

   ```bash
   slurm/fetch_online.sh --i-confirm-site-policy-allows-this --run-dir /scratch/$USER/vsft/math-qwen35 --cache-root /scratch/$USER/vsft-cache
   ```

2. Then submit the GPU stages. They run offline by default; pass `--online` to allow Hub access.

   ```bash
   slurm/submit.sh pipeline --account ACCOUNT --resume /scratch/$USER/vsft/math-qwen35 --stages boost,train,eval --cache-root /scratch/$USER/vsft-cache --min-gpu-mem-gb 79
   ```

If compute nodes do have internet, a single command runs everything, including the setup and prepare CPU jobs:

```bash
slurm/submit.sh pipeline --account ACCOUNT --run-dir /scratch/$USER/vsft/math-qwen35 --cache-root /scratch/$USER/vsft-cache --min-gpu-mem-gb 79
```

### Other commands

Check the allocation and environment only:

```bash
slurm/submit.sh gpucheck --account ACCOUNT --resume RUN_DIR
```

Resume training from a saved state:

```bash
slurm/submit.sh train --account ACCOUNT --resume RUN_DIR --train-resume RUN_DIR/sft/state-preempt-step400
```

Evaluate the run's trained model:

```bash
slurm/submit.sh eval --account ACCOUNT --resume RUN_DIR
```

Inspect a run (status, newest logs, results):

```bash
slurm/status.sh RUN_DIR
```

```bash
slurm/logs.sh RUN_DIR train
```

```bash
slurm/results.sh RUN_DIR
```

### From your Mac over SSH

`remote.sh` uses your `~/.ssh/config` alias. It rsyncs only into a directory it created itself (marked by `.vsft-checkout`), then runs `submit.sh` on the login node. Write remote paths literally or with a leading `~/`, which expands to the remote home:

```bash
slurm/remote.sh submit --host mycluster --remote-dir '~/vsft' -- pipeline --account ACCOUNT --resume /scratch/USER/vsft/math-qwen35 --stages boost,train,eval --cache-root /scratch/USER/vsft-cache --min-gpu-mem-gb 79
```

```bash
slurm/remote.sh status --host mycluster --remote-dir '~/vsft' /scratch/USER/vsft/math-qwen35
```

### CUDA H100 specifics

- **Preflight**, inside the allocation and before anything runs, requires:
  - an H100 (the name matches `--gpu-type H100`), compute capability 8.0 (`--gpu-capability 8.0`);
  - a CUDA build of torch that ships `sm_90` kernels;
  - at least 39 GB of GPU memory, or `--min-gpu-mem-gb 79` for 80 GB cards;
  - working bf16 and TF32 matmuls on every allocated GPU.
- **Recorded versions.** The NVIDIA driver, torch and CUDA versions are written to `status/<stage>.PREFLIGHT_OK`.
- **Precision.** Models run in bf16 (`model.dtype: bfloat16`, `mixed_precision: bf16`). TF32 is enabled for the remaining fp32 matmuls (Ampere and newer). Attention uses PyTorch SDPA, which picks the flash/memory-efficient kernels on H100.
- **CUDA driver.** The default PyPI torch wheels are built for CUDA 12. The node's NVIDIA driver must support that; preflight fails clearly if it doesn't.

### Resources and configs

- `--mem` defaults to `120G` of host RAM. This fits a per-H100 share such as Narval's (about 124 GB and 12 cores per GPU); check your site's share.
- `--gpus h100:N`, `--partition`, `--qos` and `--constraint` are all configurable. No site values are hard-coded.
- Pick the config for your card:
  - `configs/h100/math_qwen35.yaml`: LoRA SFT with the 4B student and 9B verifier. **Needs an 80 GB H100**, because boosting serves both models from one GPU. Submit with `--min-gpu-mem-gb 79`. 
  - `configs/h100/math_qwen35_full_fsdp.yaml`: full SFT. Needs FSDP on 2× H100-80GB: `--gpus h100:2 --accelerate-config configs/accelerate/fsdp_h100.yaml --min-gpu-mem-gb 79`.
- The FSDP config wraps `Qwen3_5DecoderLayer`. For another model family, change `fsdp_transformer_layer_cls_to_wrap`.
- `--gpus` sets the GPU request for every GPU stage. Boost and eval always use one GPU. For multi-GPU training inside `pipeline`/`compare`, use `--train-gpus h100:2`, which applies only to `train`/`train_sft`.
- `submit.sh` refuses a config with a separate verifier unless `--min-gpu-mem-gb 79` is given, so a node with less than 80 GB never wastes the queue wait.

**Before the first full run**, do two short checks on the real cluster:

1. `gpucheck` verifies the H100, CUDA and bf16, and runs one `flash-linear-attention` kernel, the Qwen3.5 training path:

   ```bash
   slurm/submit.sh gpucheck --account ACCOUNT --resume RUN_DIR --min-gpu-mem-gb 79
   ```

2. A 2-example boost builds both vLLM engines and logs GPU memory after loading and after the first batch. Delete `RUN_DIR/boosted.jsonl` afterwards, since boost resumes from it:

   ```bash
   slurm/submit.sh boost --account ACCOUNT --resume RUN_DIR --min-gpu-mem-gb 79 -- --limit 2
   ```

## Tests

```bash
uv run pytest -q
```

```bash
VSFT_RUN_SLOW=1 uv run pytest -q -m slow
```

The slow tests download and run Qwen3.5-0.8B.
