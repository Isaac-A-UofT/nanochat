# CSC490 A3 — Step-by-Step Walkthrough

> **Due:** Mar 6th, 10:59pm — GitHub PR + Quercus submission
>
> **Current progress:**
> - Part 1 (Architecture Review): complete in Google Doc
> - `--picochat` flag: implemented in `nanochat_modal.py` (d12, 40 shards)
> - Ablation branches: `exp/swiglu-activation` (SwiGLU MLP) and `exp/alibi-attention` (ALiBi positional biases)

---

## 0. One-Time Setup (do this first)

### 0.1 Modal Account & CLI
```powershell
pip install modal
modal setup                 # opens browser to authenticate
```

### 0.2 Modal Secrets
You need a HuggingFace token (for dataset download) and optionally a WandB API key:
```powershell
# Get HF token from https://huggingface.co/settings/tokens
# Get WandB key from https://wandb.ai/authorize
modal secret create nanochat-secrets WANDB_API_KEY=<your-key> HF_TOKEN=hf_<your-token>
```
If you don't want WandB yet, you can use a dummy value for `WANDB_API_KEY` — the code
disables logging when `WANDB_RUN = "dummy"`.

### 0.3 Weights & Biases Account
1. Sign up at https://wandb.ai
2. Create a project called `nanochat` (or similar)
3. Get your API key from https://wandb.ai/authorize
4. Update `nanochat_modal.py`: change `WANDB_RUN = "dummy"` to a descriptive name
   (e.g. `WANDB_RUN = "picochat-baseline"`) to enable logging

### 0.4 Verify Setup with Quick Test
Before spending credits on real runs, sanity check that everything works:
```powershell
modal run nanochat_modal.py::quick_test
```
This runs a tiny d12 model end-to-end on 4×H100 in ~15 min (~$8). If it passes, your
container image, volume, secrets, and multi-GPU training all work.

---

## 1. Part One: Architecture Review (20 marks) — DONE

**Deliverables:**
- [ ] Architecture diagram of nanochat
- [ ] Table of 3 architecture changes from literature (with papers, motivation, technical details, expected impact)

Already complete in Google Doc. Just make sure:
- Your 3 proposed changes reference real papers — you already have SwiGLU and ALiBi implemented

---

## 2. Part Two: Ablations on Picochat (30 marks)

### 2.1 Train the Picochat Baseline

**Branch:** `master`

```powershell
git checkout master
```

Edit `nanochat_modal.py`:
- Set `WANDB_RUN = "picochat-baseline"` (to enable WandB tracking)

Run:
```powershell
modal run nanochat_modal.py -- --picochat
```

This runs the full pipeline (data → tokenizer → pretrain → eval → SFT → chat eval)
with d12/40 shards. Takes ~20-30 min on 8×H100, costs ~$8-10.

**Record these numbers from the output / WandB:**
- Final training loss
- val_bpb (validation bits per byte)
- CORE metric score (22-benchmark average)
- SFT eval results (GSM8K, HumanEval, MMLU accuracy)
- Total training time and estimated cost

### 2.2 Train Ablation 1: SwiGLU Activation

**Branch:** `exp/swiglu-activation`

```powershell
git checkout exp/swiglu-activation
```

Edit `nanochat_modal.py`:
- Set `WANDB_RUN = "picochat-swiglu"` (important: different name for comparison!)

Run:
```powershell
modal run nanochat_modal.py -- --picochat
```

Record the same metrics as baseline.

### 2.3 Train Ablation 2: ALiBi Attention

**Branch:** `exp/alibi-attention`

```powershell
git checkout exp/alibi-attention
```

Edit `nanochat_modal.py`:
- Set `WANDB_RUN = "picochat-alibi"` (different name again)

Run:
```powershell
modal run nanochat_modal.py -- --picochat
```

Record the same metrics as baseline.

### 2.4 Compare Results

**Deliverables:**
- [ ] WandB dashboard screenshot showing all 3 runs overlaid (loss curves, val_bpb, CORE metric)
- [ ] Results table comparing baseline vs SwiGLU vs ALiBi
- [ ] Commentary on each change's impact
- [ ] Cost discussion (record Modal billing or estimate from GPU-hours × rate)
- [ ] Discussion of how ablations might translate to a larger (d24) run

**Example table format:**

| Model | val_bpb | CORE | GSM8K | HumanEval | MMLU | Train Time | Est. Cost |
|-------|---------|------|-------|-----------|------|------------|-----------|
| Picochat baseline (d12, relu², RoPE) | | | | | | | |
| Picochat + SwiGLU | | | | | | | |
| Picochat + ALiBi | | | | | | | |

---

## 3. Part Three: Extending Context Window (30 marks)

**Important:** Do all Part 3 work on a dedicated branch to avoid modifying `nanochat_modal.py`
on your other branches. The seq_len experiment requires changing the pretrain/SFT stage code,
and you don't want those edits to interfere with Part 2 ablations or Part 4 final training.

```powershell
# Create a fresh branch from master (or from your best ablation branch)
git checkout master
git checkout -b exp/context-window
```

### 3.1 Train Picochat at seq_len=512

On the `exp/context-window` branch:

You need to modify `nanochat_modal.py` to pass `--max-seq-len=512` and `--model-tag`
to the pretrain stage. In `stage_pretrain`, change the `_torchrun` call:

```python
_torchrun(
    "scripts.base_train",
    [
        f"--depth={depth}",
        f"--device-batch-size={device_batch_size}",
        f"--max-seq-len=512",                # shorter context
        f"--model-tag=d{depth}-seq512",       # unique checkpoint dir
        f"--run={wandb_run}",
        "--save-every=1000",
    ],
    nproc=_N_PRETRAIN_GPUS,
)
```

Also update `stage_sft` to load from the right checkpoint:
```python
_torchrun(
    "scripts.chat_sft",
    [
        f"--model-tag=d{depth}-seq512",       # load seq512 pretrained model
        f"--run={wandb_run}",
    ],
    nproc=_N_FINETUNE_GPUS,
)
```

Run with `--picochat`. This is **checkpoint 1**.

### 3.2 Continue Training at seq_len=2048

Now modify `stage_pretrain` again to:
1. Resume from the seq512 checkpoint
2. Use sequence length 2048
3. Save to a different model tag

This requires finding the final step number from the seq512 run. You can check:
```powershell
modal volume ls nanochat-vol nanochat_cache/base_checkpoints/d12-seq512/
```

Then modify the pretrain call:
```python
_torchrun(
    "scripts.base_train",
    [
        f"--depth={depth}",
        f"--device-batch-size={device_batch_size}",
        f"--max-seq-len=2048",                    # extended context
        f"--model-tag=d{depth}-seq512",            # resume from seq512 checkpoint dir
        f"--resume-from-step=<FINAL_STEP>",        # step from seq512 training
        f"--run={wandb_run}",
        "--save-every=1000",
    ],
    nproc=_N_PRETRAIN_GPUS,
)
```

**Important:** The resume overwrites the same checkpoint dir. If you want to keep both,
copy the seq512 checkpoint first or use `--model-tag=d{depth}-seq2048` (but then you need
to manually copy the checkpoint files so resume-from-step can find them).

**Alternative simpler approach:** Run SFT at 2048 on the seq512 base model. SFT already
supports `--max-seq-len=2048` override and handles the context expansion during fine-tuning:
```python
_torchrun(
    "scripts.chat_sft",
    [
        f"--model-tag=d{depth}-seq512",   # load the 512-context base model
        f"--max-seq-len=2048",            # expand to 2048 during SFT
        f"--run={wandb_run}",
    ],
    nproc=_N_FINETUNE_GPUS,
)
```

This is **checkpoint 2**.

### 3.3 Build a Custom Eval

**Deliverables:**
- [ ] A custom evaluation comparing checkpoint 1 (seq512) vs checkpoint 2 (seq2048)
- [ ] Choose a task that benefits from longer context (e.g. long-passage QA, multi-turn conversation, document summarization)
- [ ] Literature justification for your sequence length choices

**Eval ideas that test long context:**
- Feed the model a long passage (~1000+ tokens) and ask a question about details near the start
- Multi-turn conversation where later answers depend on early context
- Compare perplexity on documents of varying length (512 vs 1024 vs 2048 tokens)

You can write a small eval script or add to `tasks/customjson.py` with your test cases.

---

## 4. Part Four: Final Nanochat Training (20 marks)

### 4.1 Choose and Justify Configuration

Edit `nanochat_modal.py`:
```python
DEPTH = 24          # or your chosen depth
NUM_SHARDS = 240    # full dataset for d24
WANDB_RUN = "nanochat-final"
```

### 4.2 Train
```powershell
# On your best branch (with chosen architecture changes)
modal run nanochat_modal.py
```

This runs the full pipeline without `--picochat`.

### 4.3 Scaling Law Analysis

**Deliverables:**
- [ ] Predict performance of nanochat from picochat results using scaling laws
- [ ] Compare prediction to actual nanochat results
- [ ] Reference Chinchilla/Kaplan scaling law papers

Use the parameter counts and loss values:
```
picochat: ~125M params, val_bpb = X
nanochat: ~768M params, val_bpb = ? (predicted from scaling law: L ∝ N^(-α))
```

### 4.4 Emergent Abilities

**Deliverables:**
- [ ] Table comparing picochat vs nanochat results
- [ ] List of 10 questions nanochat can answer but picochat cannot
- [ ] Commentary on what "emerges" with scale

Run both models interactively:
```powershell
# Download checkpoints
modal volume get nanochat-vol nanochat_cache/chatsft_checkpoints/ ./checkpoints/
modal volume get nanochat-vol nanochat_cache/tokenizer.model ./checkpoints/

# Chat with each model locally
$env:NANOCHAT_BASE_DIR = "./checkpoints"
python -m scripts.chat_cli -i sft -t d12    # picochat
python -m scripts.chat_cli -i sft -t d24    # nanochat
```

Or run on Modal to avoid local GPU requirement.

---

## 5. Submission

### 5.1 Prepare the PR
```powershell
git checkout -b a3            # create submission branch
mkdir a3
# Copy your a3.pdf into a3/
git add a3/a3.pdf
git commit -m "A3 submission: pre-training nanochat"
git push origin a3
```

Then open a PR on the course GitHub repo.

### 5.2 Checklist

**Report (a3.pdf) must include:**
- [ ] Team members names + student IDs on first page
- [ ] Part 1: Architecture diagram + table of 3 changes with paper references
- [ ] Part 2: Picochat baseline + 2 ablation results table, WandB screenshots, cost discussion, scaling commentary
- [ ] Part 3: seq512 → seq2048 experiment, custom eval results, literature justification
- [ ] Part 4: Final nanochat results, scaling law prediction vs actual, 10 emergent ability questions

**GitHub must include:**
- [ ] Branch `a3` with folder `a3/` containing `a3.pdf`
- [ ] Code changes for ablations (SwiGLU, ALiBi branches, `nanochat_modal.py` with `--picochat`)
- [ ] Any custom eval scripts

### 5.3 Also submit on Quercus
Upload `a3.pdf` on Quercus by the deadline.

---

## Cost Budget Estimate

| Run | Config | GPUs | ~Time | ~Cost |
|-----|--------|------|-------|-------|
| Quick test (verify setup) | d12, 8 shards | 4×H100 | 15 min | ~$8 |
| Picochat baseline | d12, 40 shards | 8×H100 | 20 min | ~$10 |
| Picochat SwiGLU | d12, 40 shards | 8×H100 | 20 min | ~$10 |
| Picochat ALiBi | d12, 40 shards | 8×H100 | 20 min | ~$10 |
| Picochat seq512 | d12, 40 shards | 8×H100 | 15 min | ~$8 |
| Picochat seq512→2048 | d12, 40 shards | 8×H100 | 20 min | ~$10 |
| Final nanochat (d24) | d24, 240 shards | 8×H100 | 3.5 hr | ~$100 |
| **Total estimate** | | | | **~$156** |

**Tips to save money:**
- Always run `quick_test` first after any code change
- Use `--picochat` for all experiments before the final run
- Check Modal billing at https://modal.com/settings/billing
- Set `WANDB_RUN = "dummy"` when doing test runs (WandB adds minimal overhead but names clutter)
- Failed runs still cost money — review logs carefully before re-running
