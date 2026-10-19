# Training on Kaggle

The emotion training runs on Kaggle's GPU, not on a laptop. The reason is
specific: Qwen3.5's 18 Gated DeltaNet layers have no fused kernel outside
CUDA, so on an M-series Mac every optimizer step falls back to a pure-PyTorch
reference implementation. Measured here: **100 seconds per step**. On a T4
with the same code, it is seconds.

## What to upload

Create a Kaggle dataset (Add Data → New Dataset → Upload) containing one file:

```
agents3_lora(1).zip
```

That is the 26 MB LoRA zip from this repository. It carries three things the
notebook needs:

| inside | what it is |
|---|---|
| `adapter_model.safetensors` | the persona LoRA that gets continued |
| `adapter_config.json` | the target modules, rank and alpha |
| `test_conversations.json` | the 72 held-out conversations, and the canonical system prompt |

Everything else — the DeepSeek set, the hand-written pairs, the whole
pipeline — is embedded in the notebook, so there is nothing else to attach
and nothing to keep in sync.

## Running it

1. Open `train_emotion_kaggle.ipynb` in Kaggle.
2. Add your dataset in the right-hand panel → Add Data.
3. Set the accelerator to **GPU T4 x2** or **P100** (any CUDA GPU works).
4. Run all cells.

The first cell prints the GPU and its VRAM, and the teacher is chosen to fit:

| VRAM | teacher |
|---|---|
| ≥ 22 GB | `Qwen/Qwen3-14B`, 4-bit |
| ≥ 15 GB | `Qwen/Qwen3-8B`, 4-bit |
| ≥ 10 GB | `Qwen/Qwen3-4B-Instruct-2507`, bf16 |
| otherwise | `Qwen/Qwen3-1.7B`, bf16 |

Override it with `TEACHER_MODEL` in the configuration cell if you want to.

### Start small

Set `ANNOTATE_LIMIT = 20` in the configuration cell for the first run. It
annotates twenty examples and stops, so you can read the teacher's output
before committing to the full run. The real annotation is the slow part;
everything after it is minutes.

## What it produces

`/kaggle/working/emotion_adapter/` — the trained adapter, the annotated
dataset, and the four reports (annotation, leakage, splits, evaluation).

Download it from the output panel, or attach it as a new dataset to resume
from it next time.

## The gates are not optional

The notebook stops rather than training on data that will not work:

* **The leakage gate.** A classifier trained on the base prompts is asked to
  predict the affect variants. If it can, the emotion follows the topic, the
  model would learn a lookup table, and the run is aborted. The gate is
  *per label*, because `serious` is supposed to be predictable — "drop the
  jokes when there is data loss" is a hard rule — while `amused`, `warm` and
  `focused` must not be.
* **The malformed-target check.** A reply that is still too long, still has
  markdown in it, or carries a non-verbal in a serious turn is dropped rather
  than trained on.

## Why the notebook is generated

`tools/build_kaggle_notebook.py` builds it from the repository's own tools,
embedding each module verbatim and verifying the round-trip. So the notebook
cannot drift away from the code you have been reading, and the local test
harness (`tools/test_kaggle_notebook.py`) runs the notebook's cells on a
laptop to prove the plumbing works before you spend a GPU hour on it.

```bash
python3 tools/build_kaggle_notebook.py    # rebuild
python3 tools/test_kaggle_notebook.py     # run it locally
```

## After training, back home

```bash
# the adapter is the deliverable; merge it into the base and convert
python3 tools/convert_peft_to_mlx_adapter.py
python3 -m mlx_lm server --model <base> --adapter-path <adapter>
```

For llama.cpp, convert the merged model with `convert_hf_to_gguf.py`. Two
traps: pass `--no-mtp` (Qwen3.5 carries a speculative-decoding block that
otherwise becomes an extra layer), and keep `attn_gate`, `attn_qkv` and
`ssm_out` at Q8_0 — Qwen3.5 is unusually sensitive to quantising those three.
