# v3 Colab runbook

## What to upload

| file | path |
|---|---|
| notebook | `voice-agent/notebooks/train_emotion_colab.ipynb` |
| starting adapter | `voice-agent/v1_emotion_adapter.zip` (23 MB) |

Do **not** upload `agents3_lora(1).zip`, the persona bundle, or the v2
`voice_emotion-...zip`. v3 starts from the Oct-22 emotion adapter.

## Colab steps

1. Open <https://colab.research.google.com>, sign in, and upload the notebook.
2. **Runtime -> Change runtime type -> T4 GPU**. Do this before running.
3. **Run all**. The first cell installs dependencies; the second mounts Drive.
   Authorize Drive when asked. If Colab asks to restart the runtime after the
   pip install, restart and continue from the configuration cell; the install
   persists.
4. When cell 8 prompts, upload `v1_emotion_adapter.zip`. It is also found
   automatically if it is already in `/content`.
5. Wait roughly 90-120 minutes:
   - Stage A (emotion, 3 epochs): ~20-25 min
   - Stage B (tools + replay, 2 epochs): ~65-70 min
   - evals: ~15-20 min
6. At the end, download `emotion_adapter_v3.zip`.

If Colab disconnects, the latest adapter for each stage is on Drive under
`MyDrive/voice_emotion/`. If Stage A finished, you can run Stage B directly
from `agents3_emotion_stage_a/adapter`. If Stage B stopped after an epoch,
`agents3_emotion_adapter/adapter` is still a usable checkpoint.

## What to check

The notebook writes two emotion reports and two tool reports:

| report | what it means |
|---|---|
| `eval_report.json` | 98-probe emotion gate; **the discriminator** |
| `eval_report_clean.json` | 30 fresh prompts; v1 already passes, so a drop means over-serious |
| `tool_eval_report.json` | 32-row unseen-intent/ambiguity gate |
| `tool_eval_legacy_report.json` | 75-row v2 file; 39 exact prompts are in training, regression only |

Emotion gates:

- tag validity >= 0.95
- artifact rate <= 0
- teacher agreement >= 0.55
- guarded serious recall >= 0.98
- serious raw recall >= 0.60
- guarded false-reassure <= 0
- non-verbal in serious guarded <= 0
- length p95 <= 45
- repetition rate <= 0.03
- serious false-alarm rate <= 0.25

Tool gates:

- format valid >= 0.90
- function match >= 0.90
- argument match >= 0.85
- no-call accuracy >= 0.85
- invalid call rate <= 0.02
- speech tag valid >= 0.98

`tool_call_rate` is reported on the emotion set but **not gated**. The final
adapter is also the executor, and prompts such as "kill process 4821" may
legitimately produce a valid call. The tool eval is where calls and no-calls
are judged.

## Local baseline (for comparison)

The v1 adapter on the local text base, using the eval's sampled decoding
(temperature 0.8, top_p 0.95), gives:

**98 probes**

- agreement 0.5306
- serious raw 0.4894
- guarded serious 1.0
- focused 0.0625
- warm 0.8462
- amused 0.7857
- tag validity 1.0, artifacts 0, false alarm 0

**30 clean probes**

- agreement 0.7333
- serious raw 0.9167
- guarded serious 1.0
- focused 0.375
- warm 1.0
- amused 0.333
- tag validity 1.0, artifacts 0, false alarm 0

The v3 run should beat the 98-probe agreement and serious-raw numbers without
losing warm/amused. The clean set should stay at or above the v1 numbers.

## After download

Extract into a **new** directory so the v1 adapter is not overwritten:

```bash
cd /Users/maisumabbas/Developer/Voice/voice-agent
mkdir -p models/llm/v3_upload
unzip -o ~/Downloads/emotion_adapter_v3.zip -d models/llm/v3_upload
```

Convert to MLX:

```bash
.venv/bin/python tools/convert_peft_to_mlx_adapter.py \
    --peft models/llm/v3_upload/agents3_emotion_adapter/adapter \
    --out models/llm/agents3_emotion_mlx_adapter_v3
```

Emotion eval:

```bash
.venv/bin/python tools/eval_emotion_mlx.py \
    --adapter models/llm/agents3_emotion_mlx_adapter_v3 \
    --annotated data/v2/test_annotated.jsonl,data/eval/serious_probes.jsonl \
    --out data/v3/eval_emotion_v3_mlx.json --gate

.venv/bin/python tools/eval_emotion_mlx.py \
    --adapter models/llm/agents3_emotion_mlx_adapter_v3 \
    --annotated data/eval/v3_clean_probes.jsonl \
    --out data/v3/eval_emotion_v3_clean_mlx.json --gate
```

Tool eval (PyTorch/MPS, uses the PEFT adapter):

```bash
.venv/bin/python tools/eval_tools.py \
    --data data/tool/v3/eval.jsonl \
    --base models/llm/agents3_text_base \
    --adapter models/llm/v3_upload/agents3_emotion_adapter/adapter \
    --out data/v3/eval_tools_v3.json --device mps --gate
```

## Data hygiene already checked

- Emotion train: 500 rows, exactly 100 per label, 268 unique users.
- Zero exact eval-prompt overlap between emotion train and the 98 probes or the
  30 clean probes.
- Zero exact prompt overlap between the 32-row unseen tool eval and tool
  train/valid.
- All tool calls in train/valid/eval are schema-valid and policy-allowed.
- The deterministic guard covers every serious prompt in both emotion probe
  sets with zero false alarms.
- The v1 adapter preflight passed locally: 96 LoRA modules, 6,389,760
  trainable parameters, `assert_targets` OK.
