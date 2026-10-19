# voice-agent

A local-first voice assistant for a custom Linux OS. Speech in, speech out,
with emotion, on hardware we do not control.

The model that answers you is **Qwen3.5-0.8B**, fine-tuned by LoRA into a
"quietly playful" desktop companion. This repo is the part that gives it a
voice: a training pipeline that teaches the model to *express* emotion rather
than a filter that bolts emotion on afterwards.

---

## Why the model emits tags

The first design put an emotion classifier between the LLM and the TTS. It was
discarded for one reason: **a classifier reads the reply after the fact.** It
cannot decide to be amused halfway through a sentence, it cannot place a
laugh, and it cannot make the *content* depend on how the user sounded.

So the LLM itself emits the tags, and it is trained to do so:

```
[emotion:amused]Ah, the classic 'where did my space go' question - love it. [laugh]Run df -h.
```

Two tag families, because they need two different mechanisms downstream:

| tag | kind | when | what it drives |
|---|---|---|---|
| `[emotion:neutral\|amused\|warm\|focused\|serious]` | delivery | once, at the start | which reference clip the voice is cloned from |
| `[laugh] [chuckle] [sigh] [gasp] [hm] [erm]` | non-verbal | at a position | a short pre-rendered clip spliced between two speech segments |

The delivery vocabulary is not arbitrary. It is lifted straight out of the
persona prompt:

> *"be supportive when the user is frustrated **[warm]**, focused during work
> **[focused]**, and playful when the stakes are low **[amused]**"*
> … *"never trivialize data loss, security, or serious problems **[serious]**"*

So the tag is supervision for a rule the prompt already states, and the model
is learning to enforce the persona rather than to imitate a template.

Tags are plain text, not special tokens. The tokenizer is untouched, the
embedding matrix is untouched, `tie_word_embeddings` stays true, and the GGUF
conversion path is unaffected.

---

## The hard part: teaching affect, not a lookup table

If every prompt has exactly one reply, then emotion is a function of *what was
asked*, and a model trained on that learns a lookup table. It will look like it
learned emotion right up until a user phrases the same request differently.

The fix is in the dataset, and it is the reason the annotator has two modes.

**Mode A** rewrites the source reply into the canonical persona's register,
compresses it to a speakable length, and tags it.

**Mode B** rewrites the **user's** turn so it genuinely carries a named
feeling, and writes the reply that feeling calls for:

```
base:     "My build passed and all tests are green!"
frustrated:  "Build's green again, finally. Took the whole damn weekend."
               -> [emotion:warm]Right, that weekend is now yours. [sigh]Go sleep.
delighted:   "MY BUILD IS GREEN!"
               -> [emotion:amused]Ah, look at that. Green across the board. [laugh]
```

Now the same intent arrives in several moods, and the only way to pick the
right one is to read the user's words. That is the whole mechanism.

### The gate that proves it worked

`tools/check_tag_leakage.py` measures whether the labels are recoverable from
the prompt alone, with a cross-validated bag-of-words classifier. It is
deliberately **per label**, because the labels are not equally allowed to be
predictable:

| label | expected | why |
|---|---|---|
| `serious` | high (~75-85%) | "drop the jokes on data loss" is a hard rule. Being able to spot a serious situation *is* the product working. |
| `amused` `warm` `focused` `neutral` | near chance (~25-35%) | these must be read from context. Predictability here is the bug. |

**G1 is the gate, and it is subtler than it first looks.** It does not measure
"prompt → emotion" accuracy. It trains a classifier on the *base* prompts —
where the label is by construction a property of the topic — and then asks it
to predict the *affect variants*, which share the topic but were written with
a different mood.

* If it can → the topic determines the label, and a model trained on this data
  learns a lookup table and ignores how the user actually sounded.
* If it cannot → the label depends on the mood in the wording, and the model
  has to read it.

The plain "prompt → emotion" accuracy is still reported, as `G1_info`. A high
score there is *good*: it means the mood is readable. The two numbers together
are the signal — topic-only near chance, full-prompt high.

Three more checks ride along: response diversity (Jaccard over reply 4-grams),
label balance, within-intent variety, and affect-naming (if the teacher's
rewritten user turn says "I am frustrated" rather than "third attempt today,
this is doing my head in", the model will keyword-match instead of reading).

---

## Pipeline

```
deepseek_companion.jsonl (31)   ┐
make_joyful_data.py (37)         ├─▶ load_dataset.py ─▶ normalize_spoken.py
test_conversations.json (72)     ┘                          │
                                                             ▼
                                              annotate_emotion.py  (teacher: 9B)
                                                Mode A  → tagged reply
                                                Mode B  → affect variants
                                                             │
                                                             ▼
                                              check_tag_leakage.py   ← G1 G2 G3 G4
                                                             │
                                                             ▼
                                              flatten_dataset.py ─▶ train.jsonl
                                                             │
                                                             ▼
                                              train_emotion_lora.py  ← G3
                                                             │
                                                             ▼
                                              eval_emotion.py         ← G4
```

### Sources, and what is wrong with each

| source | n | register | notes |
|---|---|---|---|
| `deepseek_companion.jsonl` | 31 | joyful | 59 words/reply (~20 s of speech), 8 with emoji, 21 with code |
| `make_joyful_data.py` | 37 | joyful | needs re-registering |
| `test_conversations.json` | 72 | quietly playful | already spoken-form; this is the persona |

All three are forced onto **one** system prompt - the `test_conversations.json`
one, byte for byte - so the model sees a consistent (prompt, register) pair
throughout training and the existing adapter keeps its footing.

The canonical persona is extracted from the LoRA's own eval set and written to
`data/persona/system_prompt.txt`.

---

## Running it

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/python -m pip install peft accelerate scikit-learn

# 0. fetch models (resumable; this link drops)
.venv/bin/python tools/fetch_models.py

# 1. data
.venv/bin/python tools/load_dataset.py
.venv/bin/python tools/normalize_spoken.py --mode input \
    --in data/base/train.jsonl --out data/base/train.clean.jsonl

# 2. teacher
.venv/bin/python -m mlx_lm server --model models/teacher/Qwen3.5-9B-4bit \
    --port 8081 --chat-template-args '{"enable_thinking":false}'

# 3. annotate
.venv/bin/python tools/annotate_emotion.py \
    --backend openai --model models/teacher/Qwen3.5-9B-4bit \
    --in data/base/train.clean.jsonl

# 4. gate
.venv/bin/python tools/check_tag_leakage.py --strict

# 5. train
.venv/bin/python tools/train_emotion_lora.py --data data/annotated/train.jsonl

# 6. evaluate
.venv/bin/python tools/eval_emotion.py --adapters \
    baseline=models/llm/agents3_lora,emotion=models/llm/agents3_emotion_adapter/adapter
```

`make help` lists the same steps as targets.

### Train on Kaggle, not on a laptop

**This is the recommended way to train.** Qwen3.5's 18 Gated DeltaNet layers
have no fused kernel outside CUDA — `fla`, `causal_conv1d` and `kernels` are
all CUDA-only — so on an M-series Mac every optimizer step falls back to a
pure-PyTorch reference implementation. Measured on this machine: **100 seconds
per step**. On a Kaggle T4, the same code runs in seconds.

`notebooks/train_emotion_kaggle.ipynb` is a self-contained notebook that does
the whole pipeline on Kaggle's GPU. It is *generated* from the tools in this
repository, so it cannot drift away from the code here:

```bash
python3 tools/build_kaggle_notebook.py    # rebuild the notebook
python3 tools/test_kaggle_notebook.py     # run its cells on a laptop, to check
```

The only thing to attach is a Kaggle dataset containing `agents3_lora(1).zip`.
See `notebooks/README.md`.


---

## Teacher backends

The student is 0.8B. A 0.8B annotating its own training data amplifies its own
biases, so the teacher has to be materially stronger.

| backend | flag | notes |
|---|---|---|
| OpenAI-compatible | `--backend openai` | `mlx_lm server`, Ollama, vLLM - the default |
| Anthropic Messages | `--backend anthropic` | any gateway that speaks `/v1/messages` |
| local HF model | `--backend local` | offline fallback, **weak** - for debugging plumbing only |

`--vote-model` lets the two cheap label-vote calls (24 tokens each) use a
smaller/cheaper model than the generation call.

---

## Model surgery

`agents3_merged` is a `Qwen3_5ForConditionalGeneration`: 320 language-model
tensors and 153 vision-tower tensors. A voice agent never feeds an image, so
`tools/extract_text_model.py` drops the vision tower and produces a plain
`Qwen3_5ForCausalLM`. The extraction is exact - logits are bit-identical to the
merged model (`max|d| = 0.000000`, cosine 1.0).

The LoRA touches 96 modules:

* `mlp.{gate,up,down}_proj` on all 24 layers (72)
* `self_attn.{q,k,v,o}_proj` on the 6 full-attention layers (24)

The 18 DeltaNet layers' `in_proj_*` / `out_proj` are **frozen** - the original
regex `(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)` simply does
not match those names. `tools/train_emotion_lora.py` re-derives the module set
and aborts if it ever differs from the original adapter's 192 tensors.

That check has already earned its keep: it caught the adapter's saved weight
keys still pointing at the old `model.language_model.*` path, which PEFT
reports as "missing adapter keys" and which would have loaded a random-valued
adapter and trained it to nothing.

---

## Layout

```
tools/
  common.py                  shared paths, tag vocabulary, persona
  load_dataset.py            three sources -> one record shape
  normalize_spoken.py        strip emoji/markdown/code, validate replies
  annotate_emotion.py        Mode A + Mode B, four teacher backends, 3-shot
  check_tag_leakage.py       G1-G5, the gate that proves the data teaches
                             affect rather than a lookup table
  flatten_dataset.py         group-aware splits + replay mixture
  extract_text_model.py      multimodal -> text-only
  convert_peft_to_mlx_adapter.py
  train_emotion_lora.py      PEFT continuation, G3 target assertion
  train_emotion_lora_mlx.py  fast path (native DeltaNet kernels)
  eval_emotion.py            teacher agreement + regression vs baseline
  design_voice.py            voice bank: emotion baked into reference clips
  fetch_models.py            resumable download for a flaky link
  show_samples.py            read the teacher's output before trusting it
  test_tags.py               tag vocabulary round-trip tests
  build_kaggle_notebook.py   generates the Kaggle notebook from these files
  test_kaggle_notebook.py    runs that notebook's cells on a laptop
pipeline/
  tts/segmenter.py           tagged reply -> speech + non-verbal segments
  tts/qwen3.py               bank cloning, splicing, interruption guard
bench/
  audition.py                render the model's replies to audio
  audio/prompts.json         the audition prompt set
notebooks/
  train_emotion_kaggle.ipynb generated, self-contained
data/
  persona/system_prompt.txt  the canonical prompt
  base/                      140 unified records, split 112/14/14
  annotated/                 teacher output, reports, splits
```

---

## Known limits

* **A 0.8B has limited affect judgement.** Realistic target is 55-70%
  teacher agreement. It will sound *appropriately* toned more reliably than
  *correctly* toned. A bigger student is the lever, not more data.
* **`[emotion:serious]` has to be near-perfect.** If the model jokes about
  data loss, that is a product failure, not a quality nit. It is gated
  harder than everything else.
* **Roman Urdu will sound wrong.** Qwen3-TTS has no Urdu voice. Kept
  deliberately, to be measured in the audition bench rather than assumed.
* **The four delivery labels are a real constraint.** A fifth and sixth mood
  would need a bigger student.
