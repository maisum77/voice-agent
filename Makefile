PY      := .venv/bin/python
PORT    ?= 8081
TEACHER := models/teacher/Qwen3.5-9B-4bit
MLXBASE := models/llm/qwen35_08b_mlx_bf16

.DEFAULT_GOAL := help
.PHONY: help fetch data clean persona teacher annotate gate flatten train eval all smoke test clean-models

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

test: ## run the tag vocabulary round-trip tests
	$(PY) tools/test_tags.py

fetch: ## download the teacher and the MLX base (resumable)
	$(PY) tools/fetch_models.py

data: ## unify the three dataset sources and split them
	$(PY) tools/load_dataset.py
	$(PY) tools/normalize_spoken.py --mode input \
	  --in data/base/train.jsonl --out data/base/train.clean.jsonl
	$(PY) tools/normalize_spoken.py --mode input \
	  --in data/base/valid.jsonl --out data/base/valid.clean.jsonl
	$(PY) tools/normalize_spoken.py --mode input \
	  --in data/base/test.jsonl --out data/base/test.clean.jsonl

teacher: ## serve the teacher model (leave running in another terminal)
	$(PY) -m mlx_lm server --model $(TEACHER) --port $(PORT) \
	  --chat-template-args '{"enable_thinking":false}' --temp 0.7 \
	  --prompt-cache-size 0

sample: ## annotate 20 examples so they can be reviewed before spending the rest
	$(PY) tools/annotate_emotion.py \
	  --backend openai --model $(TEACHER) \
	  --in data/base/train.clean.jsonl \
	  --out data/annotated/sample.jsonl \
	  --report data/annotated/sample_report.json \
	  --limit 20 --affects frustrated,delighted
	$(PY) tools/show_samples.py --in data/annotated/sample.jsonl --n 20

review: ## show the annotated samples for review
	$(PY) tools/show_samples.py --in data/annotated/annotated.jsonl --n 30

annotate: ## run the teacher over the dataset
	$(PY) tools/annotate_emotion.py \
	  --backend openai --model $(TEACHER) \
	  --in data/base/train.clean.jsonl \
	  --affects frustrated,delighted

gate: ## G1-G4: is the data going to teach emotion, or a lookup table?
	$(PY) tools/check_tag_leakage.py --strict

flatten: ## annotated -> train/valid/test with a replay mixture
	$(PY) tools/flatten_dataset.py

train: ## continue training the LoRA on the emotion data
	$(PY) tools/train_emotion_lora.py \
	  --data data/annotated/train.jsonl \
	  --valid-data data/annotated/valid.jsonl \
	  --out models/llm/agents3_emotion_adapter

eval: ## teacher agreement + regression against the baseline
	$(PY) tools/eval_emotion.py --adapters \
	  baseline=models/llm/agents3_lora,emotion=models/llm/agents3_emotion_adapter/adapter

extract: ## multimodal merged checkpoint -> text-only
	$(PY) tools/extract_text_model.py

mlx-adapter: ## PEFT adapter -> MLX adapter format
	$(PY) tools/convert_peft_to_mlx_adapter.py

train-mlx: ## continue training under MLX (native DeltaNet kernels, much faster)
	$(PY) tools/train_emotion_lora_mlx.py --data data/annotated/train.jsonl \
	  --out models/llm/agents3_emotion_adapter_mlx

voice-bank: ## design the voice and render the emotion reference clips
	$(PY) tools/design_voice.py --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

audition: ## render the model's replies to audio
	$(PY) bench/audition.py --adapter models/llm/agents3_emotion_adapter/adapter

smoke: ## end-to-end plumbing check with the local (weak) teacher
	$(PY) tools/load_dataset.py
	$(PY) tools/annotate_emotion.py --backend local \
	  --model models/llm/agents3_text_base \
	  --in data/base/train.jsonl --limit 4 --workers 1

all: data annotate gate flatten train eval ## the whole thing

clean-models: ## remove every downloaded model
	rm -rf models/teacher models/llm/qwen35_08b_mlx_bf16 \
	       models/llm/agents3_text_base models/llm/agents3_text_base_mlx
