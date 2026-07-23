# CSE455 Capstone

## Stage 1 — Decoder-Only Transformer (from scratch)

### Setup
```bash
pip install -r requirements.txt
```

### Files
- `model.py` — the Transformer architecture (decoder-only, multi-head self-attention, built from scratch — no `nn.Transformer`)
- `train.py` — pretraining script: loads TinyStories, tokenizes with GPT-2 BPE, trains, saves a loss curve and sample generations

### Run
```bash
python train.py
```

This will:
1. Download a subset of TinyStories (falls back to a local `train.txt` if you have no internet access)
2. Train for 3000 steps
3. Save `loss_curve.png`
4. Print validation perplexity and sample text generations
5. Save model weights to `stage1_model.pt`

### Model config
- 6 transformer blocks, 6 attention heads, 384 embedding dim, 256 context length
- ~20-30M parameters
- AdamW optimizer, linear warmup + cosine decay, mixed precision on GPU

### Sanity check
Run `python model.py` directly to test the architecture in isolation (forward + backward pass on dummy data, confirms shapes are correct).

---

## Stage 2 — Base Qwen structured-output baseline

This branch owns **2 of 6** official evaluations: frozen Base Qwen → SNIPS and Base Qwen → ATIS.
No training, LoRA, SFT, or DPO is applied here.

### Model

`Qwen/Qwen2.5-0.5B-Instruct` loaded with Hugging Face Transformers:

```python
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
```

### Shared inference settings

See [`configs/base_qwen.yaml`](configs/base_qwen.yaml):

| Setting | Value |
|---|---|
| `model_name` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `max_input_length` | `768` (raised from 256 after token-length check; ATIS prompts with full label lists reach ~660 tokens) |
| `max_new_tokens` | `128` |
| `do_sample` | `false` |
| `seed` | `455` |
| `batch_size` | `1` |

### Datasets

Source mirror: [czhang99/Capsule-NLU](https://github.com/czhang99/Capsule-NLU) (official Coucke SNIPS splits + standard JointSLU ATIS splits).

| Dataset | Train | Validation | Test | Intents | Slots |
|---|---:|---:|---:|---:|---:|
| SNIPS | 13084 | 700 | 700 | 7 | 39 |
| ATIS | 4478 | 500 | 893 | 22 | 83 |

Canonical JSONL fields: `id`, `dataset`, `request`, `intent`, `slots` (list of `{name, value}`; BIO prefixes stripped).

### Output formats

**JSON** (selected):

```json
{"intent":"flight","slots":[{"name":"fromloc.city_name","value":"boston"},{"name":"toloc.city_name","value":"denver"}]}
```

Empty slots: `{"intent":"flight","slots":[]}`

**Key-value** (compared on validation only):

```text
intent=flight
slots=fromloc.city_name:boston | toloc.city_name:denver
```

Empty slots: `slots=NONE`

Prompts: [`prompts/json.txt`](prompts/json.txt), [`prompts/key_value.txt`](prompts/key_value.txt).

### Validation format comparison

| Dataset | Format | Valid % | Intent accuracy | Slot F1 | Exact match |
|---|---|---:|---:|---:|---:|
| SNIPS | JSON | 84.29 | 39.71 | 5.81 | 0.43 |
| SNIPS | Key-value | 58.71 | 17.86 | 2.44 | 0.00 |
| ATIS | JSON | 80.20 | 33.80 | 0.38 | 0.40 |
| ATIS | Key-value | 14.80 | 0.00 | 0.98 | 0.00 |

**Selected format: `json`.** Higher mean exact-match (0.415 vs 0.000), much higher format-valid rate (82.25% vs 36.76%), and higher intent accuracy on both datasets. Full write-up: [`results/base/format_comparison.md`](results/base/format_comparison.md).

### Official base-model test results (locked JSON)

| Dataset | Valid % | Intent accuracy | Slot F1 | Exact match | N |
|---|---:|---:|---:|---:|---:|
| SNIPS | 84.00 | 34.43 | 5.11 | 0.00 | 700 |
| ATIS | 81.08 | 43.67 | 0.81 | 0.22 | 893 |

Artifacts:
- [`results/base/snips_test_predictions.jsonl`](results/base/snips_test_predictions.jsonl)
- [`results/base/snips_test_metrics.json`](results/base/snips_test_metrics.json)
- [`results/base/atis_test_predictions.jsonl`](results/base/atis_test_predictions.jsonl)
- [`results/base/atis_test_metrics.json`](results/base/atis_test_metrics.json)

Error analysis: [`results/base/error_analysis.md`](results/base/error_analysis.md).

### Reproduce

```bash
git clone https://github.com/K3v1njk/CSE455-capstone.git
cd CSE455-capstone
git checkout stage2-base-model

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Prepare SNIPS + ATIS
python src/prepare_data.py --dataset all --seed 455

# Validation format comparison (do not use test for selection)
python src/run_base.py --dataset snips --split validation --format json
python src/run_base.py --dataset snips --split validation --format key_value
python src/run_base.py --dataset atis --split validation --format json
python src/run_base.py --dataset atis --split validation --format key_value

# Or load the model once:
# python src/run_jobs.py --jobs snips:validation:json snips:validation:key_value atis:validation:json atis:validation:key_value
python src/_select_format.py

# Locked-format test evaluations
python src/run_base.py --dataset snips --split test --format json
python src/run_base.py --dataset atis --split test --format json
```

### Training pairs for SFT / DPO teammates

Checklist-style pairs (`request` / `structured_output`), train+validation only:

- `data/processed/{snips,atis}_{train,validation}_pairs_json.jsonl`
- `data/processed/{snips,atis}_{train,validation}_pairs_key_value.jsonl`

Teammate-friendly aliases (`dataset` / `input` / `target`) with the same targets:

- `data/processed/{snips,atis}_{train,validation}_pairs_json_sft.jsonl`
- `data/processed/{snips,atis}_{train,validation}_pairs_key_value_sft.jsonl`

Because **JSON** is locked, teammates should train on the `*_pairs_json*` files.

Example lines from the selected JSON SFT files:

```json
{"dataset":"snips","input":"listen to westbam alumb allergic on google music","target":"{\"intent\":\"PlayMusic\",\"slots\":[{\"name\":\"artist\",\"value\":\"westbam\"},{\"name\":\"album\",\"value\":\"allergic\"},{\"name\":\"service\",\"value\":\"google music\"}]}"}
{"dataset":"snips","input":"add step to me to the 50 clásicos playlist","target":"{\"intent\":\"AddToPlaylist\",\"slots\":[{\"name\":\"entity_name\",\"value\":\"step to me\"},{\"name\":\"playlist\",\"value\":\"50 clásicos\"}]}"}
{"dataset":"atis","input":"i want to fly from baltimore to dallas round trip","target":"{\"intent\":\"atis_flight\",\"slots\":[{\"name\":\"fromloc.city_name\",\"value\":\"baltimore\"},{\"name\":\"toloc.city_name\",\"value\":\"dallas\"},{\"name\":\"round_trip\",\"value\":\"round trip\"}]}"}
```

### Handoff checklist for teammates

Use unchanged:

- Processed datasets under `data/processed/`
- Selected format: **JSON**
- Selected prompt: `prompts/json.txt`
- Request/structured-output (and optional `input`/`target`) training pairs
- Intent / slot label lists
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Shared settings: `configs/base_qwen.yaml`
- Evaluator: `src/evaluate.py` + `src/formats.py`
- Frozen test JSONL files

Pipeline:

```text
Pretrained Qwen → base evaluation (this branch)
Pretrained Qwen + SFT → teammate 2
SFT checkpoint + DPO → teammate 3
```

Do **not** change the frozen test files, JSON schema, prompt, labels, or evaluator.
