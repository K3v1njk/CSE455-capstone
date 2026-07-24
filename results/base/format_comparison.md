# Validation format comparison (frozen Qwen2.5-0.5B-Instruct)

Settings: `do_sample=false`, `max_new_tokens=128`, `max_input_length=768`, `seed=455`.

| Dataset | Format | Valid % | Intent accuracy | Slot F1 | Exact match |
|---|---|---:|---:|---:|---:|
| SNIPS | JSON | 84.29 | 39.71 | 5.81 | 0.43 |
| SNIPS | Key-value | 58.71 | 17.86 | 2.44 | 0.00 |
| ATIS | JSON | 80.20 | 33.80 | 0.38 | 0.40 |
| ATIS | Key-value | 14.80 | 0.00 | 0.98 | 0.00 |

## Selection rule

Prefer higher **mean exact-match** across SNIPS + ATIS validation; break ties with format-valid %, then slot F1.

| Format | Mean exact match | Mean valid % | Mean slot F1 |
|---|---:|---:|---:|
| JSON | **0.415** | **82.25** | 3.10 |
| Key-value | 0.000 | 36.76 | 1.71 |

## Selected format: `json`

**Justification:** JSON wins on every primary criterion. On both datasets it produces far more schema-valid outputs and higher intent accuracy. Key-value is especially fragile on ATIS (14.8% valid, 0% intent accuracy), so invalid parses dominate the score. Exact match is near zero for the untouched 0.5B base model in both formats; the decisive gap is format validity and intent accuracy, which favor JSON.

Locked prompt for all official evaluations and teammate handoff: [`prompts/json.txt`](../../prompts/json.txt).

SFT pair files to use: `data/processed/*_pairs_json.jsonl`.
