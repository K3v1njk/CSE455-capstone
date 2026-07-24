#!/usr/bin/env python3
"""Download and convert SNIPS / ATIS into canonical JSONL + SFT pair files."""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from formats import to_structured_output  # noqa: E402

DATA_BASE = (
    "https://raw.githubusercontent.com/czhang99/Capsule-NLU/master/data"
)
# Official Coucke SNIPS splits + standard ATIS JointSLU splits (Capsule-NLU mirror).
SPLITS = ("train", "valid", "test")
SPLIT_OUT = {"train": "train", "valid": "validation", "test": "test"}


def bio_to_slots(tokens: Sequence[str], tags: Sequence[str]) -> List[Dict[str, str]]:
    """Convert word-level BIO tags into a list of {name, value} slot dicts."""
    if len(tokens) != len(tags):
        raise ValueError(
            f"Token/tag length mismatch: {len(tokens)} tokens vs {len(tags)} tags"
        )
    slots: List[Dict[str, str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        tag = tags[i]
        if tag == "O" or tag == "":
            i += 1
            continue
        if tag.startswith("B-"):
            name = tag[2:]
            values = [tokens[i]]
            i += 1
            while i < n and tags[i].startswith("I-") and tags[i][2:] == name:
                values.append(tokens[i])
                i += 1
            slots.append({"name": name, "value": " ".join(values)})
            continue
        if tag.startswith("I-"):
            # Malformed BIO: treat as beginning of a new span.
            name = tag[2:]
            values = [tokens[i]]
            i += 1
            while i < n and tags[i].startswith("I-") and tags[i][2:] == name:
                values.append(tokens[i])
                i += 1
            slots.append({"name": name, "value": " ".join(values)})
            continue
        i += 1
    return slots


def download_text(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest.read_text(encoding="utf-8")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    dest.write_text(text, encoding="utf-8")
    return text


def load_split(dataset: str, split: str, raw_dir: Path) -> List[dict]:
    base = f"{DATA_BASE}/{dataset}/{split}"
    cache = raw_dir / dataset / split
    seq_in = download_text(f"{base}/seq.in", cache / "seq.in").splitlines()
    seq_out = download_text(f"{base}/seq.out", cache / "seq.out").splitlines()
    labels = download_text(f"{base}/label", cache / "label").splitlines()

    if not (len(seq_in) == len(seq_out) == len(labels)):
        raise RuntimeError(
            f"{dataset}/{split}: length mismatch "
            f"seq.in={len(seq_in)} seq.out={len(seq_out)} label={len(labels)}"
        )

    out_split = SPLIT_OUT[split]
    records = []
    for idx, (utt, bio, intent) in enumerate(zip(seq_in, seq_out, labels)):
        tokens = utt.strip().split()
        tags = bio.strip().split()
        slots = bio_to_slots(tokens, tags)
        records.append(
            {
                "id": f"{dataset}-{out_split}-{idx}",
                "dataset": dataset,
                "request": utt.strip(),
                "intent": intent.strip(),
                "slots": slots,
            }
        )
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def collect_labels(records: Sequence[dict]) -> Tuple[List[str], List[str]]:
    intents: Set[str] = set()
    slots: Set[str] = set()
    for rec in records:
        intents.add(rec["intent"])
        for s in rec["slots"]:
            slots.add(s["name"])
    return sorted(intents), sorted(slots)


def check_leakage(
    left: Sequence[dict], right: Sequence[dict], left_name: str, right_name: str
) -> None:
    left_ids = {r["id"] for r in left}
    right_ids = {r["id"] for r in right}
    id_overlap = left_ids & right_ids
    if id_overlap:
        raise RuntimeError(
            f"{left_name}/{right_name}: ID overlap: {sorted(id_overlap)[:5]}"
        )

    left_req = {r["request"] for r in left}
    right_req = {r["request"] for r in right}
    req_overlap = left_req & right_req
    if req_overlap:
        print(
            f"NOTE {left_name}/{right_name}: {len(req_overlap)} exact request "
            f"string(s) appear in both splits (known for these benchmarks)."
        )


def write_pairs(path: Path, records: Sequence[dict], fmt: str) -> None:
    """Checklist-style pairs: request / structured_output."""
    pairs = [
        {
            "request": r["request"],
            "structured_output": to_structured_output(r["intent"], r["slots"], fmt),
        }
        for r in records
    ]
    write_jsonl(path, pairs)


def write_sft_pairs(path: Path, records: Sequence[dict], fmt: str) -> None:
    """Teammate-friendly SFT pairs: dataset / input / target."""
    pairs = [
        {
            "dataset": r["dataset"],
            "input": r["request"],
            "target": to_structured_output(r["intent"], r["slots"], fmt),
        }
        for r in records
    ]
    write_jsonl(path, pairs)


def spot_check(records: Sequence[dict], n: int, seed: int) -> None:
    rng = random.Random(seed)
    sample = records if len(records) <= n else rng.sample(list(records), n)
    print(f"\n--- Spot-check ({len(sample)} examples) ---")
    for r in sample:
        print(json.dumps(r, ensure_ascii=False))


def prepare_dataset(dataset: str, seed: int, processed_dir: Path, raw_dir: Path) -> None:
    by_split: Dict[str, List[dict]] = {}
    for split in SPLITS:
        by_split[SPLIT_OUT[split]] = load_split(dataset, split, raw_dir)

    train = by_split["train"]
    validation = by_split["validation"]
    test = by_split["test"]

    check_leakage(train, test, "train", "test")
    check_leakage(train, validation, "train", "validation")

    all_records = train + validation + test
    intent_labels, slot_labels = collect_labels(all_records)

    for split_name, records in by_split.items():
        out = processed_dir / f"{dataset}_{split_name}.jsonl"
        write_jsonl(out, records)
        print(f"Wrote {out} ({len(records)} examples)")

    (processed_dir / f"{dataset}_intent_labels.txt").write_text(
        "\n".join(intent_labels) + "\n", encoding="utf-8"
    )
    (processed_dir / f"{dataset}_slot_labels.txt").write_text(
        "\n".join(slot_labels) + "\n", encoding="utf-8"
    )
    print(f"Intents ({len(intent_labels)}): {intent_labels}")
    print(f"Slots ({len(slot_labels)})")

    # SFT pairs: train + validation only (never test).
    # Checklist files keep request/structured_output; *_sft.jsonl uses dataset/input/target.
    for split_name in ("train", "validation"):
        records = by_split[split_name]
        for fmt, suffix in (("json", "json"), ("key_value", "key_value")):
            path = processed_dir / f"{dataset}_{split_name}_pairs_{suffix}.jsonl"
            write_pairs(path, records, fmt)
            print(f"Wrote {path} ({len(records)} pairs)")
            sft_path = processed_dir / f"{dataset}_{split_name}_pairs_{suffix}_sft.jsonl"
            write_sft_pairs(sft_path, records, fmt)
            print(f"Wrote {sft_path} ({len(records)} pairs)")

    # Intent distribution sanity
    for split_name, records in by_split.items():
        counts: Dict[str, int] = defaultdict(int)
        for r in records:
            counts[r["intent"]] += 1
        print(f"{dataset}/{split_name} intent counts: {dict(sorted(counts.items()))}")

    spot_check(train + validation + test, n=10, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SNIPS / ATIS JSONL data")
    parser.add_argument(
        "--dataset",
        choices=("snips", "atis", "all"),
        default="all",
        help="Which dataset to prepare",
    )
    parser.add_argument("--seed", type=int, default=455)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "raw",
    )
    args = parser.parse_args()

    datasets = ["snips", "atis"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        print(f"\n===== Preparing {ds} =====")
        prepare_dataset(ds, args.seed, args.processed_dir, args.raw_dir)


if __name__ == "__main__":
    main()
