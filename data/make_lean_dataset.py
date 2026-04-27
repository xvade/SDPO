"""Prepare a miniF2F Lean 4 dataset for SDPO training.

Downloads miniF2F from HuggingFace and converts it to the JSON format
expected by data/preprocess.py, then calls run_proprocessing() to produce
train.parquet and test.parquet.

Usage:
    python data/make_lean_dataset.py [--output_dir datasets/lean/minif2f]

The HuggingFace dataset used is `cat-searcher/minif2f-lean4`. Each split
("valid" and "test") is mapped as follows:
  - valid  → train split (used during RL training)
  - test   → test  split (used for evaluation)

Output JSON schema (matches make_map_fn in preprocess.py):
  idx, kind, dataset, answer, tests, description, elo, prompt, system
"""

import argparse
import json
import os
import sys

SYSTEM_PROMPT = (
    "You are a Lean 4 expert. Write complete tactic proofs. "
    "Always present your proof inside a ```lean ... ``` code block. "
    "Do not use `sorry`."
)

HF_DATASET = "cat-searcher/minif2f-lean4"


def make_record(idx: int, example: dict, split: str) -> dict:
    """Convert one miniF2F row into the SDPO JSON record format."""
    formal = example.get("formal_statement", "").strip()
    informal = (
        example.get("informal_stmt")
        or example.get("header")
        or example.get("problem_name")
        or formal
    ).strip()

    # ground_truth = the full Lean preamble the model's proof body is appended to.
    # miniF2F formal_statement already includes imports and ends with ":= by" or similar.
    # If not, we add ":= by" so the model only needs to supply tactic steps.
    if not formal.rstrip().endswith("by"):
        answer = formal.rstrip() + " := by"
    else:
        answer = formal

    prompt = (
        f"Prove the following theorem in Lean 4:\n\n"
        f"{informal}\n\n"
        f"Formal statement:\n{formal}\n\n"
        f"Present your tactic proof in a ```lean code block."
    )

    return {
        "idx": idx,
        "kind": "lean",
        "dataset": "lean",
        "answer": answer,
        "tests": "-",
        "description": informal,
        "elo": 1500,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
    }


def load_and_convert(output_dir: str):
    try:
        import datasets as hf_datasets
    except ImportError:
        print("ERROR: install the `datasets` package first: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {HF_DATASET} from HuggingFace...")
    ds = hf_datasets.load_dataset(HF_DATASET)

    os.makedirs(output_dir, exist_ok=True)

    # miniF2F uses "valid" and "test" splits; we map valid→train, test→test.
    split_map = {}
    for hf_split, out_split in [("valid", "train"), ("test", "test")]:
        if hf_split in ds:
            split_map[hf_split] = out_split
        else:
            print(f"WARNING: split '{hf_split}' not found in dataset, skipping.")

    if not split_map:
        print("ERROR: no usable splits found.", file=sys.stderr)
        sys.exit(1)

    for hf_split, out_split in split_map.items():
        records = []
        for idx, example in enumerate(ds[hf_split]):
            records.append(make_record(idx, example, out_split))

        out_path = os.path.join(output_dir, f"{out_split}.json")
        with open(out_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {len(records)} records → {out_path}")

    # Produce parquet via the existing preprocessing pipeline.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.preprocess import run_proprocessing
    print(f"\nRunning preprocess.py on {output_dir} ...")
    run_proprocessing(output_dir)
    print("Done. Parquet files written to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare miniF2F Lean 4 dataset for SDPO.")
    parser.add_argument(
        "--output_dir",
        default="datasets/lean/minif2f",
        help="Directory to write train.json, test.json, train.parquet, test.parquet",
    )
    args = parser.parse_args()
    load_and_convert(args.output_dir)
