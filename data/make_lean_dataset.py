"""Prepare a miniF2F Lean 4 dataset for SDPO training.

Downloads individual .lean files from yangky11/miniF2F-lean4 on GitHub and
converts them to the JSON format expected by data/preprocess.py, then calls
run_proprocessing() to produce train.parquet and test.parquet.

Each file has the structure:
    import Mathlib
    set_option maxHeartbeats 0
    open BigOperators Real Nat Topology Rat
    theorem <name> <params> : <statement> := by sorry

The `ground_truth` for each example is the file content with `sorry` stripped,
i.e. everything up to and including `:= by`. The model generates the tactic body.

Usage:
    python data/make_lean_dataset.py [--output_dir datasets/lean/minif2f]

Splits:
  MiniF2F/Valid/ → train split (used during RL training)
  MiniF2F/Test/  → test  split (used for evaluation)
"""

import argparse
import json
import os
import sys
import time
import urllib.request

GITHUB_API = "https://api.github.com/repos/yangky11/miniF2F-lean4/contents/MiniF2F/{split}"
RAW_BASE = "https://raw.githubusercontent.com/yangky11/miniF2F-lean4/main/MiniF2F/{split}/{filename}"

SYSTEM_PROMPT = (
    "You are a Lean 4 expert. Write complete tactic proofs. "
    "Always present your proof inside a ```lean4 ... ``` code block. "
    "Do not use `sorry`."
)


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "make_lean_dataset/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def list_lean_files(gh_split: str) -> list:
    """Return list of .lean filenames in MiniF2F/{gh_split}/."""
    url = GITHUB_API.format(split=gh_split)
    data = json.loads(_get(url))
    return [entry["name"] for entry in data if entry["name"].endswith(".lean")]


def fetch_file(gh_split: str, filename: str) -> str:
    url = RAW_BASE.format(split=gh_split, filename=filename)
    return _get(url)


def parse_ground_truth(content: str) -> str:
    """Strip `sorry` from the end of a miniF2F file, leaving `:= by`.

    Returns None if the file doesn't look like a miniF2F theorem.
    """
    stripped = content.rstrip()
    # All files end with ':= by sorry' on one line
    if stripped.endswith(" sorry"):
        return stripped[: -len(" sorry")].rstrip()
    # Fallback: multi-line sorry block
    if "\nsorry" in stripped:
        idx = stripped.rfind("\nsorry")
        candidate = stripped[:idx].rstrip()
        if candidate.endswith("by"):
            return candidate
    return None


def extract_theorem_statement(content: str) -> str:
    """Extract just the theorem declaration line(s), without the preamble."""
    lines = content.splitlines()
    in_theorem = False
    theorem_lines = []
    for line in lines:
        if line.startswith("theorem "):
            in_theorem = True
        if in_theorem:
            theorem_lines.append(line)
    stmt = "\n".join(theorem_lines)
    if stmt.endswith(" sorry"):
        stmt = stmt[: -len(" sorry")]
    return stmt.strip()


def make_record(idx: int, theorem_name: str, content: str) -> dict:
    ground_truth = parse_ground_truth(content)
    if ground_truth is None:
        return None

    description = theorem_name.replace("_", " ")
    theorem_stmt = extract_theorem_statement(content)

    prompt = (
        f"Prove the following Lean 4 theorem:\n\n"
        f"```lean4\n{theorem_stmt}\n```\n\n"
        f"Present your complete tactic proof in a ```lean4 code block."
    )

    return {
        "idx": idx,
        "kind": "lean",
        "dataset": "lean",
        "answer": theorem_name,   # not used; ground_truth comes from tests
        "tests": ground_truth,    # Lean preamble (imports + theorem header := by)
        "description": description,
        "elo": 1500,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
    }


def download_split(gh_split: str, out_split: str, output_dir: str):
    print(f"\nFetching file list for {gh_split}/ ...")
    filenames = list_lean_files(gh_split)
    print(f"  Found {len(filenames)} .lean files")

    records = []
    skipped = 0
    for idx, filename in enumerate(sorted(filenames)):
        theorem_name = filename[: -len(".lean")]
        try:
            content = fetch_file(gh_split, filename)
            time.sleep(0.05)  # avoid GitHub rate limiting
        except Exception as e:
            print(f"  WARNING: could not fetch {filename}: {e}")
            skipped += 1
            continue

        record = make_record(idx, theorem_name, content)
        if record is None:
            print(f"  WARNING: could not parse {filename}, skipping")
            skipped += 1
            continue
        records.append(record)

    out_path = os.path.join(output_dir, f"{out_split}.json")
    with open(out_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records → {out_path}  ({skipped} skipped)")


def load_and_convert(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Valid → train, Test → test (standard miniF2F convention)
    download_split("Valid", "train", output_dir)
    download_split("Test", "test", output_dir)

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
