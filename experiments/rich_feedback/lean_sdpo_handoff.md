# Lean SDPO Handoff

## What was built

A custom reward function for SDPO that verifies Lean 4 proofs using the Kimina server,
integrated into the existing rich-feedback training pipeline.

## Files created / modified

- `verl/utils/reward_score/feedback/lean.py` — new reward function
- `verl/utils/reward_score/feedback/__init__.py` — added routing for `data_source in ["lean", "minif2f", "lean4"]`
- `data/make_lean_dataset.py` — downloads miniF2F from `yangky11/miniF2F-lean4` on GitHub
- `data/preprocess.py` — added `elif reward_style == "lean": solution = tests` (mirrors code branch)
- `experiments/rich_feedback/run_lean_sdpo.sh` — training submission script

## Non-obvious technical decisions

### Dataset format
- `tests` field holds the Lean preamble (imports + theorem header ending `:= by`), not `answer`.
  This mirrors how code datasets work (`answer` is unused; `tests` becomes `reward_model.ground_truth`).
- `answer` is set to the theorem name (human-readable label, not used by reward fn).
- `data_source = "lean"` routes to the lean reward function via `feedback/__init__.py`.
- miniF2F `Valid/` → train split, `Test/` → test split.

### Reward function interface
- `ground_truth` received by `lean.compute_score` is the full Lean preamble (the file content
  with `sorry` stripped). The model's extracted proof body is concatenated as:
  `full_code = f"{ground_truth}\n{lean_code}"` before sending to Kimina.
- `result.response` from `KiminaClient.check()` is a plain dict at runtime, not a Pydantic model.
  All field access goes through `_resp_get()` / `_item_get()` helpers in `lean.py`.
- Code blocks must be ` ```lean4 ``` ` (not ` ```lean ``` `). System prompt and format-error
  feedback both say `lean4`. `extract_lean_code` tries `lean4` first, falls back to `lean`.

### Kimina server wiring
- Server address discovered from `$KIMINA_DISCOVERY_DIR/*.addr` (default:
  `/mmfs1/gscratch/scrubbed/sgvtc/kimina_server_discovery`).
- `KIMINA_DISCOVERY_DIR` is exported into the Apptainer container in `setup_cmds` in
  `run_lean_sdpo.sh`. Without this export the reward workers cannot find the server.
- `kimina_client` installed from `/mmfs1/gscratch/scrubbed/sgvtc/kimina-engine` (the repo root,
  where `pyproject.toml` lives). The `client/` subdirectory alone is NOT installable.
- `run_lean_sdpo.sh` has a preflight check that aborts if no `.addr` file exists.

### Container / path binding
- `run_command_in_apptainer.sh` binds `/gscratch/scrubbed/sgvtc/SDPO` → `/users/sgvtc/SDPO`.
  This is why `user.yaml` uses `/users/sgvtc/SDPO` as the base path.
- `kimina_server_discovery` and `kimina-engine` are NOT bound by default in the training
  container — they are accessed via their `/mmfs1/gscratch/scrubbed/sgvtc/` paths, which
  Apptainer exposes because `/mmfs1` is available inside the container.

### Data prep
- Run `make_lean_dataset.py` from a **login node** (needs internet for GitHub) inside the
  Apptainer container (needs `pyarrow` and `datasets`):
  ```bash
  cd /gscratch/scrubbed/sgvtc/SDPO
  apptainer exec \
    --bind "$(pwd):/users/sgvtc/SDPO" \
    sdpo-gh200.sif \
    bash -c "cd /users/sgvtc/SDPO && pip install -e . -q && python data/make_lean_dataset.py"
  ```
- Dataset written to `datasets/lean/minif2f/` (244 train, 244 test, all parsed successfully).

## Current state

- Dataset: ready at `datasets/lean/minif2f/train.parquet` + `test.parquet`
- Reward function: tested end-to-end against live Kimina server — all 4 cases pass
- Training: job 34893528 ran with NaiveRewardManager (sequential, ~85 min/step)
- Kimina server: SLURM job 34893410 on n3465

### Parallel reward computation: DONE

`verl/workers/reward_manager/lean.py` — `@register("lean") class LeanRewardManager`
- Decodes all responses, extracts lean code
- Skips Kimina for format errors / truncations (reward=0 immediately)
- Single `client.check(all_snippets, max_workers=15, show_progress=False)` call
  — Kimina batches internally (batch_size=8), 15 threads ≈ 15× speedup
- Maps results back by `Snippet.id == str(batch_index)`
- `reward_model.reward_manager=lean` added to `run_lean_sdpo.sh`

**Next training run will use `LeanRewardManager` automatically.**

## User preferences observed

- Prefers terse responses with no trailing summaries.
- Wants non-obvious decisions explained, not file contents restated.
- Uses `lean4` fenced blocks, not `lean`.
- Follows the same structural patterns as existing experiments (e.g., sciknoweval) —
  don't invent new patterns when an existing one fits.
- Runs training on `gpu-l40s` with 2 GPUs, account `amath`.
- Model: `Qwen/Qwen2.5-3B-Instruct` for initial experiments.
