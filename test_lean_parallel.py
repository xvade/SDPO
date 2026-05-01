"""Stress-test LeanRewardManager parallelism against the live Kimina server.

Loads 64 preambles from the test split, sends each with a `simp` proof attempt
(succeeds for some, fails for others), times the batch call with max_workers=63
and batch_size=1, and prints a summary.
"""

import json
import os
import time

os.environ.setdefault(
    "KIMINA_DISCOVERY_DIR",
    "/mmfs1/gscratch/scrubbed/sgvtc/kimina_server_discovery",
)

from kimina_client import Snippet
from kimina_client.models import SnippetStatus
from verl.utils.reward_score.feedback.lean import _get_client

N = 64
PROOF_ATTEMPT = "  simp"  # fast tactic — succeeds for some theorems, fails for others

# Load preambles from test.json
records = []
with open("datasets/lean/minif2f/test.json") as f:
    for line in f:
        records.append(json.loads(line))
        if len(records) >= N:
            break

snippets = [
    Snippet(id=str(i), code=f"{rec['tests']}\n{PROOF_ATTEMPT}")
    for i, rec in enumerate(records)
]

print(f"Sending {len(snippets)} snippets with max_workers=63, batch_size=1 ...")
client = _get_client()

t0 = time.perf_counter()
response = client.check(snippets, max_workers=63, batch_size=1, timeout=15, show_progress=True)
elapsed = time.perf_counter() - t0

status_counts = {}
times = []
for result in response.results:
    status = result.analyze().status.value
    status_counts[status] = status_counts.get(status, 0) + 1
    if result.time is not None:
        times.append(result.time)

print(f"\n--- Results ({len(response.results)} proofs in {elapsed:.1f}s wall time) ---")
for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
    print(f"  {status:20s}: {count}")
if times:
    print(f"\nPer-proof times (Kimina-side):")
    print(f"  min={min(times):.2f}s  median={sorted(times)[len(times)//2]:.2f}s  max={max(times):.2f}s  sum={sum(times):.1f}s")
    print(f"Wall time / sum(proof_times) = {elapsed:.1f}s / {sum(times):.1f}s  "
          f"(effective parallelism: {sum(times)/elapsed:.1f}x)")
