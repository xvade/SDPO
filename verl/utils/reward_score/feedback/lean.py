import glob
import os
import re
from typing import Optional

_client = None


def _discover_server_url() -> Optional[str]:
    discovery_dir = os.environ.get(
        "KIMINA_DISCOVERY_DIR",
        f"/mmfs1/gscratch/scrubbed/{os.environ.get('USER', 'sgvtc')}/kimina_server_discovery",
    )
    addr_files = glob.glob(os.path.join(discovery_dir, "*.addr"))
    if not addr_files:
        return None
    with open(addr_files[0]) as f:
        return f.read().strip()


def _get_client():
    global _client
    if _client is None:
        from kimina_client import KiminaClient
        url = _discover_server_url()
        if url is None:
            raise RuntimeError(
                "No Kimina server address found. Set KIMINA_DISCOVERY_DIR or start the server "
                "with: cd /gscratch/scrubbed/sgvtc/klone-kimina-setup && bash submit_server.sh"
            )
        _client = KiminaClient(api_url=url, http_timeout=120, n_retries=2)
    return _client


def extract_lean_code(response: str) -> Optional[str]:
    for pattern in (r"```lean4\s*(.*?)```", r"```lean\s*(.*?)```"):
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[-1].strip()
    return None


def format_lean_feedback(result, was_truncated: bool) -> str:
    if was_truncated:
        return "Your response was truncated because it exceeded the maximum length."

    from kimina_client.models import SnippetStatus
    analysis = result.analyze()

    if analysis.status == SnippetStatus.timeout_error:
        return f"Lean verification timed out after {result.time:.1f} seconds."

    if analysis.status in (SnippetStatus.repl_error, SnippetStatus.server_error):
        return f"Server error during verification: {result.error}"

    if analysis.status == SnippetStatus.sorry:
        sorries = (result.response.sorries if result.response else [])
        if sorries:
            s = sorries[0]
            goal_info = f" Remaining goal:\n{s.goal}" if getattr(s, "goal", None) else ""
            return f"Your proof is incomplete — it contains `sorry` at line {s.pos.line}.{goal_info}"
        return "Your proof is incomplete — it contains `sorry`."

    if analysis.status == SnippetStatus.lean_error:
        messages = (result.response.messages if result.response else [])
        errors = [m for m in messages if m.severity == "error"]
        if errors:
            lines = [f"Line {m.pos.line}: {m.data}" for m in errors[:3]]
            return "Lean reported the following error(s):\n" + "\n".join(lines)
        return "Lean reported an error."

    return ""


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    """Compute reward for a Lean 4 proof attempt.

    ground_truth: Lean preamble (imports + theorem header ending with ':= by').
    The extracted proof body is appended before sending to Kimina.
    """
    extra_info = extra_info or {}
    was_truncated = extra_info.get("truncated", False)

    lean_code = extract_lean_code(solution_str)
    incorrect_format = lean_code is None

    if incorrect_format or was_truncated:
        if was_truncated:
            feedback = "Your response was truncated because it exceeded the maximum length."
        else:
            feedback = (
                "Your answer had the wrong format. "
                "The Lean proof must be given in a ```lean ... ``` code block."
            )
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": None,
            "incorrect_format": int(incorrect_format),
            "truncated": int(was_truncated),
            "truncated_and_missing_answer": int(incorrect_format and was_truncated),
            "feedback": feedback,
            "lean_status": "incorrect_format",
            "lean_time": None,
        }

    full_code = f"{ground_truth}\n{lean_code}"

    try:
        client = _get_client()
        response = client.check(full_code, timeout=60, reuse=False, infotree="original")
        result = response.results[0]
    except Exception as e:
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": lean_code,
            "incorrect_format": 0,
            "truncated": int(was_truncated),
            "truncated_and_missing_answer": 0,
            "feedback": f"Could not connect to Lean verification server: {e}",
            "lean_status": "server_unavailable",
            "lean_time": None,
        }

    from kimina_client.models import SnippetStatus
    analysis = result.analyze()
    is_valid = analysis.status == SnippetStatus.valid
    score = 1.0 if is_valid else 0.0

    return {
        "score": score,
        "acc": score,
        "pred": lean_code,
        "incorrect_format": 0,
        "truncated": int(was_truncated),
        "truncated_and_missing_answer": 0,
        "feedback": format_lean_feedback(result, was_truncated),
        "lean_status": analysis.status.value,
        "lean_time": result.time,
    }
