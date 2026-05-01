from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.naive import NaiveRewardManager


@register("lean")
class LeanRewardManager(NaiveRewardManager):
    """Reward manager for Lean 4 proofs that batches all Kimina verification calls in parallel."""

    def __call__(self, data: DataProto, return_dict: bool = True) -> torch.Tensor | dict[str, Any]:
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        from kimina_client import Snippet
        from kimina_client.models import SnippetStatus
        from verl.utils.reward_score.feedback.lean import (
            _get_client,
            extract_lean_code,
            format_lean_feedback,
        )

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Decode all responses and extract lean code in one pass
        items = []
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores
            extra_info["truncated"] = not (valid_response_ids == self.tokenizer.eos_token_id).any().item()

            lean_code = extract_lean_code(response_str)
            items.append({
                "idx": i,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "valid_response_length": valid_response_length,
                "lean_code": lean_code,
                "incorrect_format": lean_code is None,
                "was_truncated": extra_info["truncated"],
            })

        # Build one Snippet per valid example; skip format errors and truncations
        snippets = [
            Snippet(id=str(item["idx"]), code=f"{item['ground_truth']}\n{item['lean_code']}")
            for item in items
            if not item["incorrect_format"] and not item["was_truncated"]
        ]

        # Single batched call — Kimina parallelizes internally with max_workers threads
        kimina_results = {}
        conn_error = None
        if snippets:
            try:
                client = _get_client()
                check_response = client.check(snippets, max_workers=63, batch_size=1, timeout=60, show_progress=False)
                for result in check_response.results:
                    kimina_results[result.id] = result
            except Exception as e:
                conn_error = str(e)

        # Assign rewards
        already_print_data_sources = {}
        for item in items:
            i = item["idx"]
            valid_response_length = item["valid_response_length"]
            lean_code = item["lean_code"]
            incorrect_format = item["incorrect_format"]
            was_truncated = item["was_truncated"]
            data_source = item["data_source"]

            if incorrect_format or was_truncated:
                if was_truncated:
                    feedback = "Your response was truncated because it exceeded the maximum length."
                else:
                    feedback = (
                        "Your answer had the wrong format. "
                        "The Lean proof must be given in a ```lean4 ... ``` code block."
                    )
                score = {
                    "score": 0.0,
                    "acc": 0.0,
                    "pred": "",
                    "incorrect_format": int(incorrect_format),
                    "truncated": int(was_truncated),
                    "truncated_and_missing_answer": int(incorrect_format and was_truncated),
                    "feedback": feedback,
                    "lean_status": "incorrect_format",
                    "lean_time": 0.0,
                }
            elif conn_error is not None:
                score = {
                    "score": 0.0,
                    "acc": 0.0,
                    "pred": lean_code,
                    "incorrect_format": 0,
                    "truncated": 0,
                    "truncated_and_missing_answer": 0,
                    "feedback": f"Could not connect to Lean verification server: {conn_error}",
                    "lean_status": "server_unavailable",
                    "lean_time": 0.0,
                }
            else:
                result = kimina_results[str(i)]
                analysis = result.analyze()
                is_valid = analysis.status == SnippetStatus.valid
                score = {
                    "score": 1.0 if is_valid else 0.0,
                    "acc": 1.0 if is_valid else 0.0,
                    "pred": lean_code,
                    "incorrect_format": 0,
                    "truncated": 0,
                    "truncated_and_missing_answer": 0,
                    "feedback": format_lean_feedback(result, False),
                    "lean_status": analysis.status.value,
                    "lean_time": result.time or 0.0,
                }

            reward_tensor[i, valid_response_length - 1] = score["score"]
            for key, value in score.items():
                reward_extra_info[key].append(value)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", item["prompt_str"])
                print("[response]", item["response_str"])
                print("[ground_truth]", item["ground_truth"])
                for key, value in score.items():
                    print(f"[{key}]", value)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
