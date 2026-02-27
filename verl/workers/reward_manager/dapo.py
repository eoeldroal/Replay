# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import re

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    """The reward manager."""

    LEGACY_PARSER_MODE = "legacy_answer_v1"
    THINK_ANSWER_PARSER_MODE = "think_answer_v1"

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        if hasattr(cfg, "get"):
            try:
                return cfg.get(key, default)
            except TypeError:
                pass
        return getattr(cfg, key, default)

    def _compute_length_reward_components(self, valid_response_length: int):
        overlong_buffer_len = self._cfg_get(self.overlong_buffer_cfg, "len")
        if overlong_buffer_len is None or overlong_buffer_len <= 0:
            raise ValueError(f"Invalid overlong_buffer_cfg.len={overlong_buffer_len}")

        expected_len = self.max_resp_len - overlong_buffer_len
        exceed_len = valid_response_length - expected_len
        is_overlong = exceed_len > 0

        length_base = 0.0 if is_overlong else 1.0
        weight = float(self._cfg_get(self.overlong_buffer_cfg, "weight", 1.0))
        overlong_reward = length_base * weight

        return float(overlong_reward), bool(is_overlong), float(length_base)

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        answer_score_cfg=None,
        format_reward_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.answer_score_cfg = answer_score_cfg
        self.format_reward_cfg = format_reward_cfg
        self.max_resp_len = max_resp_len

        if self._cfg_get(self.overlong_buffer_cfg, "enable", False):
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            overlong_buffer_len = self._cfg_get(self.overlong_buffer_cfg, "len")
            assert overlong_buffer_len is not None and overlong_buffer_len > 0, (
                f"overlong_buffer_cfg.len must be positive, but got {overlong_buffer_len}"
            )
            assert self.max_resp_len >= overlong_buffer_len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def _infer_parse_success(self, result: dict | None) -> bool:
        if not isinstance(result, dict):
            return False

        if "parse_success" in result:
            return bool(result["parse_success"])

        pred = result.get("pred", None)
        if pred is None:
            return False
        if isinstance(pred, str):
            return pred not in ("", "[INVALID]")
        return True

    def _get_format_parser_mode(self) -> str:
        parser_mode = str(self._cfg_get(self.format_reward_cfg, "parser_mode", self.LEGACY_PARSER_MODE)).lower()
        if parser_mode in (self.LEGACY_PARSER_MODE, "legacy", "answer_only"):
            return self.LEGACY_PARSER_MODE
        if parser_mode in (self.THINK_ANSWER_PARSER_MODE, "think_answer", "think"):
            return self.THINK_ANSWER_PARSER_MODE
        raise ValueError(f"Unknown format_reward_cfg.parser_mode={parser_mode}")

    def _compute_think_answer_parse(self, response_str: str) -> dict:
        """Parse think/answer format:
        - well-formed <think>...</think> pair(s)
        - answer appears after the last </think> when enabled
        - think content length reaches configured minimum
        """
        open_tag = "<think>"
        close_tag = "</think>"
        think_min_chars = int(self._cfg_get(self.format_reward_cfg, "think_min_chars", 1))
        require_answer_after_think = bool(self._cfg_get(self.format_reward_cfg, "require_answer_after_think", True))
        answer_pattern = str(self._cfg_get(self.format_reward_cfg, "answer_pattern", r"(?i)Answer\s*:\s*([^\n]+)"))

        open_spans = [(m.start(), m.end()) for m in re.finditer(re.escape(open_tag), response_str)]
        close_spans = [(m.start(), m.end()) for m in re.finditer(re.escape(close_tag), response_str)]

        tokens = [(s, "open", e) for s, e in open_spans] + [(s, "close", e) for s, e in close_spans]
        tokens.sort(key=lambda x: x[0])

        depth = 0
        think_pair_ok = True
        fail_reason = ""
        think_block_count = 0
        think_char_count = 0
        last_close_end = -1
        current_open_end = -1

        if not open_spans or not close_spans:
            think_pair_ok = False
            fail_reason = "missing_think_pair"
        else:
            for pos, kind, end_pos in tokens:
                if kind == "open":
                    if depth == 0:
                        current_open_end = end_pos
                    depth += 1
                    continue

                if depth == 0:
                    think_pair_ok = False
                    fail_reason = "close_before_open"
                    break

                depth -= 1
                if depth == 0:
                    think_block_count += 1
                    think_text = response_str[current_open_end:pos].strip()
                    think_char_count += len(think_text)
                    last_close_end = end_pos

            if think_pair_ok and depth != 0:
                think_pair_ok = False
                fail_reason = "unclosed_think_tag"

        think_min_len_ok = think_pair_ok and think_char_count >= think_min_chars

        answer_after_think = False
        extracted_answer = ""
        if think_pair_ok and last_close_end >= 0:
            post_think_text = response_str[last_close_end:]
            post_matches = re.findall(answer_pattern, post_think_text)
            answer_after_think = len(post_matches) > 0
            if require_answer_after_think:
                extracted_answer = post_matches[-1].strip() if post_matches else ""
            else:
                full_matches = re.findall(answer_pattern, response_str)
                extracted_answer = full_matches[-1].strip() if full_matches else ""

        answer_non_empty = extracted_answer not in ("", "[INVALID]")
        parse_success = think_pair_ok and think_min_len_ok and answer_non_empty
        if require_answer_after_think:
            parse_success = parse_success and answer_after_think

        if not fail_reason:
            if require_answer_after_think and not answer_after_think:
                fail_reason = "answer_not_after_think"
            elif not answer_non_empty:
                fail_reason = "invalid_or_empty_answer"
            elif not think_min_len_ok:
                fail_reason = "think_too_short"

        return {
            "parse_success_active": bool(parse_success),
            "think_pair_ok": bool(think_pair_ok),
            "think_min_len_ok": bool(think_min_len_ok),
            "think_block_count": int(think_block_count),
            "think_char_count": int(think_char_count),
            "answer_after_think": bool(answer_after_think),
            "require_answer_after_think": bool(require_answer_after_think),
            "format_fail_reason": fail_reason,
        }

    def _compute_format_reward(self, response_str: str, result: dict | None) -> tuple[float, bool, dict]:
        """Compute format reward from parser-mode-specific parse success."""
        if not self._cfg_get(self.format_reward_cfg, "enable", False):
            return 0.0, False, {}

        parser_mode = self._get_format_parser_mode()
        parse_success_legacy = self._infer_parse_success(result)
        if parser_mode == self.LEGACY_PARSER_MODE:
            format_debug = {
                "format_parser_mode": parser_mode,
                "parse_success_active": bool(parse_success_legacy),
                "parse_success_legacy": bool(parse_success_legacy),
            }
            parse_success = bool(parse_success_legacy)
        else:
            format_debug = self._compute_think_answer_parse(response_str=response_str)
            format_debug["format_parser_mode"] = parser_mode
            format_debug["parse_success_legacy"] = bool(parse_success_legacy)
            parse_success = bool(format_debug["parse_success_active"])

        format_base = 1.0 if parse_success else 0.0
        weight = float(self._cfg_get(self.format_reward_cfg, "weight", 0.0))
        format_reward = format_base * weight
        return format_reward, parse_success, format_debug

    def _map_answer_score(self, score: float, result: dict | None):
        """Remap correctness to binary base (0/1) scaled by weight."""
        if not self._cfg_get(self.answer_score_cfg, "enable", False):
            return float(score)

        if not isinstance(result, dict) or "acc" not in result:
            return float(score)

        is_correct = bool(result["acc"])
        answer_base = 1.0 if is_correct else 0.0
        weight = float(self._cfg_get(self.answer_score_cfg, "weight", 1.0))
        return answer_base * weight

    def _apply_answer_format_gate(self, mapped_score: float, format_parse_success: bool) -> float:
        """Optionally force answer reward when format fails."""
        if not self._cfg_get(self.format_reward_cfg, "enable", False):
            return float(mapped_score)
        if not self._cfg_get(self.answer_score_cfg, "require_format_success", False):
            return float(mapped_score)
        if format_parse_success:
            return float(mapped_score)
        return float(self._cfg_get(self.answer_score_cfg, "format_fail_value", 0.0))

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})

            extra_info["rollout_reward_scores"] = rollout_reward_scores

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            format_reward, format_parse_success, format_debug = self._compute_format_reward(
                response_str=response_str, result=result if isinstance(result, dict) else None
            )

            mapped_score = self._map_answer_score(score=score, result=result if isinstance(result, dict) else None)
            effective_answer_score = self._apply_answer_format_gate(
                mapped_score=mapped_score, format_parse_success=format_parse_success
            )

            if self._cfg_get(self.answer_score_cfg, "log", False):
                reward_extra_info["answer_score_raw"].append(float(score))
                reward_extra_info["answer_score_mapped"].append(float(mapped_score))
                reward_extra_info["answer_score_effective"].append(float(effective_answer_score))
            reward = effective_answer_score

            reward += format_reward
            if self._cfg_get(self.format_reward_cfg, "log", False):
                reward_extra_info["format_reward"].append(float(format_reward))
                for key, value in format_debug.items():
                    reward_extra_info[key].append(value)

            if self._cfg_get(self.overlong_buffer_cfg, "enable", False):
                overlong_reward, overlong, length_reward = self._compute_length_reward_components(
                    valid_response_length=valid_response_length
                )
                reward += overlong_reward
                if self._cfg_get(self.overlong_buffer_cfg, "log", False):
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong)
                    reward_extra_info["length_reward"].append(length_reward)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
