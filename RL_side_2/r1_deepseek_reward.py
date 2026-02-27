# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction
from typing import Any


_STRICT_PATTERN = re.compile(
    r"(?s)\A\s*<think>\n?(?P<think>.*?)\n?</think>\s*<answer>\n?(?P<answer>.*?)\n?</answer>\s*\Z"
)


class ParseResult:
    __slots__ = ("ok", "think", "answer", "fail_reason")

    def __init__(self, ok: bool, think: str, answer: str, fail_reason: str):
        self.ok = ok
        self.think = think
        self.answer = answer
        self.fail_reason = fail_reason


def _tag_diagnostics(text: str, open_tag: str, close_tag: str, prefix: str) -> str | None:
    open_count = text.count(open_tag)
    close_count = text.count(close_tag)

    if open_count == 0 or close_count == 0:
        return f"missing_{prefix}_tag"
    if open_count > 1 or close_count > 1:
        return f"too_many_{prefix}_tags"
    if open_count != close_count:
        return f"unbalanced_{prefix}_tags"
    return None


def parse_deepseek_format(
    text: str,
    think_min_chars: int = 32,
    allow_leading_trailing_whitespace: bool = True,
) -> ParseResult:
    if not allow_leading_trailing_whitespace and text.strip() != text:
        return ParseResult(False, "", "", "outer_whitespace_not_allowed")

    think_tag_error = _tag_diagnostics(text, "<think>", "</think>", "think")
    if think_tag_error:
        return ParseResult(False, "", "", think_tag_error)

    answer_tag_error = _tag_diagnostics(text, "<answer>", "</answer>", "answer")
    if answer_tag_error:
        return ParseResult(False, "", "", answer_tag_error)

    match = _STRICT_PATTERN.match(text)
    if not match:
        return ParseResult(False, "", "", "structure_not_strict")

    think = match.group("think").strip()
    answer = match.group("answer").strip()

    if len(think) < think_min_chars:
        return ParseResult(False, think, answer, "think_too_short")
    if not answer:
        return ParseResult(False, think, answer, "answer_empty")

    nested_tags = ("<think>", "</think>", "<answer>", "</answer>")
    if any(tag in think for tag in nested_tags):
        return ParseResult(False, think, answer, "nested_tag_in_think")
    if any(tag in answer for tag in nested_tags):
        return ParseResult(False, think, answer, "nested_tag_in_answer")

    return ParseResult(True, think, answer, "")


def _normalize_text(text: str) -> str:
    norm = str(text)
    norm = norm.replace("\\\\", "\\")
    norm = norm.replace("\u2212", "-")
    norm = norm.split("=")[-1]
    norm = norm.replace("\\left", "")
    norm = norm.replace("\\right", "")
    norm = norm.replace("$", "")
    norm = norm.replace("\\,", "")
    norm = norm.replace("~", "")
    norm = norm.replace("\\ ", "")
    norm = re.sub(r"\\text\{(.*?)\}", r"\1", norm)
    norm = re.sub(r"\\boxed\{(.*?)\}", r"\1", norm)
    norm = norm.replace(",", "")
    norm = re.sub(r"\s+", " ", norm.strip().lower())
    return norm


def _unwrap_wrappers(text: str) -> str:
    wrapped = text.strip()
    changed = True
    while changed:
        changed = False
        for prefix, suffix in (("(", ")"), ("[", "]"), ("{", "}")):
            if wrapped.startswith(prefix) and wrapped.endswith(suffix) and len(wrapped) >= 2:
                wrapped = wrapped[1:-1].strip()
                changed = True
    return wrapped


def _latex_frac_to_plain(text: str) -> str:
    frac_pattern = re.compile(r"\\frac\{\s*([^{}]+)\s*\}\{\s*([^{}]+)\s*\}")

    prev = None
    out = text
    while out != prev:
        prev = out
        out = frac_pattern.sub(r"(\1)/(\2)", out)
    return out


def _to_numeric_value(text: str) -> Decimal | None:
    s = _normalize_text(text)
    s = _unwrap_wrappers(s)
    s = _latex_frac_to_plain(s)
    s = re.sub(r"^\(([^()]+)\)/\(([^()]+)\)$", r"\1/\2", s)
    s = _unwrap_wrappers(s)
    s = s.replace(" ", "")

    if not s:
        return None

    # Simple percentage support: 25% == 0.25
    if s.endswith("%"):
        inner = _to_numeric_value(s[:-1])
        if inner is None:
            return None
        return inner / Decimal("100")

    # Simple rational numbers like 1/2, -3/4
    if re.fullmatch(r"[+-]?\d+/[+-]?\d+", s):
        try:
            frac = Fraction(s)
            return Decimal(frac.numerator) / Decimal(frac.denominator)
        except Exception:
            return None

    # Decimal / integer / scientific notation
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", s):
        try:
            return Decimal(s)
        except Exception:
            return None

    return None


def _answers_equal(pred_text: str, gt_text: str) -> tuple[bool, str]:
    pred_norm = _normalize_text(pred_text)
    gt_norm = _normalize_text(gt_text)
    if pred_norm == gt_norm:
        return True, "exact"

    pred_num = _to_numeric_value(pred_text)
    gt_num = _to_numeric_value(gt_text)
    if pred_num is None or gt_num is None:
        return False, "mismatch"

    diff = abs(pred_num - gt_num)
    tolerance = max(Decimal("1e-8"), Decimal("1e-6") * max(abs(pred_num), abs(gt_num), Decimal("1")))
    if diff <= tolerance:
        return True, "numeric"
    return False, "numeric_mismatch"


def _extract_after_hashes(text: str) -> str:
    if "####" in text:
        return text.rsplit("####", 1)[-1].strip()
    return text.strip()


def _extract_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None

    i = start
    open_braces = 0
    end = None
    while i < len(text):
        if text[i] == "{":
            open_braces += 1
        elif text[i] == "}":
            open_braces -= 1
            if open_braces == 0:
                end = i
                break
        i += 1

    if end is None:
        return None
    raw = text[start : end + 1]
    return raw[len(marker) : -1]


def _candidate_variants(raw: str) -> list[str]:
    variants = [str(raw)]
    variants.append(_extract_after_hashes(raw))
    boxed = _extract_boxed(raw)
    if boxed is not None:
        variants.append(boxed)
    return variants


def _iter_gt_values(ground_truth: Any):
    if ground_truth is None:
        return
    if isinstance(ground_truth, (str, int, float)):
        yield str(ground_truth)
        return
    if isinstance(ground_truth, dict):
        keys = (
            "ground_truth",
            "target",
            "targets",
            "answer",
            "answers",
            "solution",
            "final_answer",
            "gold",
            "label",
        )
        yielded = False
        for key in keys:
            if key in ground_truth:
                yielded = True
                yield from _iter_gt_values(ground_truth[key])
        if not yielded:
            for value in ground_truth.values():
                yield from _iter_gt_values(value)
        return
    if isinstance(ground_truth, (list, tuple, set)):
        for value in ground_truth:
            yield from _iter_gt_values(value)
        return
    yield str(ground_truth)


def _get_gt_candidates(ground_truth: Any) -> list[str]:
    normalized_candidates: list[str] = []
    seen: set[str] = set()

    for raw in _iter_gt_values(ground_truth):
        for variant in _candidate_variants(raw):
            norm = _normalize_text(variant)
            if norm and norm not in seen:
                seen.add(norm)
                normalized_candidates.append(norm)

    return normalized_candidates


def answer_reward_fn(parsed: ParseResult, ground_truth: Any) -> tuple[float, bool, str, str]:
    if not parsed.ok:
        return 0.0, False, "[INVALID]", ""

    pred_text = parsed.answer
    pred_norm = _normalize_text(pred_text)
    pred_candidates: list[str] = []
    pred_seen: set[str] = set()
    for variant in _candidate_variants(pred_text):
        norm = _normalize_text(variant)
        if norm and norm not in pred_seen:
            pred_seen.add(norm)
            pred_candidates.append(norm)
    gt_candidates = _get_gt_candidates(ground_truth)
    is_correct = False
    for pred in pred_candidates:
        for gt in gt_candidates:
            equal, _ = _answers_equal(pred, gt)
            if equal:
                is_correct = True
                break
        if is_correct:
            break
    return float(is_correct), is_correct, pred_text, pred_norm


def _resolve_response_length(solution_str: str, extra_info: dict[str, Any], fallback_mode: str) -> int:
    if "response_length" in extra_info and extra_info["response_length"] is not None:
        return int(extra_info["response_length"])
    if "valid_response_length" in extra_info and extra_info["valid_response_length"] is not None:
        return int(extra_info["valid_response_length"])
    if "response_token_length" in extra_info and extra_info["response_token_length"] is not None:
        return int(extra_info["response_token_length"])
    if "completion_tokens" in extra_info and extra_info["completion_tokens"] is not None:
        return int(extra_info["completion_tokens"])

    if fallback_mode == "char_count":
        return len(solution_str)
    return len(solution_str.split())


def length_reward_fn(
    solution_str: str,
    extra_info: dict[str, Any] | None,
    *,
    max_len: int,
    pass_len: int,
    fallback_mode: str,
) -> tuple[float, int, int]:
    info = dict(extra_info or {})
    response_length = _resolve_response_length(solution_str, info, fallback_mode=fallback_mode)

    if response_length > max_len:
        return 0.0, response_length, max_len
    return float(response_length <= pass_len), response_length, max_len


def _fail_flags(reason: str) -> dict[str, float]:
    return {
        "fail_missing_think_tag": float(reason == "missing_think_tag"),
        "fail_too_many_think_tags": float(reason == "too_many_think_tags"),
        "fail_unbalanced_think_tags": float(reason == "unbalanced_think_tags"),
        "fail_missing_answer_tag": float(reason == "missing_answer_tag"),
        "fail_too_many_answer_tags": float(reason == "too_many_answer_tags"),
        "fail_unbalanced_answer_tags": float(reason == "unbalanced_answer_tags"),
        "fail_structure_not_strict": float(reason == "structure_not_strict"),
        "fail_think_too_short": float(reason == "think_too_short"),
        "fail_answer_empty": float(reason == "answer_empty"),
        "fail_nested_tag": float(reason in {"nested_tag_in_think", "nested_tag_in_answer"}),
        "fail_outer_whitespace": float(reason == "outer_whitespace_not_allowed"),
    }


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del data_source

    w_answer = float(kwargs.get("w_answer", 1.0))
    w_format = float(kwargs.get("w_format", 0.2))
    w_length = float(kwargs.get("w_length", 0.1))

    think_min_chars = int(kwargs.get("think_min_chars", 32))
    allow_outer_ws = bool(kwargs.get("allow_leading_trailing_whitespace", True))
    require_format_for_answer = bool(kwargs.get("require_format_for_answer", True))

    max_len = int(kwargs.get("max_len", 16 * 1024))
    pass_len = int(kwargs.get("pass_len", 15 * 1024))
    fallback_mode = str(kwargs.get("length_fallback_mode", "word_count"))

    parsed = parse_deepseek_format(
        solution_str,
        think_min_chars=think_min_chars,
        allow_leading_trailing_whitespace=allow_outer_ws,
    )
    format_reward = float(parsed.ok)

    answer_reward, answer_correct, pred_answer, pred_answer_norm = answer_reward_fn(parsed, ground_truth)
    if require_format_for_answer and not parsed.ok:
        answer_reward = 0.0
        answer_correct = False
        pred_answer = "[INVALID]"
        pred_answer_norm = ""

    length_reward, response_length, effective_max_len = length_reward_fn(
        solution_str,
        extra_info,
        max_len=max_len,
        pass_len=pass_len,
        fallback_mode=fallback_mode,
    )

    score = w_answer * answer_reward + w_format * format_reward + w_length * length_reward
    fail_reason = parsed.fail_reason

    result = {
        "score": float(score),
        "answer_reward": float(answer_reward),
        "format_reward": float(format_reward),
        "length_reward": float(length_reward),
        "format_ok": float(parsed.ok),
        "answer_correct": float(answer_correct),
        "within_length_limit": float(length_reward > 0.5),
        "response_length": float(response_length),
        "max_len": float(effective_max_len),
        "pass_len": float(pass_len),
        "think_content_len": float(len(parsed.think)),
        "answer_text_len": float(len(parsed.answer)),
        "num_gt_candidates": float(len(_get_gt_candidates(ground_truth))),
        "pred_answer": pred_answer,
        "pred_answer_norm": pred_answer_norm,
        "format_fail_reason": fail_reason,
        "w_answer": float(w_answer),
        "w_format": float(w_format),
        "w_length": float(w_length),
    }
    result.update(_fail_flags(fail_reason))
    return result
