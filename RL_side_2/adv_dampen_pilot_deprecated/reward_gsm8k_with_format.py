# Custom reward function for GSM8K with format reward.
# Used via verl's custom_reward_function mechanism so the global
# default_compute_score router is never modified.
#
# Reward scheme:
#   #### <correct>  → 1.0  (correctness)
#   #### <wrong>    → 0.1  (format only)
#   no #### format  → 0.0  (nothing)

from verl.utils.reward_score.gsm8k import compute_score as _gsm8k_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return _gsm8k_score(solution_str, ground_truth, format_score=0.1)
