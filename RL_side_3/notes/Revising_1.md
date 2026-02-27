# Revising_1: Replay Priority Revision

## Goal

Current replay selection over-samples very recent QueryGroups, which can behave like repeating the same Q-G pairs for extra steps.
This revision keeps M2 gating, but replaces candidate ranking with a unified score that combines:

1. uncertainty preference (`p` near 0.5),
2. recency penalty for recently trained samples.

## New Priority Score

For each query-group `g` at step `t`:

- `p_g`: success probability proxy (existing binary score-based estimate),
- `t_last(g)`: last step where `g` was used in actor update (on-policy or replay),
- `age_g = t - t_last(g)`.

Define:

- `u_g = 2 * |p_g - 0.5|`  (smaller is better, `[0, 1]`)
- `r_g = exp(-age_g / lambda)` (recently used -> larger penalty)
- `S_g = u_g + beta * r_g`

Sort replay candidates by ascending `S_g` (smaller first), then apply existing M2 gating (`M2 <= tau`).

## Default Revision Setting (E)

Use the discussed setting as default:

- `beta = 1.0`
- `lambda = 4.0`

Interpretation:

- strong recency penalty,
- slow decay across several steps,
- prevents immediate repeated replay of the same recent Q-Gs.

## Implementation Scope

1. Add `last_training_step` metadata to `QueryGroup`.
2. Initialize `last_training_step = insertion_step` for newly built on-policy groups.
3. In replay selection, compute `S_g` with current step and sort by `S_g`.
4. After replay groups are actually used, set `last_training_step = current_step`.
5. Keep M2 gating and existing batch divisibility logic unchanged.

## Config Controls

Introduce optional replay selection controls under `algorithm.m2_replay.selection`:

- `recent_learning_beta` (default `1.0`)
- `recent_learning_decay_lambda` (default `4.0`)

When keys are absent, defaults above are used.

## Expected Effect

Compared to pure `|p-0.5|` ranking:

- lower probability of selecting just-trained groups again in adjacent steps,
- better replay diversity under fixed buffer,
- preserve stability via unchanged M2 gate.
