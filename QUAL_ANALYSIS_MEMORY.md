# Qualitative Analysis Memory (GCISPO v2 vs CISPO)

## Scope Lock (Do Not Deviate)
- `v2` W&B run: `DDAI_revised/verl/wandb/run-20260214_142412-u3jfs85c/files/*`
- `GCISPO v2 mapped rollout`: `DDAI_revised/verl/logs/GCISPO_pilot_rollout/1.jsonl` ... `162.jsonl`
- `GCISPO v2 mapped validation`: `DDAI_revised/verl/logs/GCISPO_pilot_validation/{10,20,...,160,162}.jsonl`
- `CISPO baseline rollout`: `DDAI_revised/verl/logs/CISPO_test_rollout/1.jsonl` ... `162.jsonl`
- `CISPO baseline validation`: `DDAI_revised/verl/logs/CISPO_test_validation/{10,20,...,160,162}.jsonl`

## Exclusions
- Any `v3` logs under `GCISPO_pilot_v3_*`
- Other W&B runs unless explicitly needed for provenance check
- Purely numeric-only claims without text evidence

## Key Facts (Living)
- `GCISPO.pilot.v2` writes checkpoints under `GCISPO.pilot.v2` but rollout/validation dumps to `GCISPO_pilot_*` paths.
- Qualitative analysis must be case-based: read generated text and error modes, not only metrics.
- Cross-model comparison must match by `input` text within the same step; same line index is not guaranteed to be the same prompt.
- Late-stage reversal is not a single failure mode: combinatorial under/over-counting, symmetry miss, and extraction/format failures co-exist.
- Clipping/truncation signatures appear in both models in 70-95, with stronger incidence noted in CISPO in one sub-agent pass.

## Evidence Log
### Fact
- Status: validated
- Claim: Prompt alignment by raw line index can be wrong across methods.
- Source: `DDAI_revised/verl/logs/GCISPO_pilot_rollout/*`, `DDAI_revised/verl/logs/CISPO_test_rollout/*`
- Evidence: Cross-window comparator explicitly re-matched by full prompt text and found index mismatches.
- Implication: All head-to-head qualitative cards must be prompt-matched first.
- Confidence: high

- Status: validated
- Claim: Transition/late degradation includes many combinatorial and symmetry bookkeeping failures.
- Source: `DDAI_revised/verl/logs/GCISPO_pilot_rollout/55..162.jsonl`, `DDAI_revised/verl/logs/CISPO_test_rollout/55..162.jsonl`
- Evidence: Recurrent classes: inclusion-exclusion mistakes, missing invalid-case filtering, permutation overcount, domain-constraint violations.
- Implication: Pure clipping fix is insufficient; reasoning-structure guardrails are needed.
- Confidence: high

- Status: validated
- Claim: Formatting/extraction failures are independent from reasoning quality and hurt both methods.
- Source: `DDAI_revised/verl/logs/GCISPO_pilot_rollout/1..54.jsonl`, `DDAI_revised/verl/logs/CISPO_test_rollout/1..54.jsonl`
- Evidence: Cases with correct-looking chain but parser-unfriendly finals (`\\boxed{}`, prose-only ending, duplicated answer lines).
- Implication: Need output normalization layer before reward/extract.
- Confidence: high

### Hypothesis
- Status: updated
- Claim: H1' (refined): late GCISPO catch-up is driven by mixed effects: update-constraint pressure + class-specific reasoning failures, not one single clip mechanism.
- Source: Multi-agent qualitative pass (A1/A2/A3/A4/B1/A5b/B2/B3)
- Evidence: Supports: truncation and abrupt endings in 70-95, late combinatorial misses. Counterevidence: many correct late GCISPO cases remain.
- Implication: Intervention must be multi-pronged (clip/length + reasoning checks + extraction normalization).
- Confidence: medium-high

- Status: open
- Claim: H2: reward-induced mode-seeking on common numeric patterns causes wrong confident finals on rare-structure prompts.
- Source: B3 late-window critique
- Evidence: Repeated wrong numeric attractors on some late cases despite plausible chains.
- Implication: Add diversity/anti-collapse term or supervised anchor.
- Confidence: medium

- Status: open
- Claim: H3: domain-specific drift (geometry/combinatorics) explains much of reversal more than global capability drop.
- Source: A4/A5b/B2
- Evidence: Some domains remain robust while others degrade.
- Implication: Domain-aware ablation needed.
- Confidence: medium

### Counterevidence
- Status: validated
- Claim: GCISPO still wins many late examples; reversal is not universal collapse.
- Source: `DDAI_revised/verl/logs/GCISPO_pilot_rollout/121..162.jsonl` vs CISPO matched prompts
- Evidence: Late-window cards include GCISPO wins and ties on geometric/symmetry tasks.
- Implication: Avoid overclaiming “global degradation”; target specific failure classes.
- Confidence: medium-high

### Decision
- Status: finalized
- Claim: Use three-layer diagnosis for next iteration: (1) extraction/format robustness, (2) clipping/length policy, (3) combinatorial verification checks.
- Source: Round 1 and Round 2 synthesis
- Evidence: Each layer maps to recurring observed error classes.
- Implication: Experiments should isolate these layers via ablation.
- Confidence: high

### Todo
- Status: open
- Task: Build prompt-matched casebank CSV for steps 60-95 and 121-162 with gc_line/cis_line/cause_class.
- Owner: main agent
- Source: B1, A3, A5b
- Priority: high

- Status: open
- Task: Draft controlled ablation matrix: clip range widening, output normalization, combinatorial self-check toggle.
- Owner: main agent
- Source: B2, B3
- Priority: high

## Round Summaries
### Round 1
- Status: completed
- Goal: Multi-agent qualitative pass over transition and reversal windows.
- Open Questions:
  - Does clipping-linked degeneration appear as format failure, reasoning collapse, or exploration collapse?
  - Which failure mode explains catch-up most directly: wrong math, answer extraction error, or overlong penalty interaction?

### Round 2
- Status: completed
- Goal: Late-window reinforcement + taxonomy + H1 critique/alternatives.
- Findings:
  - Clipping/truncation explains part of the transition shock but not the whole reversal.
  - Most damaging late errors are combinatorial bookkeeping and structural misread classes.
  - H1 is supported but must be refined into H1' with explicit competing mechanisms (H2/H3).
