# TODO (Current)

기준일: 2026-03-06

안정 축 SSOT: `RL_side_3/grpo-qwen25-math-1.5b-s8-m2-replay-rev2.sh`

확장 축 대표 실행:

- `RL_side_3/grpo-qwen25-math-1.5b-s8-m2-replay-rev5.sh`
- `RL_side_3/grpo-qwen3-1.7b-base-s8-m2-replay-rev6.sh`
- `RL_side_3/grpo-qwen3-1.7b-s8-m2-replay-rev7.sh`

## 현재 고정 기준

- `selection.mode=recency_only`
- `selection.ingress_filter_mode=rlvr_halfband`
- `tau=0.001`
- `replay_target_groups=128`
- `buffer.max_query_groups=1024`

## 현재 확장 구현 기준

- `selection.mode=zvp_recency`
- sign-only 대칭 ZVP 점수
- `selection.logprob_groups_per_chunk`
- `selection.log_prob_micro_batch_size_per_gpu`
- `schedule.one_turnover_gate`

## 실행 TODO

- [ ] 안정 축(rev2) 장기 추적: `pass1/acceptance_rate`, `pass2/used_groups`, `reward/acc_mean`, `timing_s/update_actor`
- [ ] 확장 축(rev6/rev7) 추적: `timing_s/m2_select_logprob_eval`, `selection/replay_used`, `selection/used_zvp_mean`, `gating/one_turnover_ready`
- [ ] 버퍼 steady-state 이후 재사용 tail(`top_replayed_query_ids`) 재점검
- [ ] `accepted_groups << replay_target_groups`가 지속되면 `tau` 완화 실험(`0.001 -> 0.002`)
- [ ] ZVP 축에서 `buffer=1024 vs 512`, `tau=0.001 vs 0.002` 비교
- [ ] `replay vs on-policy` 분리 actor 메트릭(`pg_clipfrac`, `ppo_kl`, `grad 기여`) 추가
- [ ] m2 replay 테스트 재실행 및 현재 워크트리 기준 pass 여부 재확인
- [ ] checkpoint resume에서 `m2_replay_state.pt` 복원 회귀 테스트 추가

## 문서 TODO

- [ ] `FACTS_AND_DECISIONS.md`에서 안정 축 SSOT와 확장 축 frontier 구분 유지
- [ ] `ANALYSIS_POINTS.md`에서 “이미 구현된 로그”와 “추가 구현이 필요한 로그”를 분리 유지
- [ ] `Revising_8.md`의 실행 설정/검증 기록을 현재 워크트리 기준으로 재확인
