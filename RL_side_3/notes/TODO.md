# TODO (Current)

기준일: 2026-02-27  
SSOT 실행 설정: `RL_side_3/grpo-qwen25-math-1.5b-s8-m2-replay-rev2.sh`

## 현재 고정 기준

- `selection.mode=recency_only`
- `selection.ingress_filter_mode=rlvr_halfband`
- `tau=0.001`
- `replay_target_groups=128`
- `buffer.max_query_groups=1024`

## 실행 TODO

- [ ] rev2 런 장기 추적: `pass1/acceptance_rate`, `pass2/used_groups`, `reward/acc_mean`, `timing_s/update_actor`
- [ ] 버퍼가 충분히 찬 이후(steady-state) 재사용 tail(`top_replayed_query_ids`) 재점검
- [ ] `accepted_groups << replay_target_groups`가 지속되면 `tau` 완화 실험(`0.001 -> 0.002`)
- [ ] `accepted_groups`가 과도하게 높고 재사용 tail이 심하면 `buffer`/`target` 추가 축소안 비교
- [ ] m2 replay 통합 테스트 보강 (`ray_trainer + adapter + core` 경로)
- [ ] checkpoint resume에서 `m2_replay_state.pt` 복원 회귀 테스트 추가

## 문서 TODO

- [ ] `FACTS_AND_DECISIONS.md` 본문의 과거 기준(`tau=0.04`, `|p-0.5|`)을 “히스토리” 섹션으로 명시 분리
- [ ] `Revising_2.md`의 구버전 수치(예: `tau=0.005`, `buffer=2048`)에 "historical" 라벨 추가
