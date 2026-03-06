# M2-Replay: Facts and Decisions

업데이트: 2026-03-06

## 1) Stable Canonical (SSOT)

기준 실행 파일: `RL_side_3/grpo-qwen25-math-1.5b-s8-m2-replay-rev2.sh`

- 알고리즘: GRPO + M2 replay
- selection mode: `recency_only`
- ingress filter: `rlvr_halfband` (`1 <= success_count <= floor(G/2)`만 버퍼 적재)
- `tau=0.001`
- `replay_target_groups=128`
- `buffer.max_query_groups=1024`
- on-policy 학습 그룹 수와 ingress 그룹 수는 분리:
  - `onpolicy_train_groups`: 실제 actor update에 들어가는 온폴리시 그룹 수
  - `onpolicy_ingress_groups`: 버퍼 적재 필터를 통과한 그룹 수

## 1-2) Active Frontier (현재 작업축)

대표 실행 파일:

- `RL_side_3/grpo-qwen25-math-1.5b-s8-m2-replay-rev5.sh`
- `RL_side_3/grpo-qwen3-1.7b-base-s8-m2-replay-rev6.sh`
- `RL_side_3/grpo-qwen3-1.7b-s8-m2-replay-rev7.sh`

현재 확장축의 핵심:

- `selection.mode=zvp_recency`
- sign-only 대칭 ZVP 점수(`A>0 -> 1-p`, `A<0 -> p`)
- `selection.logprob_groups_per_chunk` 기반 batched M2 후보 평가
- `selection.log_prob_micro_batch_size_per_gpu` 기반 M2 전용 micro-batch override
- `schedule.one_turnover_gate` 기반 초기 replay 지연

주의:

- 본 문서의 SSOT는 여전히 `rev2` 안정 축이다.
- `rev5~rev7`은 코드 반영이 끝난 확장축이지만, 아직 분석/문서 SSOT를 완전히 대체한 상태는 아니다.

## 2) Stable Implementation Facts

1. same-step 혼입 방지: `select -> append` 순서 고정
2. replay off 호환성: `algorithm.m2_replay.enable=false`면 기존 경로 유지
3. replay 목표 수 독립 제어: `schedule.replay_target_groups`
4. 실제 사용 수는 항상 `used <= selected <= target` (tau, 배수 보정 영향)
5. 배치 결합 순서: `replay + onpolicy` (replay-first)
6. 체크포인트에 `m2_replay_state.pt` 저장/복원
7. 동적 배치(`actor.use_dynamic_bsz=true`)와 replay 동시 사용은 미지원
8. `selection.mode=zvp_recency`는 opt-in으로 구현되어 있으며 기본 모드는 아님
9. M2 후보 평가 경로는 chunked batch eval과 per-group fallback을 모두 지원
10. worker는 `log_prob_micro_batch_size_override`와 `calculate_entropy` meta flag를 읽는다
11. `schedule.one_turnover_gate`는 구현되어 있으나 기본값은 off다
12. `fixed_total_floor_groups`는 opt-in이며 `fixed_total_with_adv0_drop`에서만 의미가 있다
13. `replay vs on-policy` 분리 actor 메트릭(`pg_clipfrac`, `ppo_kl`, `grad 기여`)은 아직 직접 로깅되지 않는다

핵심 코드 위치:
- `verl/trainer/ppo/ray_trainer.py` (`_m2_build_actor_batch`)
- `verl/trainer/ppo/m2_replay.py` (선택/게이팅)
- `verl/trainer/ppo/m2_replay_adapter.py` (그룹 변환/배치 결합)

## 3) Operations Note

- 저장 실패(`No space left on device`)는 알고리즘 문제가 아니라 디스크/경로 운영 이슈였다.
- 원인: 상대경로 실행 시 작은 로컬 디스크(`/home/work`)에 체크포인트가 쌓인 케이스.
- 대응: 실행 경로/체크포인트 경로를 고정 관리하고 잔여 용량을 사전 확인.

## 4) History (Preserved)

### Phase A: 초기 실험

- 설정: `tau=0.04`, `|p-0.5|` 중심 우선순위, replay 목표를 온폴리시와 1:1로 운용
- 관측: pass1 acceptance가 높아 replay full-use 구간이 자주 발생, `update_actor` 시간 증가

### Phase B: Revising_2

- 변경: `recency_only` + `rlvr_halfband` 도입, replay 목표를 on-policy와 분리
- 의도: “가치 있는 replay” 중심으로 전환, 최신 샘플 과재사용 완화
- 당시 문서에는 `tau=0.005` 등 중간값이 기록되어 있음(히스토리 값)

### Phase C: Revising_3 -> 안정 축 수렴

- 방향: replay 압력 완화 + 품질 우선
- 현재 적용값: `tau=0.001`, `replay_target=128`, `buffer=1024`
- 목적: 비용(시간)과 과재사용 tail을 동시에 완화

### Phase D: Revising_6 / Revising_8 -> 확장 축

- 방향: recency-only에서 replay 가치 기반(`zvp_recency`) 우선순위로 확장
- 추가 구현: batched M2 eval, M2 전용 micro-batch override, one-turnover gate
- 모델 확장: Qwen3-1.7B-Base -> Qwen3-1.7B(Non-Base)
- 상태: 코드 반영 완료, 분석용 메트릭은 일부 추가 구현 필요

## 5) Reading Order

1. 본 문서의 `1) Stable Canonical`
2. `notes/Revising_8.md`, `notes/Revising_6.md` (현재 확장 축)
3. `notes/Revising_5.md`, `notes/Revising_3.md` (1.5B 실험 분석)
4. `notes/Revising_2.md`, `notes/Revising_1.md` (히스토리)
