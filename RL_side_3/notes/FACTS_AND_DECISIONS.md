# M2-Replay: Facts and Decisions

업데이트: 2026-02-27

## 1) Current Canonical (SSOT)

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

## 2) Stable Implementation Facts

1. same-step 혼입 방지: `select -> append` 순서 고정
2. replay off 호환성: `algorithm.m2_replay.enable=false`면 기존 경로 유지
3. replay 목표 수 독립 제어: `schedule.replay_target_groups`
4. 실제 사용 수는 항상 `used <= selected <= target` (tau, 배수 보정 영향)
5. 배치 결합 순서: `replay + onpolicy` (replay-first)
6. 체크포인트에 `m2_replay_state.pt` 저장/복원
7. 동적 배치(`actor.use_dynamic_bsz=true`)와 replay 동시 사용은 미지원

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

### Phase C: Revising_3 -> 현재 실행값 수렴

- 방향: replay 압력 완화 + 품질 우선
- 현재 적용값: `tau=0.001`, `replay_target=128`, `buffer=1024`
- 목적: 비용(시간)과 과재사용 tail을 동시에 완화

## 5) Reading Order

1. 본 문서의 `1) Current Canonical`
2. `notes/Revising_3.md` (최신 분석/의도)
3. `notes/Revising_2.md`, `notes/Revising_1.md` (히스토리)
