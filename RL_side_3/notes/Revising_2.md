# Revising_2: 로그 분석 결과 + 개선 방향

본 문서는 최신 합의 반영본이며, 이전 문서와 충돌 시 본 문서를 우선한다.

## 확정 개선안 (1/2/3)

1. 버퍼 적재 규칙 변경
- 쿼리당 생성 수를 `G`, 정답(성공) 개수를 `c`라 할 때 `1 <= c <= floor(G/2)`인 query-group만 버퍼에 적재.
- `c=0` 또는 `c=G`인 샘플은 버퍼에 넣지 않음.

2. 버퍼 샘플링 단순화
- `|p-0.5|` 기반 우선순위는 제거.
- “최근 학습에 사용되었는지(최근성)”만 기준으로 샘플링.
- 최근 사용 샘플은 뒤로 미루고, 오래 사용되지 않은 샘플을 우선 사용.

3. M2 게이팅 강화
- `tau`를 기존 `0.02`보다 낮춰 더 엄격하게 선별.
- 현재 합의안: `tau=0.005`.
- 목표는 replay 양 증가가 아니라 replay 품질 강화.

## 1) 로그 분석 결과 (핵심)

- 속도 비용이 매우 큼
  - replay 적용 시 `timing_s/update_actor`가 baseline 대비 크게 증가(대략 3배 수준 구간 다수 관찰).
  - `timing_s/step` 증가, `perf/throughput` 감소가 동반됨.
- 선택 게이팅이 약함
  - `m2_replay/pass1/accepted_groups`가 대부분 step에서 `256/256`에 가까움.
  - 즉, M2 게이팅이 실질적으로는 대량 통과에 가까웠음.
- 재사용 집중 현상
  - `top_replayed_query_ids` 카운트가 빠르게 증가(버퍼 1024, step당 replay 256 사용 구조 영향).
  - 특정 query가 반복적으로 재선택되는 패턴이 명확함.
- 점수 동질화 문제
  - `m2_replay_scores` 로그에서 상위 후보의 `success_prob=0.5`, `score` 동일값 반복이 관찰됨.
  - 우선순위가 의미 있게 분산되지 못하고, 유사 샘플 군집 반복으로 흐를 위험이 큼.

## 2) 문제 원인 정리

- 기존 우선순위(`|p-0.5|` 축 + 최근학습 항)가 실제로는 샘플 다양성을 충분히 확보하지 못함.
- replay 비율(온폴리시와 동일 규모) 대비 게이팅 강도가 약해, 비용은 크고 선별 이득은 제한적.
- 결과적으로 “가치 있는 재사용”보다 “최근/유사 샘플 반복 학습” 성격이 강해짐.

## 3) 개선 방향 (합의안)

### A. 버퍼 적재 규칙 변경

- 기존의 `0.5 근처 우선` 논리를 버퍼 선택 점수에서 제거.
- 버퍼 **적재 시점**에 query-group 필터링:
  - 쿼리당 생성 수를 `G`, 정답(성공) 개수를 `c`라 할 때,
  - `1 <= c <= floor(G/2)` 인 query-group만 버퍼에 적재.
  - 목적: 너무 쉬운 샘플(`c=G`)과 너무 어려운 샘플(`c=0`) 배제.

### B. 버퍼 샘플링 규칙 단순화

- 샘플링 우선순위는 “최근 학습 사용 여부(= recency)”만 사용.
- 즉, 최근에 학습에 쓰인 샘플일수록 뒤로 미루고, 오래 안 쓰인 샘플을 먼저 사용.
- 기존 `uncertainty(|p-0.5|)` 기반 우선순위 항은 제거.

### C. 게이팅 강화

- `tau`를 기존 `0.02`보다 더 보수적으로 하향.
- 현재 논의 우선안: `tau = 0.005`.
- 목표: replay 양보다 질(엄격 선별).

### D. 샘플링 점수 하이퍼파라미터 축소

- recency-only 정렬이라면 점수 스케일 하이퍼파라미터(`beta`, `lambda`)는 제거 가능.
- 구현은 단순히 `last_training_step` 기준 정렬 + 기존 tie-break 유지(`seed`, `insertion_order`).

## 4) 실험 관찰 포인트

- `m2_replay/pass1/acceptance_rate`
- `m2_replay/pass2/used_groups`
- `reward/acc_mean`
- `timing_s/update_actor`, `timing_s/step`, `perf/throughput`

해석 기준:
- `acceptance_rate`가 여전히 높으면 `tau` 추가 하향 검토.
- `used_groups`가 낮아져도 성능/효율이 개선되면 현재 목표(quality-first)와 정합적.

## 5) 구현 반영 사항 (추가)

- `replay_target_groups`를 독립 제어 키로 분리:
  - replay 선택 목표 수는 더 이상 `onpolicy_group_count`에 고정되지 않음.
  - 실행 스크립트에서 `+algorithm.m2_replay.schedule.replay_target_groups=256`으로 제어.
- 온폴리시 학습 수와 ingress(버퍼 적재) 수를 분리:
  - `onpolicy_train_groups`: 실제 actor 학습에 들어가는 온폴리시 그룹 수.
  - `onpolicy_ingress_groups`: `1 <= c <= floor(G/2)` 필터를 통과해 버퍼에 들어간 그룹 수.
  - 필터는 replay ingress 전용이며, 온폴리시 학습 배치를 직접 줄이지 않음.
- 로그/덤프 보강:
  - 콘솔 로그에 `onpolicy_groups`, `onpolicy_ingress_groups`, `replay_target` 추가.
  - full score dump에 `onpolicy_ingress_query_ids`, `replay_target_groups` 저장.
- 버퍼 크기 상향:
  - rev2 실행 스크립트 기준 `max_query_groups=2048`로 확장.
