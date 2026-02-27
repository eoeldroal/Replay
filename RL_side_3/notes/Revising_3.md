# Revising_3: Replay 압력 완화 + 품질 우선 재설계

본 문서는 `Revising_2`의 후속 분석/개선안이다. 충돌 시 최신 문서(`Revising_3`)를 우선한다.

## 0) 목적

현재 rev2에서 관측된 핵심 이슈는 다음 두 가지다.

1. replay가 학습 시간(`timing_s/update_actor`)을 크게 늘린다.
2. 평균 재사용 횟수는 높지 않지만, 일부 query가 비정상적으로 여러 번 재사용되는 tail이 존재한다.

이번 revising의 목적은 **replay 양을 줄이더라도 품질이 높은 replay만 쓰는 방향**으로 전환하는 것이다.

---

## 1) 로그 기반 분석 요약

### 1.1 재사용 통계의 모순처럼 보이는 현상

- 집계 기준 `used_avg_repeat_prev`는 대체로 1 미만~1.x 수준이지만,
- 일부 query는 재사용 횟수 5~7까지 올라간다.

이는 모순이 아니다. 이유는 다음과 같다.

- 평균은 256개 사용 집합의 평균값이라, 다수(0~1회) + 소수(고반복) 구조면 평균이 낮게 유지된다.
- 동시에 매 step 고정 replay 목표가 크면(기존 256), 소수 샘플이 반복 선택되는 tail이 생길 수 있다.

### 1.2 현재 구조에서 tail이 생기는 구조적 이유

- replay 수요가 큼: step당 replay 목표가 높을수록 이전 샘플 재사용이 강제된다.
- recency-only 동률 구간: `last_training_step`이 같은 후보가 많아지는 구간에서 변별력이 약해진다.
- M2 게이트 통과율: 통과율이 높은 구간에서는 정렬 앞쪽 샘플이 그대로 반복 사용되기 쉽다.

### 1.3 저장 실패 이슈(운영 이슈)

- 저장 실패의 직접 원인은 replay 알고리즘이 아니라 **상대경로 checkpoint + 작은 로컬 디스크(100%)** 조합이었다.
- 즉, 실행 경로/저장 경로 관리 이슈이며, 알고리즘 품질 이슈와 분리해서 봐야 한다.

---

## 2) 개선 방향 (이번 합의)

### A. M2 게이트 강화

- `tau: 0.005 -> 0.001`
- 목표: replay 양 증가가 아니라 replay 품질 강화.
- 기대효과: 낮은 품질/노이즈 replay 필터링 강화, tail 완화 보조.
- 리스크: 초반 step 또는 어려운 구간에서 `accepted_groups` 급감 가능.

### B. 버퍼 크기 축소

- `max_query_groups: 1280 -> 640`
- 목표: 오래된 샘플 누적 폭을 줄여 계산량/추적 복잡도/logprob 비용을 낮춤.
- 기대효과: update 시간 완화, 운영 안정성 개선.
- 리스크: 지나친 축소 시 다양성 저하(너무 최신 쪽으로 쏠릴 위험).

### C. replay 목표량 축소

- `replay_target_groups: 256 -> 128`
- 목표: on-policy(256) 대비 replay 비율을 1.0 -> 0.5로 낮춰 학습 압력 균형화.
- 기대효과:
  - `timing_s/update_actor` 감소
  - 과도한 off-policy 영향 완화
  - tail 재사용 빈도 감소
- 리스크: replay 기여 약화 가능.

---

## 3) 왜 이 조합이 타당한가

세 파라미터는 서로 같은 방향으로 작동한다.

- `tau` 하향: 통과 질을 올림(quality gate 강화)
- `buffer` 축소: 탐색 풀 축소로 계산 비용 감소
- `target` 축소: 실제 replay 사용량 감소

즉, 이번 조합은 **품질 우선 + 비용 절감** 방향으로 일관적이다.

---

## 3.1) 계산 순서/정합성 원칙 (추가 합의)

최근 논의에서 다음을 명확히 고정했다.

- 학습 순서는 유지한다: `old_log_prob -> advantage -> replay 결합 -> update_actor`.
- 이유: 현재 설정(`rollout_correction.bypass_mode=False`)에서는 `old_log_prob` 재계산 경로가 기준이며, 이를 바꾸면 다른 실행 코드/실험과 정합성이 크게 흔들릴 수 있다.
- 특히 `advantage -> old_log_prob`로 순서를 뒤집는 변경은 단순 최적화가 아니라 파이프라인 의미 자체를 바꾸는 큰 변경으로 본다.

따라서 효율 개선은 다음처럼 분리한다.

- **유지되는 비용**: 온폴리시 전체 `old_log_prob` 계산 비용.
- **줄일 비용**: `update_actor`의 loss/backprop 비용(예: ADV=0 샘플 제거 시 이 구간 감소 기대).

운영 관점에서는 `timing_s/old_log_prob`보다 `timing_s/update_actor`를 주요 개선 지표로 본다.

---

## 4) 관측 지표 (반드시 함께 확인)

아래 지표를 step 단위로 함께 본다.

- 품질/량
  - `m2_replay/pass1/accepted_groups`
  - `m2_replay/pass1/acceptance_rate`
  - `m2_replay/pass2/used_groups`
- 성능
  - `reward/acc_mean`
- 효율
  - `timing_s/update_actor`
  - `timing_s/step`
  - `perf/throughput`
- 재사용 편향
  - `top_replayed_query_ids`
  - full dump 기반 per-step 재사용 평균/최대

---

## 5) 판정 기준 (초기)

### 성공 신호

- `timing_s/update_actor` 유의미한 감소
- `used_groups`가 너무 낮지 않으면서(완전 고갈 아님) `reward/acc_mean` 유지 또는 개선
- 고반복 tail(특정 query 과다 재사용) 완화

### 실패 신호

- `accepted_groups`가 장기간 매우 낮아 replay가 사실상 비활성
- 성능 하락 + 효율 개선도 미미

---

## 6) 다음 조정 우선순위

1. 먼저 이번 조합(`tau=0.001, buffer=640, replay_target=128`) 고정 실험
2. `accepted_groups` 과소 시 `tau`만 소폭 완화(`0.001 -> 0.002`)
3. 그래도 tail이 심하면 sampling tie-break 추가 개선(추후)
