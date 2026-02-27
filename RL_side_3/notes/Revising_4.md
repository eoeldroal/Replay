# Revising_4: ADV=0 제거 실험 설계 (변인 통제 버전)

본 문서는 `Revising_3` 후속 논의 정리본이다. 이번 목적은 **속도 개선**이며, 성능 해석이 가능하도록 변인을 최소화한다.

## 0) 실험 목적

- replay의 강점(추가 generation 없음)은 유지하되, actor update 계산 비용을 줄인다.
- 핵심 가설: `ADV=0` 샘플 제거 시 학습 신호 밀도가 올라가고 `update_actor` 시간이 줄어든다.

## 1) 이번 변경/비변경 범위

### 변경 (1개)

- `ADV=0` query-group을 actor 학습 배치에서 제외한다.
- GRPO 기준으로는 그룹 내 보상이 전부 같아 advantage가 0이 되는 경우(대표적으로 all-correct / all-wrong)가 대상이다.

### 고정 (해석용 통제)

- `loss_agg_mode=token-mean` 유지
- legacy 경로 유지(현재 micro-batch + grad accumulation 구조 유지)
- 마지막 찌꺼기(mini-batch remainder) 허용 유지
- replay 로직의 기존 rev2 방향(ingress filter, recency-only, M2 gate)은 유지

즉, 이번 실험에서 바뀌는 핵심은 `ADV=0 제거` 하나다.

## 2) 학습 역학 영향 (핵심)

- 방향 자체는 기존 PPO/GRPO와 동일하다.
- 다만 `token-mean`에서 `ADV=0` 샘플을 제거하면 분모 희석이 줄어 **유효 gradient scale이 커진다**.
- 실질적으로는 step별 유효 학습률이 올라가는 효과가 날 수 있다.

참고:
- 이 변화는 의도된 변화이며, 이번 실험의 관찰 대상이다.
- 찌꺼기 mini-batch 영향은 기존과 동일하게 남는다(이번 변경으로 새로 생기는 이슈는 아님).

## 3) 효율 관점 기대치

- `update_actor` 시간 감소 가능성: 높음
- `old_log_prob` 시간 감소 가능성: 낮음
  - 현재 학습 순서상 old_log_prob 계산 이후에 제거하면, 절감은 주로 actor update 구간에서 발생한다.

## 4) 리스크

- `keep_ratio`(유지된 그룹 비율) 변동이 크면 step별 gradient 변동성 증가 가능
- 초반 구간에서 `grad_norm`, `ppo_kl`, `clipfrac` 진동 가능
- 찌꺼기 mini-batch는 그대로 존재하므로 완전한 스케일 안정화는 아님

## 5) 필수 로깅

- `adv0/dropped_groups`
- `adv0/kept_groups`
- `adv0/kept_ratio`
- `adv0/effective_scale` (예: `1/max(kept_ratio, eps)`)
- 기존 지표 동시 추적:
  - `timing_s/update_actor`
  - `timing_s/step`
  - `perf/throughput`
  - `actor/grad_norm`
  - `actor/ppo_kl`
  - `actor/pg_clipfrac`

## 6) 판정 기준

### 성공 신호

- `update_actor` 유의미 감소
- 성능(`reward/acc_mean`) 유지 또는 개선
- 학습 안정성 지표(`grad_norm`, `ppo_kl`)가 허용 범위 내

### 실패 신호

- `kept_ratio` 과도한 변동 + 안정성 지표 급진동
- 속도 이득은 작고 성능만 하락

## 7) 결론

- 이번 단계는 합리적이며, 변인 통제 측면에서 해석 가능성이 높다.
- 다음 의사결정은 `ADV=0 제거` 단일 변경의 실측 결과를 기반으로 진행한다.
