# Replay Analysis Points

## 상태 정리

- 본 문서 상단의 `step 1~6` 관찰은 초기 `tau=0.04`, full-use 구간의 **히스토리 로그**다.
- 현재 안정 축 SSOT는 `rev2`(`recency_only + tau=0.001 + target=128`)다.
- 현재 확장 축은 `rev5~rev7`(`zvp_recency + batched M2 eval + one_turnover_gate`)다.
- 아래 가설 U1/U2/U3는 **확장 축 검증 항목**으로 읽는다.

## 현재 구현 상태

- 구현됨:
  - `selection.mode=zvp_recency`
  - sign-only 대칭 ZVP 점수
  - chunked M2 후보 평가
  - `selection.log_prob_micro_batch_size_per_gpu`
  - `schedule.one_turnover_gate`
- 아직 직접 로깅 안 됨:
  - replay vs on-policy 분리 `pg_clipfrac`
  - replay vs on-policy 분리 `ppo_kl`
  - replay vs on-policy 분리 `grad 기여`

## 현재 관찰 (step 1~6)

- step1은 replay 0, step2+는 replay 256 full-use
- `tau=0.04`에서 pass1 acceptance가 거의 1.0
- `update_actor`/`step` 시간이 baseline 대비 크게 증가
- 누적 재선택 상위 query_id가 빠르게 증가하며, 현재 설정에서는 max count가 4에서 상한
- 사용자 관찰상 초기 구간에서는 baseline 대비 성능 우세 신호가 있으나, 통계적 결론은 보류

## 최신 수치 스냅샷 (step 1 vs step 2~6)

- `timing_s/update_actor`: 약 `44.6s` -> `103~112s`
- `timing_s/step`: 약 `114s` -> `200~214s`
- `m2_replay/pass1/accepted_groups`: 거의 매 step `256`
- `m2_replay/pass1/acceptance_rate`: 거의 `1.0`

## 분석 목표

- 재사용 제한 없이도 M2 게이팅이 replay 재선택을 자연스럽게 줄이는지 확인

## 핵심 질문

1. 한 번 선택된 QueryGroup은 다음 step에서 `M2`가 올라가 재선택이 줄어드는가?
2. `tau`가 작을수록 이 효과가 강해지는가?
3. 재선택 감소가 M2 때문인지 FIFO/배치 제약 때문인지 분리 가능한가?

## 최소 지표

- step별 `acceptance_rate`, `accepted_groups`, `scanned_groups`
- QueryGroup별 선택 횟수(재사용 횟수) 분포
- 재사용 횟수별 `M2` 평균/분포
- step별 unique replay ratio (`unique_selected / total_selected`)

## 최소 비교 실험

1. `tau` 값 변화(예: 0.02 / 0.04 / 0.08)
2. replay on vs replay off
3. buffer 크기 변화( FIFO 영향 분리 )
4. replay 비율 축소(예: target 50%) vs full-use(100%)

## 현재 전제

- 재사용 제한(`max_reuse/cooldown/age_limit`) 없음
- 선택된 replay는 버퍼에서 제거하지 않음(FIFO로만 축출)
- same-step 혼입은 `select -> append`로 방지

## 사용자 직접 가설 (ZVP 우선순위, 향후 정밀 검증용)

- 아래 항목은 "모델의 일반 이론"이 아니라 **사용자 본인 가설**을 그대로 추적하기 위한 체크포인트
- 분석 시 해석을 덧붙이기보다, 먼저 아래 가설의 성립/불성립을 분리해서 검증

### 가설 U1 (clip 증가 가설)

- `low logprob(=high entropy)`이면서 `positive advantage`인 replay 샘플을 우선 선택하면,
  replay 구간에서 PPO ratio clipping(`actor/pg_clipfrac`)이 유의하게 증가한다.

### 가설 U2 (replay-first 영향 증폭 가설)

- 위와 같은 샘플이 replay-first 구간에 많이 들어오면, 해당 구간의 정책 변화가 커져
  같은 step의 on-policy 구간까지 영향을 주며 `actor/grad_norm` 및 `actor/ppo_kl` 변동성이 증가한다.

### 가설 U3 (자연 감쇠 가설)

- 한 번 replay로 학습된 샘플은 엔트로피가 내려가므로, 다음 step들에서 ZVP 점수가 자연히 낮아져
  재선택 확률이 하락한다(자기증폭이 아니라 자기감쇠).

## 사용자 가설 검증용 최소 로그 항목

- 추가 구현 필요:
  - source 분리 지표: replay vs on-policy 각각의 `pg_clipfrac`, `ppo_kl`, `grad 기여`
- 현재 코드에서 직접 수집 가능:
  - `m2_replay/pass1/acceptance_rate`, `accepted_groups`, `scanned_groups`
  - `m2_replay/selection/used_zvp_mean`, `used_zvp_surprisal_mean`, `used_zvp_neg_frac_mean`, `used_zvp_pos_frac_mean`
  - `m2_replay/selection/log_prob_micro_batch_size_per_gpu`
  - `m2_replay/gating/one_turnover_ready`
- full score dump / 후처리로 추적 가능:
  - replay 후보/사용 집합의 엔트로피 분포와 step 간 변화
  - replay 샘플별 `선택 직전 entropy` vs `선택 후 entropy` 변화량
  - 샘플 단위 재사용 곡선: `(selection_count, entropy, zvp_score)` 시계열

## 사용자 원문 취지 보강 (해석 금지)

- 아래 두 우려는 사용자 원문 그대로 후속 분석에서 직접 검증:
  1. `low logprob/high entropy + positive advantage` replay 선호 시, PPO ratio clipping이 생각보다 많이 발생할 수 있음.
  2. 같은 조건의 replay를 replay-first로 먼저 학습하면 정책 변화량이 커져, 같은 step의 on-policy 구간에도 연쇄 영향이 갈 수 있음.
- 위 두 항목은 결론을 미리 두지 않고, 실측 로그로만 판정.
