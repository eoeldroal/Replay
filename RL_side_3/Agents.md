# RL_side_3: M2-Gated Replay for GRPO — Knowledge Base

이 파일은 분석/실험에서 발견된 유용한 사실들을 누적 정리한다.

---

## 프로젝트 개요

- **목표**: KL-free GRPO에서 off-policy replay를 M2 게이팅으로 안전하게 재사용
- **안정 축 모델**: Qwen2.5-Math-1.5B
- **확장 축 모델**: Qwen3-1.7B-Base, Qwen3-1.7B
- **평가**: MATH-500 pass@16, AIME2024/2025
- **프레임워크**: verl (FSDP2, vLLM rollout, 4/8 GPU)

## 현재 상태 요약

- **Stable SSOT**: `rev2` 기준 `recency_only + rlvr_halfband + tau=0.001 + replay_target=128 + buffer=1024`
- **Active Frontier**: `rev5~rev8` 기준 `zvp_recency + batched M2 eval + one_turnover_gate + Qwen3`
- **문서 해석 원칙**:
  - `FACTS_AND_DECISIONS.md`는 안정 축 기준
  - `Revising_6.md`, `Revising_8.md`는 확장 축 기준
  - `Revising_9.md`는 Qwen3 failure diagnosis + rev8 방향 + split logging 반영본
  - 아래 성능 표는 주로 Qwen2.5-Math-1.5B 축 기록이며, Qwen3 축과 직접 혼용하지 않음

## 2026-03-06 최신 운용 메모

### 지금 실제로 중요한 실행 스크립트

- Qwen3 baseline: `RL_side_3/grpo-qwen3-1.7b-s8.sh`
- Qwen3 replay frontier:
  - `RL_side_3/grpo-qwen3-1.7b-s8-m2-replay-rev7.sh`
  - `RL_side_3/grpo-qwen3-1.7b-s8-m2-replay-rev8.sh`
- Qwen2.5-Math replay 비교 기준:
  - `RL_side_3/grpo-qwen25-math-1.5b-s8-m2-replay-rev2.sh`
  - `RL_side_3/grpo-qwen25-math-1.5b-s8-m2-replay-rev5.sh`

### Qwen3에서 현재 가장 중요한 진단

- `rev7`의 핵심 문제는 "replay가 조금 약하다"가 아니라, `replay_used=0`에 가까운 deadlock이다
- direct cause는 top-ranked replay candidate가 stale해서 `tau=0.001` M2 gate를 통과하지 못하는 것이다
- 이 문제는 `tau` 숫자 하나보다 아래 조합에서 발생한다
  - thinner halfband ingress
  - no replay refresh
  - long / clipped trajectory
  - sign-only ZVP의 `1/16` hard-negative 과선호

### Qwen2.5-Math와 Qwen3의 가장 중요한 차이

- Qwen2.5-Math에서는 top-ZVP hard negative가 **fresh**하다
- Qwen3 rev7에서는 top-ZVP hard negative가 **stale**하다
- 요약:
  - Qwen2.5-Math: `fresh, short, tau-safe hard negative`
  - Qwen3 rev7: `stale, long, clipped, tau-unsafe hard negative`

### `age` 해석

- `age`는 단순 "버퍼에 들어온 지 몇 step"이 아니라 `current_step - last_training_step`이다
- replay가 실제로 사용되면 `last_training_step`가 refresh된다
- 따라서 replay가 한 번도 안 돌면 top candidate age가 계속 누적된다
- FIFO와 모순되지 않는다
  - buffer size가 같아도 step당 ingress group이 적으면 더 긴 step history를 담게 된다

### 현재 codebase에서 바로 기억할 사실

- replay actor batch는 **replay-first**로 concat된다
- critic update는 현재 replay를 직접 보지 않는다. replay는 actor batch에만 들어간다
- 따라서 replay 분석의 핵심 clip metric은 `critic/vf_clipfrac`가 아니라 actor 쪽 split metric이다

## 최신 replay 옵션 지도

### ingress filter mode

- config key: `algorithm.m2_replay.selection.ingress_filter_mode`
- 지원 값:
  - `none`
  - `rlvr_halfband` : 기존 `1 <= success_count <= floor(n / 2)`
  - `rlvr_non_degenerate` : 새 `1 <= success_count <= n - 1`
- 기본값은 기존 동작 유지다

### ZVP mode

- config key: `algorithm.m2_replay.selection.zvp_mode`
- 지원 값:
  - `sign_only` : 기존 방식
  - `adv_magnitude` : `|adv|` self-normalized weighted ZVP
- 기본값은 기존 동작 유지다

### rev8의 의미

- `rev8`은 `rev7`에서 두 가지만 바꾼 버전이다
  - `ingress_filter_mode=rlvr_non_degenerate`
  - `zvp_mode=adv_magnitude`
- 목적은 final uplift 보장보다 먼저 replay activation deadlock을 깨는 것이다

## 최신 코드 위치 메모

- replay source flag 추가 위치:
  - `verl/trainer/ppo/m2_replay_adapter.py`
  - key: `m2_replay_source`
- source별 actor metric 계산:
  - `verl/trainer/ppo/core_algos.py`
  - helper: `_build_source_split_pg_metrics`
- actor worker 전달 경로:
  - `verl/workers/actor/dp_actor.py`
  - `verl/workers/actor/megatron_actor.py`
  - `verl/workers/utils/losses.py`

## W&B 해석 규칙

### sequence length clipping 관련

- `response_length/clip_ratio`는 exact truncation ratio가 아니라 proxy에 가깝다
- `response_length_non_aborted/clip_ratio`를 우선적으로 보는 편이 안전하다
- 작은 차이는 과해석하지 말고, 큰 차이만 strong signal로 해석한다

### PPO clipping 관련

- `actor/pg_clipfrac`와 `critic/vf_clipfrac`는 sequence truncation과 무관하다
- 둘은 PPO objective/value clipping 비율이다
- replay가 들어간 후에는 `actor/pg_clipfrac` 단일 값만으로는 해석이 부족하다

### 새로 추가된 replay/on-policy split actor 로그

- `actor/replay_pg_clipfrac`
- `actor/onpolicy_pg_clipfrac`
- `actor/replay_pg_clipfrac_lower`
- `actor/onpolicy_pg_clipfrac_lower`
- `actor/replay_ppo_kl`
- `actor/onpolicy_ppo_kl`
- `actor/replay_seq_frac`
- `actor/replay_token_frac`

해석 원칙:

- `replay_token_frac` 없이 replay clipfrac만 보지 말 것
- `replay_pg_clipfrac > onpolicy_pg_clipfrac`이면 replay 쪽이 더 aggressive / off-policy update 영역에 있을 가능성이 높다
- `replay_ppo_kl > onpolicy_ppo_kl`이면 replay 샘플이 현재 policy와 더 멀다는 뜻이다

## rev8 이후 체크리스트

1. `m2_replay/selection/replay_used > 0`가 안정적으로 유지되는가
2. top-ranked candidate `age`가 `rev7` 대비 낮아지는가
3. selected success bucket이 `1/16` 독점에서 벗어나는가
4. `actor/replay_token_frac`가 충분한가
5. `actor/replay_pg_clipfrac`가 `actor/onpolicy_pg_clipfrac`보다 과도하게 높지 않은가
6. `actor/replay_ppo_kl`가 `actor/onpolicy_ppo_kl`보다 구조적으로 너무 크지 않은가

## 테스트 상태

- replay option / adapter / core algo 관련 CPU 테스트는 `conda activate verl` 환경에서 통과 확인
- 현재 보강 이후 기준:
  - replay adapter source flag 테스트 포함
  - replay/on-policy split actor metric 테스트 포함

## 핵심 코드 위치

- `verl/trainer/ppo/m2_replay.py` — 버퍼, M2 계산, replay 선택
- `verl/trainer/ppo/m2_replay_adapter.py` — 그룹 변환, 배치 결합, ADV=0 분리
- `verl/trainer/ppo/ray_trainer.py` (`_m2_build_actor_batch`) — 통합 오케스트레이터

## 실험 설정 요약

| | Baseline | Rev2 | Rev3 | Rev4 |
|---|---|---|---|---|
| training_mode | on-policy only | legacy_bonus | fixed_total_with_adv0_drop | fixed_total_with_adv0_drop |
| replay target | - | 128 | fixed_total=256 | fixed_total=320, floor=256 |
| buffer | - | 1024 | 2048 | 1024 |
| tau | - | 0.001 | 0.001 | 0.001 |
| ADV=0 drop | No | No | Yes | Yes |
| GPU | 0-3 | 4-7 | 0-3 | 4-7 |

## 문서 읽기 순서

1. `FACTS_AND_DECISIONS.md` (안정 축 SSOT)
2. `Revising_8.md` → `Revising_6.md` (현재 확장 축)
3. `Revising_5.md` → `Revising_3.md` (1.5B 실험 변천)
4. `specification.md` (원래 설계)
5. `implementation.md` (파이프라인 상세)

---

## 확인된 사실 (Verified Facts)

### 성능 데이터

#### MATH-500 @16 (검증 스코어)
| Step | Baseline | Rev2 | Rev3 | Rev4 |
|------|----------|------|------|------|
| 50   | 0.4766   | 0.4856 | 0.4655 | 0.4758 |
| 100  | 0.4964   | 0.5007 | 0.4846 | 0.4954 |
| 150  | 0.5045   | 0.5083 | 0.4911 | 0.5078 |
| 200  | 0.5123   | 0.5131 | 0.5019 | 0.5086 |
| 250  | **0.5203** | 0.5162 | 0.5079 | 0.5107 |
| 300  | 0.5173   | **0.5228** | 0.5174 | --- |
| 350  | 0.5189   | **0.5235** | 0.5183 | --- |
| 400  | 0.5201   | 0.5182 | 0.5174 | --- |
| 450  | 0.5073↓  | 0.4856↓ | **0.5173** | --- |

- Baseline peak: 0.5203 @ step 250
- Rev2 peak: 0.5235 @ step 350 (최고)
- Rev3 peak: 0.5183 @ step 350
- Rev4: 0.5107 @ step 250 (아직 진행 중)

#### Baseline 붕괴 궤적 (step 500+)
- step 500: grad_norm=268, entropy=0.037, score=0.391
- step 680: grad_norm=1018, entropy=0.952
- step 695: grad_norm=6.5, entropy=1.196, score=0.253 (near-random)
- **3단계 붕괴**: policy concentration → gradient explosion → entropy diffusion

#### Replay 안정성 비교 (마지막 step)
- Rev2 @486: grad_norm=371, entropy=0.033 (준-붕괴)
- Rev3 @464: grad_norm=0.31, entropy=0.104 (매우 안정)
- Rev4 @268: grad_norm=0.06, entropy=0.116 (극도로 안정)

### M2 Replay 역학

#### Acceptance Rate 추이
- 모든 실험에서 step 300 이후 acceptance rate 급락
- Rev2 step 433: 1024개 전체 스캔, 0개 accept (완전 차단)
- Rev3 step 461: 2048개 스캔, 45개 accept (거의 차단)
- Rev4 step 268: 254/1024 스캔, 174 accept (아직 건강)

#### Accepted M2 Mean
- 초기~중기: ~0.0005 (tau=0.001의 50% 수준)
- 후기(step 400+): ~0.0007로 상승 → 통과 가능 샘플 소진 징후

### 효율성

#### update_actor 시간 (초, 평균)
- Baseline: ~52s / Rev2: ~110s (2.1x) / Rev3: ~67s (1.3x) / Rev4: ~87s (1.7x)

#### Wall-clock 추정 (step 250 도달)
- Baseline: 3.57h / Rev2: 7.58h / Rev3: 4.62h / Rev4: 5.95h

### Entropy 보존 효과
| Step | Baseline | Rev2 | Rev3 | Rev4 |
|------|----------|------|------|------|
| 250  | 0.085    | 0.125 | **0.165** | 0.130 |
| 450  | 0.027    | 0.040 | **0.121** | --- |

- Rev3가 가장 높은 entropy 유지 (baseline의 4.5배 @ step 450)
- entropy_coeff=0, use_kl_loss=False 조건에서도 replay만으로 entropy 보존

---

## 분석 인사이트

### Baseline 붕괴 메커니즘
1. KL-free GRPO에서 policy가 소수 패턴에 집중 (entropy 감소)
2. Importance ratio가 clip 범위를 누적적으로 초과
3. Gradient explosion → 불안정 진동 → entropy diffusion
4. PPO/GRPO 알려진 문제, 보통 KL penalty로 해결

### M2 Replay 안정화 메커니즘 (가설, 분리 검증 필요)
- **Implicit Behavior Regularization**: 과거 trajectory가 mode collapse 방지
- **Effective LR 감소**: replay가 on-policy gradient smoothing
- **자동 조절**: 정책 급변 → acceptance rate 하락 → 자연스러운 brake

### ADV=0 Drop 기여
- Gradient SNR 향상, policy concentration 촉진 학습 억제
- 실질 batch size 감소 → 일관된 update magnitude

### M2 게이팅 한계
- "유해 replay 선별"보다 "시간 기반 일괄 거부"에 가까움
- Step 300+ acceptance rate 급락, 대부분 buffer 거부
- IS 보정 없이 통과 샘플 동등 처리

---

## 변인 통제 문제점

Rev2 vs Rev3: 3개 변인 동시 변경 (training_mode, buffer, effective batch)
→ 인과 귀속 불가, 격리 실험 필수

필수 격리 실험:
- P1: Rev3 + buffer=1024 → buffer 효과 분리
- P2: ADV=0 drop만 (replay off) → ADV=0 drop 독립 효과
- P3: tau ablation {0.001, 0.005, 0.01}

---

## 논문 관점

### 핵심 발견
"entropy_coeff=0, use_kl_loss=False에서 M2-gated replay만으로 entropy 보존 + 붕괴 방지"

### 리뷰어 예상 요구
1. tau ablation / 2. replay off + ADV=0 drop only / 3. random replay 비교 / 4. 7B 재현

### 스토리라인 강도: 7/10

---

## 메모

- Rev4 완주 후 floor 발동 비율 확인 필요
- wandb API로 정밀 시계열 분석 가능 (로컬 로그는 50 step 간격)
- recency_only: last_training_step 오름차순 = "오래된 샘플 우선" (코드 확인)
- same-step 혼입 방지: select → append 순서 고정 (코드 확인)

# M2-Gated Replay 분석 라운드 1 종합 (2026-03-01)

5명의 연구 전문가 에이전트가 독립적으로 분석한 결과를 종합한다.
다음 라운드에서는 에이전트 팀 토론 방식으로 심화 분석 예정.

---

## 실험 현황 요약

| | Baseline | Rev2 | Rev3 | Rev4 |
|---|---|---|---|---|
| Steps 완료 | 695 | 486 | 464 | 268 |
| training_mode | on-policy only | legacy_bonus | fixed_total_with_adv0_drop | fixed_total_with_adv0_drop |
| replay target | - | 128 | fixed_total=256 | fixed_total=320, floor=256 |
| buffer | - | 1024 | 2048 | 1024 |
| tau | - | 0.001 | 0.001 | 0.001 |
| ADV=0 drop | No | No | Yes | Yes |

---

## 1. 핵심 발견: 5명 공통 합의

### 1-1. Baseline은 step ~500에서 붕괴한다 (강한 합의)

- step 500: grad_norm 268 → step 680: 1017 → entropy 1.0+ → score 0.25
- 원인: KL-free GRPO에서 policy concentration → gradient explosion → entropy diffusion
- 3단계 붕괴 시퀀스: 정책 집중 → off-policy gap → clip 우회 → 진동/확산

### 1-2. M2 Replay + ADV=0 drop은 붕괴를 방지한다 (강한 합의)

- Rev3 step 464: grad_norm 0.31, entropy 0.104 (매우 안정)
- Rev4 step 268: grad_norm 0.06, entropy 0.116 (극도로 안정)
- M2 replay 단독(Rev2)은 불충분: step 433에서 grad_norm spike 발생

### 1-3. Rev2 vs Rev3 성능 역설은 단일 seed 노이즈일 가능성 (중간 합의)

- Rev2 peak 0.5235 vs Rev3 peak 0.5183 (Δ=0.005)
- 그러나 Rev2는 step 433에서 불안정해지고, Rev3는 step 464까지 안정
- step 450 기준: Rev3 0.5173 > Baseline 0.5073 > Rev2 0.4856
- 단일 seed 실험이므로 통계적 유의성 미확보

---

## 2. 핵심 쟁점: 전문가 간 관점 차이

### 2-1. "replay가 entropy를 보존하는 메커니즘"

| 전문가 | 주장 |
|--------|------|
| 성능분석가 | "Distributional Buffering" - 과거 유효 샘플이 안정적 gradient 제공 |
| 역학분석가 | "과거 분포의 앵커링 + ADV=0 제거의 신호 밀도 효과" |
| 안정성분석가 | "Implicit Behavior Regularization + Effective LR 감소" |
| 비교전략가 | "replay가 entropy regularizer의 implicit 역할 수행" |

→ 모두 방향은 같으나, 어떤 메커니즘이 지배적인지에 대해서는 실험적 분리가 필요

### 2-2. "tau=0.001의 적절성"

- 역학분석가: "너무 엄격. 긴 응답에 불리. tau=0.003~0.005 또는 길이 정규화 필요"
- 비교전략가: "매우 엄격하지만 acceptance rate 78~80%로 과도하게 restrictive하지 않음"
- accepted_m2_mean ≈ 0.0005 (tau의 50% 수준) → 여유 있어 보이지만 step 300+ 급락

### 2-3. "M2 게이팅이 실제 유해 replay를 선별 차단하는가?"

- 역학분석가: "선별적 차단보다는 시간 기반 일괄 거부에 가까움"
- 안정성분석가: "data-level support constraint로 작동 (policy conservatism)"
- 핵심 증거: Rev2 step 433에서 전체 1024개 스캔, 0개 accept → 전면 차단

---

## 3. 변인 통제 문제 (5명 공통 지적)

### 심각한 confound: Rev2 vs Rev3 비교

동시에 변한 변인:
1. training_mode (legacy_bonus vs fixed_total_with_adv0_drop)
2. buffer 크기 (1024 vs 2048)
3. effective batch size (최대 384 vs 고정 256)

→ **Rev2 vs Rev3 비교에서 어떤 결론도 인과적으로 귀속 불가**

### 필수 격리 실험 (전문가 공통 권고)

**P1 (최우선)**: Rev3 설정 + buffer=1024 실행 → buffer 효과 분리
**P2**: Rev4 step 450까지 완주 → floor 메커니즘 장기 검증
**P3**: tau=0.005 또는 0.01 변경 실험 → tau sensitivity 파악

---

## 4. 효율성 비교

### Wall-clock (step 250까지)
| | 시간(추정) | Baseline 대비 | MATH@16 |
|---|---|---|---|
| Baseline | 3.57h | 1.00x | 0.5203 |
| Rev2 | 7.58h | 2.12x | 0.5162 |
| Rev3 | 4.62h | 1.29x | 0.5079 |
| Rev4 | 5.95h | 1.67x | 0.5107 |

→ **Rev3가 비용-편익 최적** (1.29x 오버헤드로 장기 안정성 확보)

### "학습 수명 연장" 가치
- Baseline: 유효 학습 ~480 steps → 재학습 시 추가 ~7시간
- Rev3: 464+ steps (아직 안정, 더 갈 수 있음)
- Rev3의 1.05h 오버헤드 < Baseline 재학습 비용

---

## 5. 논문 가능성 평가

### 가장 강력한 발견
"entropy_coeff=0, use_kl_loss=False 조건에서도 M2-gated replay만으로 entropy가 보존되고 학습 붕괴가 방지된다"

### 스토리라인 강도: 7/10 (현재) → 8.5/10 (P0+P1 완료 시)

### 필수 추가 실험 (우선순위, 토론 결과 기반)
1. **P0 (긴급)**: ADV=0 drop only (replay off) — 기여도 분리의 결정적 실험
2. **P1 (높음)**: Random replay (tau=∞) + Rev3 설정 — M2 gating 순수 기여
3. **P2 (높음)**: Rev3 + buffer=1024 — buffer confound 제거
4. **P3 (중간)**: tau ablation {0.001, 0.005, 0.01} — tau 민감도 + M2PO(τ=0.04) 비교
5. **P4 (중간)**: Multi-seed Rev3 (3 seeds) — 통계적 유의성

---

## 6. 토론 Round 2 핵심 결론 (2026-03-01 에이전트 팀 토론 결과)

상세: `notes/debate_final_synthesis.md`

### 확정된 사실
- Baseline 3단계 붕괴는 KL-free GRPO의 구조적 문제 (DAPO 논문에서도 확인)
- M2+ADV=0 결합이 붕괴 방지, M2 단독(Rev2)은 불충분
- ADV=0 drop이 entropy 보존의 주 기여자일 가능성 높음 (65-80%, 불확실)

### M2 게이팅 특성 (토론 수정 합의)
- M2는 policy distance 기반이나, 단조적 policy concentration 환경에서는 recency filter와 구분 어려움
- M2의 핵심 가치: **동적 조건(gradient spike 등)에서의 적응성** (recency filter에는 없음)
- IS variance bound (Var[w] < τ) 유효, 그러나 IS bias는 미보정 (의도적 설계)
- "on-policy와 동등" → "IS 추정 분산 충분히 낮음 (ESS > 99.9%)" 으로 교정

### 논문 메커니즘 해석
- M2-gated replay = implicit KL(π_θ || Π_old) regularization
- KL-free 세팅에서 명시적 KL penalty를 데이터 구성으로 대체
- 알고리즘(목적함수) 변경 없이 안정화 → DAPO 등과 직교적

### 핵심 외부 논문 (권위 출처 확인됨)
- R²VPO (arXiv:2601.03320): IS variance를 연속 regularizer로 사용 — M2의 이론적 근거 독립 검증
- M2PO (arXiv:2510.01161): 동일 M2 메트릭, 다른 맥락(stale data masking, replay 아님)
- RePO (arXiv:2506.09340): 단순 replay +18.4pp, IS correction 없음, 장기 안정성 미검증
- BAPO (arXiv:2602.20722): IS correction + buffer reuse, 어려운 샘플 재평가
- DAPO (arXiv:2503.14476): dynamic sampling ≈ ADV=0 drop, 목적함수 변경 접근
- "GRPO is Off-Policy" (arXiv:2509.24203): GRPO의 내재적 off-policy 성질 이론적 규명
- "Rethinking Trust Region" (arXiv:2602.04879): PPO clip 한계 → M2 group-level이 더 안정적

### 논문 추천 제목
"M2-Gated Experience Replay: Implicit Behavior Regularization for Stable KL-Free GRPO Training"

### 주장 금지 사항 (토론 합의)
- "M2가 유해 replay를 선별 차단한다" (실제는 policy drift 크기 기반 결정)
- "M2 통과 = on-policy와 동등" (IS bias 미보정)
- "Rev3 안정성 = M2 단독 효과" (결합 시스템의 효과, confound 존재)

## 7. 실행 우선순위 (최종)

1. **P0: ADV=0 drop only** (최우선, 논문 핵심 주장 방어에 필수)
2. **P1: Random replay** (M2 gating 순수 기여 분리)
3. **Rev4 완주** (진행 중)
4. **P2: Rev3-buffer1024** (buffer confound)
5. **P3: tau ablation**
6. **P4: Multi-seed**

---

## 8. 토론 Round 3-4 최종 결론 (2026-03-01, 문헌 검증 + 교차 토론)

상세: `notes/debate_round4_cross_verdict.md`, `notes/experiment_design_P0_P1.md`

### 문헌 대결 결과 (Skeptic 3:3 Advocate)

| 쟁점 | 승자 | 근거 |
|------|------|------|
| M2 vs 시간 필터 | Skeptic | Entropy Mechanism (2505.22617): 단조 entropy → 단조 M2 → recency와 구분 불가 |
| IS correction 필요성 | Skeptic | R²VPO, V-trace 등 전 문헌이 IS 보정 사용; M2 binary gating은 IS spectrum 하단 |
| ADV=0 drop 참신성 | Skeptic | DAPO 선행, **RL-ZVP (ICLR 2026, 2509.21880)**: ADV=0 drop이 suboptimal (-4.06pt) |
| M2 metric 이론 | Advocate | R²VPO, M2PO, GSPO가 group-level M2 설계를 독립적으로 지지 |
| 장기 안정성 | Advocate | 기존 문헌에 replay+GRPO 500+ step 안정성 데이터 없음 → 우리 결과 유일 |
| 알고리즘 직교성 | Advocate | 목적함수 변경 없는 안정화 = DAPO/BAPO와 다른 차원 |

### 핵심 위협: RL-ZVP (ICLR 2026)
- ADV=0 drop이 suboptimal임을 실험적으로 입증
- all-correct group removal이 -4.06pt 손실 초래
- entropy-guided advantage 제안
- **대응**: ADV=0 drop을 "replay-opportunity maximization"으로 reframe

### 논문 강도 업데이트
- **현재: 6.5/10** (Round 2의 7/10에서 하락, RL-ZVP + IS 비판)
- **P0+P1 완료 시: 8/10** (기여도 분리 확정)

### 추가된 외부 논문 (Round 3-4)
- **RL-ZVP** (arXiv:2509.21880, ICLR 2026): ADV=0 drop 비판, entropy-guided advantage
- **GSPO** (arXiv:2507.18071): Qwen 팀, group-level IS ratio = sequence-level, M2 group 설계 지지
- **Revisiting GRPO** (arXiv:2505.22257): ADV=0 drop의 policy improvement lower bound 이론적 정당화
- **Entropy Mechanism** (arXiv:2505.22617): GRPO entropy 단조 감소 → M2 monotonic 증가
- **ExGRPO** (arXiv:2503.07362): sophisticated replay > random replay
- **PTR-PPO** (arXiv:2501.12262): prioritized trajectory replay, off-policy correction 동반

### IS Correction Spectrum (문헌 기반 위치 파악)
```
Random replay → M2 gating (binary) → R²VPO (continuous) → V-trace (truncated) → Full IS
     ↑ no correction            ↑ 우리 위치                                    ↑ 비현실적
```

### 주장 금지 사항 (Round 3-4 추가)
- "ADV=0 drop이 새로운 기여" (DAPO 선행, RL-ZVP 비판 존재)
- "M2가 stale 데이터를 선별 차단" (monotonic entropy 환경에서 recency와 동치)
- "IS correction 불필요" (문헌 표준과 괴리)

### P0/P1 실험 설계 (코드 수정 포함)
- **P0** (ADV=0 only): `schedule.adv0_no_fallback=true` 코드 수정 필요 (ray_trainer.py 3곳)
- **P1** (Random replay): `tau=1000000.0` 설정만으로 구현 (코드 수정 불필요)
- **P2** (Buffer=1024): `buffer.max_query_groups=1024` (코드 수정 불필요)
- 상세 설계: `notes/experiment_design_P0_P1.md`

### 토론 산출물 목록 (17개 문서)
```
notes/debate_data_brief.md          — 데이터 요약 (에이전트용)
notes/debate_position_*.md (5)      — Round 1 독립 입장서
notes/debate_rebuttal_*.md (5)      — Round 2 반박서
notes/debate_synthesis_round1_2.md  — Round 1-2 종합
notes/debate_final_synthesis.md     — Round 2 최종 합의
notes/debate_round3_lit_pro_m2.md   — Round 3 문헌: M2 지지 (10 papers)
notes/debate_round3_lit_skeptic.md  — Round 3 문헌: M2 회의 (11 papers)
notes/debate_strategy_update_v2.md  — 전략 v2 (GRPO variant taxonomy)
notes/debate_round4_cross_verdict.md — Round 4 교차 토론 최종 판정
notes/experiment_design_P0_P1.md    — P0/P1/P2 실험 구체 설계
```
