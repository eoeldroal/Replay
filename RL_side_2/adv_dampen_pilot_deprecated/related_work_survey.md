# 관련 문헌 조사: GRPO Advantage 수정 접근법 비교

> **Date**: 2026-02-13
> **Scope**: GRPO의 advantage estimation / policy loss 불안정을 해결하는 기존 연구 전수 조사
> **Focus**: 우리 Â-axis dampening과의 위치 관계 및 novelty 근거

---

## 1. 분류 체계

GRPO 불안정 해결을 위한 기존 연구를 **수정 축(axis)** 기준으로 분류한다.

```
GRPO Policy Gradient = -E[ f(ratio) × Â ]

  ① r축: f(ratio) 수정 — importance ratio 자체를 제어
  ② Â축: Â 수정 — advantage 값을 변형/제한
  ③ 집계축: loss aggregation 방식 변경 — token/seq/geometric mean
  ④ 데이터축: 입력 데이터 필터링/재가중
```

우리의 접근(Â-axis dampening)은 ②에 해당한다.

---

## 2. r축 (Importance Ratio) 수정

ratio를 제어하여 policy update의 크기를 제한하는 접근.
**가장 활발하게 연구된 축**이다.

### 2.1 PPO Hard Clipping (Schulman et al., 2017)

- **수식**: `clip(ratio, 1-ε, 1+ε)` where ε=0.2
- **한계**: ratio만 clip하므로 |Â|가 클 때 gradient는 여전히 Â에 비례
- **우리와의 관계**: PPO clip은 Â축과 직교. 동시 적용 가능하나 Â를 제어하지 못함

### 2.2 DAPO Clip-Higher (Yu et al., 2025)

- **논문**: "DAPO: An Open-Source LLM RL System at Scale" ([arxiv 2503.14476](https://arxiv.org/html/2503.14476v1))
- **수식**: asymmetric clip `[1-ε_low, 1+ε_high]` where ε_low=0.2, ε_high=0.28
- **핵심**: ε_high를 키워 positive ratio의 exploration 허용. Entropy collapse 완화.
- **한계**: 여전히 r축만 제어. Â 크기에 대한 보호 없음.
- **AIME 결과**: 30% → 50% (but dynamic sampling과 결합)

### 2.3 SAPO — Soft Adaptive Policy Optimization (Gao et al., 2025)

- **논문**: "Soft Adaptive Policy Optimization" ([arxiv 2511.20347](https://arxiv.org/pdf/2511.20347))
- **수식**: hard clip 대신 temperature-controlled sigmoid gate
  ```
  w(ratio) = sigmoid((ratio - 1) / τ)  (soft gate on ratio)
  ```
- **핵심**: r축을 부드럽게 감쇠. τ_neg > τ_pos로 음의 ratio에 더 강한 감쇠.
- **우리와의 관계**: **축이 다르다**. SAPO는 r축, 우리는 Â축.
  이론적으로 SAPO × Â-dampening 동시 적용 가능 (직교적 결합).

### 2.4 CISPO — Clipped Importance Sampling Policy Optimization (2025)

- **논문**: [Swift CISPO docs](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/CISPO.html)
- **수식**: `clipped_ratio.detach() * Â * log_prob` (ratio에 stop-gradient)
- **핵심**: ratio 방향의 gradient 차단. Low-probability reasoning token 보호.
- **한계**: 초기 학습(ratio ≈ 1.0)에서는 clipping이 비활성 → 극단적 Â가 그대로 전파.
- **우리와의 관계**: CISPO는 r축 gradient 차단, 우리는 Â축 크기 억제. 결합 가능.

### 2.5 PSPO — Probability Smoothing Policy Optimization (2025)

- **논문**: "It's Not You, It's Clipping" ([arxiv 2509.21282](https://arxiv.org/html/2509.21282))
- **수식**: `r̃ = (1-α)r + α` (smoothed ratio, α로 1.0 방향 수축)
- **핵심**: ratio를 1.0 방향으로 수축시켜 soft trust region 구현.
  gradient = (1-α) × Â everywhere (advantage는 uniform 스케일링만)
- **한계**: Â 크기를 차별적으로 처리하지 않음. |Â|=3.75도 |Â|=0.5도 같은 비율로 스케일.
- **우리와의 관계**: PSPO의 uniform 스케일 vs 우리의 selective dampening. 철학이 다름.

### 2.6 P3O / SCOPIC (Chen et al., 2023)

- **수식**: sigmoid-based soft clip on ratio
- **핵심**: hard clip의 zero-gradient 문제를 sigmoid로 해결.
- **한계**: Â 자체는 수정하지 않음.
- **참고**: SAPO의 선행 연구에 해당.

---

## 3. Â축 (Advantage Value) 수정

advantage 값 자체를 변형하는 접근. **가장 희소한 연구 영역**.

### 3.1 Dr.GRPO — std 나눗셈 제거 (Liu et al., 2025)

- **논문**: "Understanding R1-Zero-Like Training" ([arxiv 2503.20783](https://arxiv.org/html/2503.20783v2))
- **수식**: `Â = r_i - mean(r_group)` (std 나눗셈 없음)
- **핵심**: |Â| 폭발의 근원인 1/std 항을 완전 제거.
  결과적으로 |Â| ≤ 1.0 (binary reward 기준).
- **성과**: AIME 2024에서 43.3% (7B 모델 SOTA)
- **부작용**: **난이도별 자동 가중치 조절 기능 상실**. 쉬운 문제와 어려운 문제를 동일하게 취급.
- **우리와의 차이**:
  - Dr.GRPO = **사전 차단** (계산 방식 자체 변경)
  - 우리 = **사후 억제** (std 나눗셈 유지, 극단적 결과만 dampening)
  - 우리 접근은 난이도 가중치를 **보존**하면서 꼬리 위험만 제거

### 3.2 REINFORCE++ — Global Batch Normalization (Hu, 2025)

- **논문**: "Stabilizing Critic-Free Policy Optimization" ([arxiv 2501.03262](https://arxiv.org/html/2501.03262v9))
- **수식**: `Â = (r_i - mean_batch) / std_batch` (global batch 단위)
- **핵심**: per-group std 대신 전체 batch std 사용.
  batch 크기가 크면 (1024+) std가 안정 → |Â| 극단값 감소.
- **수학적 보장**: batch → ∞이면 unbiased estimator 수렴.
- **한계**: per-group normalization의 난이도 적응 기능 상실.
  모든 prompt를 하나의 분포로 간주.
- **우리와의 차이**: REINFORCE++는 normalization **범위**를 변경, 우리는 normalization **결과**를 억제.

### 3.3 AGPO — Zero-Variance 고정값 (2025)

- **논문**: "Adaptive Group Policy Optimization" ([arxiv 2503.15952](https://arxiv.org/abs/2503.15952))
- **수식**:
  ```
  Â = +1   if mean(r) = r_max   (all correct)
  Â = -1   if mean(r) = r_min   (all wrong)
  Â = (r - mean) / std   otherwise   (일반 GRPO)
  ```
- **핵심**: k=0 또는 k=G (Â=0 vanishing) 그룹에 고정 advantage 부여.
  zero-gradient 문제 해결.
- **한계**: **k=G-1 등 극단적 |Â| 폭발은 전혀 처리하지 않음**. 일반 케이스는 GRPO 그대로.
- **우리와의 차이**: AGPO는 Â=0 (vanishing) 해결, 우리는 |Â|>>1 (explosion) 해결.
  상호 보완적이며 동시 적용 가능.

### 3.4 EDGE-GRPO — Entropy 기반 Â 재가중 (2025)

- **논문**: "Entropy-Driven GRPO with Guided Error Correction" ([arxiv 2507.21848](https://arxiv.org/html/2507.21848v1))
- **수식**: `Â_new = Â / P̂` where P̂ = normalized policy entropy
- **핵심**: entropy가 낮은(확신 높은) 정답에 더 큰 advantage 부여.
  zero-advantage 그룹에서도 학습 신호 생성.
- **성과**: 다양한 base model에서 20%+ 개선
- **한계**: advantage를 **키우는** 방향. |Â| 폭발에 대한 억제가 없음.
- **우리와의 차이**: EDGE-GRPO는 Â를 키움(zero-var 해결), 우리는 Â를 줄임(explosion 해결).
  정반대 방향이나 결합 가능 (먼저 EDGE-GRPO로 zero-var 해결, 그 후 dampening으로 explosion 해결).

### 3.5 ProGRPO — 확률 신호 기반 Â 분포 Reshape (2025)

- **논문**: "Back to Basics: Revisiting Exploration in RL for LLM Reasoning" ([arxiv 2602.05281](https://arxiv.org/html/2602.05281))
- **수식**: LLM의 내부 확률 신호(generative probability)를 사용하여 advantage 분포를 reshape
- **핵심**: entropy collapse 완화. Qwen2.5-7B에서 GRPO 대비 Pass@1 +5.7%, Pass@32 +13.9%.
- **우리와의 차이**: ProGRPO는 확률 기반 reshape, 우리는 크기 기반 dampening.
  접근 철학이 다르지만 같은 축(Â)을 수정한다는 점에서 가장 유사한 동시대 연구.

### 3.6 SCAPPO — Sigmoid Smooth Clip on Advantage (Wang et al., 2023)

- **논문**: "Smooth Clip Advantage PPO in Reinforcement Learning"
  ([ResearchGate](https://www.researchgate.net/publication/371886267_Smooth_Clip_Advantage_PPO_in_Reinforcement_Learning))
- **수식**: PPO의 advantage hard clip을 sigmoid smooth clip으로 대체
- **핵심**: sigmoid 함수를 advantage에 적용하여 극단값의 gradient vanishing 문제 해결.
- **한계**:
  - **Game RL** (OpenAI Gym) 환경. LLM 맥락이 아님.
  - GRPO의 group normalization 문제와 무관.
  - Advantage 전체에 sigmoid 적용 (우리는 |Â| > threshold인 극단값만 선택적 억제).
- **우리와의 관계**: **형식적으로 가장 유사**. 둘 다 sigmoid를 advantage에 적용.
  하지만 적용 맥락(game vs LLM), 목표(smooth gradient vs explosion suppression),
  선택성(전체 vs 극단값만)이 다르다.

---

## 4. 집계축 (Loss Aggregation) 수정

### 4.1 GMPO — Geometric Mean Policy Optimization (2025)

- **논문**: ([arxiv 2507.20673](https://arxiv.org/html/2507.20673v1))
- **수식**: arithmetic mean 대신 geometric mean으로 token-level loss 집계
  ```
  J_GMPO = (∏_t [ratio_t × |Â|])^(1/|o|) × sgn(Â)
  ```
- **핵심**: geometric mean은 outlier에 덜 민감 → ratio 극단값의 영향 감쇠.
- **성과**: GRPO 대비 +4.1% average on math benchmarks (7B)
- **한계**: Â 값 자체는 수정하지 않음. 집계 시 outlier 영향만 감소.
- **우리와의 관계**: GMPO는 aggregation 레벨에서 robustness, 우리는 advantage 레벨에서 직접 억제.

### 4.2 λ-GRPO — Learnable Token Preferences (2025)

- **논문**: "Unifying the GRPO Frameworks with Learnable Token Preferences"
  ([openreview](https://openreview.net/pdf?id=0czAcXMBNO))
- **수식**: GRPO, DAPO, Dr.GRPO를 token preference 가중치로 통합하는 프레임워크
- **핵심**: token별 loss 가중치를 학습. 길이 편향(length bias) 해결.
- **우리와의 관계**: token-level 가중치 vs sequence-level advantage dampening. 레벨이 다름.

### 4.3 BAE — Blockwise Advantage Estimation (2025)

- **논문**: "Blockwise Advantage Estimation for Multi-Objective RL"
  ([arxiv 2602.10231](https://arxiv.org/html/2602.10231))
- **수식**: 전체 응답을 block으로 분할, block별 독립 advantage 계산
- **핵심**: 중간 결과(intermediate outcome)에 조건부 baseline 사용.
  fine-grained credit assignment.
- **우리와의 관계**: BAE는 advantage의 **granularity** 변경, 우리는 **magnitude** 억제.

### 4.4 GSPO — Group Sequence Policy Optimization (2025)

- **논문**: ([arxiv 2507.18071](https://arxiv.org/pdf/2507.18071))
- **핵심**: sequence-level importance ratio의 geometric mean으로 token-level ratio 대체.
  장 sequence에서의 IS ratio 분산 감소.
- **verl 구현**: `loss_mode=gspo`로 이미 사용 가능.

---

## 5. 데이터축 (Sampling/Filtering) 수정

### 5.1 DAPO Dynamic Sampling (Yu et al., 2025)

- **수식**: accuracy=0 또는 accuracy=1인 prompt 그룹 제거 후 oversampling
- **핵심**: |Â| 폭발 및 Â=0 vanishing 그룹을 사전에 배제.
- **한계**: 데이터 낭비 (oversampling 필요). 극단 그룹에서의 학습 기회 상실.
- **우리와의 관계**: dynamic sampling은 Â-dampening과 직교적으로 결합 가능.

### 5.2 F-GRPO — Forgetting-aware GRPO (2025)

- **논문**: "Don't Let Your Policy Learn the Obvious and Forget the Rare"
  ([arxiv 2602.06717](https://arxiv.org/pdf/2602.06717))
- **핵심**: rare correct 응답(소수 정답)에 추가 가중치 부여. Catastrophic forgetting 방지.
- **우리와의 관계**: rare correct = k가 작은 그룹. 이 그룹의 정답 Â가 크다(|Â|=3.75).
  F-GRPO는 이를 **강화**, 우리는 이를 **억제**. 상충 가능성 있음.

---

## 6. Novelty 분석: 우리 접근의 고유 위치

### 6.1 선행 연구 부재 영역

```
┌──────────────────────────────────────────────────────────────┐
│               Advantage 수정 연구 지도                        │
│                                                              │
│   Â 계산 방식 변경           Â 값 직접 수정                   │
│   (사전 차단)                (사후 처리)                      │
│                                                              │
│   ● Dr.GRPO (std 제거)                                      │
│   ● REINFORCE++ (global std)   ● AGPO (zero-var → ±1)      │
│                                ● EDGE-GRPO (entropy 재가중)  │
│                                ● ProGRPO (확률 reshape)      │
│                                ● SCAPPO (sigmoid, game RL)   │
│                                                              │
│                                ★ 우리: g(Â) dampening ★     │
│                                  (크기 기반, 극단값만,        │
│                                   std 보존, LLM GRPO)       │
│                                                              │
│   ※ |Â| 폭발을 "사후적"으로 "선택적"으로 억제하는            │
│     LLM GRPO 연구 = 부재                                    │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Novelty Claim

| 차별화 요소 | Dr.GRPO | REINFORCE++ | AGPO | EDGE-GRPO | SCAPPO | **우리** |
|-------------|---------|-------------|------|-----------|--------|---------|
| std normalization 보존 | ❌ 제거 | ❌ global | ✅ | ✅ | N/A | **✅** |
| |Â| 폭발 해결 | ✅ (사전) | ✅ (사전) | ❌ | ❌ | ⚠️ (전체) | **✅ (선택적)** |
| Â=0 vanishing 해결 | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| 난이도 자동조절 보존 | ❌ | ❌ | ✅ | ✅ | N/A | **✅** |
| LLM GRPO 맥락 | ✅ | ✅ | ✅ | ✅ | ❌ game | **✅** |
| 점진적 억제 (smooth) | N/A | N/A | ❌ hard | N/A | ✅ sigmoid | **✅ sigmoid** |
| 극단값만 선택적 처리 | N/A | N/A | zero-var만 | zero-var만 | ❌ 전체 | **✅** |

### 6.3 학술적 위치 (Positioning Statement)

> 기존 GRPO 변형들은 |Â| 폭발을 **사전에 차단** (std 제거, global std)하거나,
> **Â=0 vanishing을 해결** (고정값 부여, entropy 재가중)하는 데 집중했다.
> |Â| 폭발을 **사후적으로, 선택적으로, 점진적으로 억제**하면서
> per-group std normalization의 난이도 적응 기능을 보존하는 접근은 제안된 바 없다.
>
> 우리의 Â-axis dampening은 이 공백을 채우며,
> r축 수정(SAPO, CISPO)과 데이터축 수정(DAPO dynamic sampling)과
> 직교적으로 결합 가능한 독립적 축을 제공한다.

---

## 7. 결합 가능성 분석

우리 dampening이 기존 해법들과 결합 가능한지 분석한다.

| 기존 해법 | 결합 가능? | 기대 효과 | 충돌 위험 |
|-----------|-----------|----------|----------|
| DAPO Dynamic Sampling | ✅ 직교적 | dampening이 explosion 해결, dynamic sampling이 vanishing 해결 | 없음 |
| DAPO Clip-Higher | ✅ 직교적 | ratio 비대칭 + Â dampening | 없음 |
| CISPO | ✅ 직교적 | ratio stop-gradient + Â dampening | 초기 학습에서 이중 억제 가능성 |
| GSPO | ✅ 직교적 | seq-level ratio + Â dampening | 없음 |
| SAPO | ✅ 직교적 | r축 dampening + Â축 dampening | 이중 억제로 학습 속도 저하 가능성 |
| Dr.GRPO | ❌ 불필요 | Dr.GRPO가 std 제거하면 |Â| 폭발 자체가 없음 | dampening 무의미 |
| REINFORCE++ | ❌ 불필요 | global std가 |Â| 안정화 | dampening 거의 무의미 |
| AGPO | ✅ 상호보완 | AGPO가 vanishing 해결, 우리가 explosion 해결 | 없음 |
| EDGE-GRPO | ⚠️ 주의 | entropy 재가중 후 dampening하면 재가중 효과 일부 상쇄 | 상충 가능성 |
| F-GRPO | ⚠️ 주의 | F-GRPO가 rare correct 강화, 우리가 extreme Â 억제 → 상충 | 직접 상충 |

**최적 조합 후보**:
1. **Â-dampening + DAPO Dynamic Sampling**: explosion + vanishing 양쪽 해결
2. **Â-dampening + CISPO**: Â축 + r축 이중 보호
3. **Â-dampening + AGPO**: explosion + vanishing 양쪽 해결 (DAPO보다 데이터 낭비 없음)

---

## 8. 주의사항 및 한계

### 8.1 SCAPPO와의 차별화 필요

SCAPPO(2023)는 형식적으로 우리와 가장 유사하다 (sigmoid on advantage).
논문 작성 시 다음을 명확히 구분해야 한다:
- SCAPPO는 game RL (Atari, MuJoCo), 우리는 LLM GRPO
- SCAPPO는 advantage 전체에 sigmoid 적용, 우리는 |Â| > threshold만 선택적 적용
- SCAPPO는 smooth gradient가 목표, 우리는 group normalization의 |Â| explosion 억제가 목표
- SCAPPO는 PPO의 hard clip 대체, 우리는 GRPO advantage 후처리

### 8.2 Dr.GRPO 대비 장점의 실증 필요

"난이도 자동조절 보존"이 실제로 성능 차이를 만드는지는 미검증이다.
Dr.GRPO가 std를 제거해도 잘 작동한다면 (AIME 43.3%), 우리의 "보존" 주장은
이론적 우위일 뿐 실질적 우위가 아닐 수 있다.
→ **Pilot 실험에서 Dr.GRPO(norm_adv_by_std_in_grpo=False)도 baseline으로 포함해야 함**.

### 8.3 "반쪽 해결"의 한계

우리 dampening은 |Â| explosion(34%)만 해결하고, Â=0 vanishing(44%)은 해결하지 않는다.
독립적으로는 효과가 제한적일 수 있으며, **DAPO dynamic sampling 또는 AGPO와의 결합이
실전 배포 시 필수**일 가능성이 높다.

---

## 부록: 전체 논문 목록

| # | 논문 | 연도 | 축 | arxiv/link |
|---|------|------|---|------------|
| 1 | PPO (Schulman et al.) | 2017 | r축 | 1707.06347 |
| 2 | GRPO (Shao et al.) | 2024 | baseline | 2402.03300 |
| 3 | DAPO (Yu et al.) | 2025 | r축+데이터 | [2503.14476](https://arxiv.org/html/2503.14476v1) |
| 4 | Dr.GRPO (Liu et al.) | 2025 | Â축(사전) | [2503.20783](https://arxiv.org/html/2503.20783v2) |
| 5 | REINFORCE++ (Hu) | 2025 | Â축(사전) | [2501.03262](https://arxiv.org/html/2501.03262v9) |
| 6 | SAPO (Gao et al.) | 2025 | r축 | [2511.20347](https://arxiv.org/pdf/2511.20347) |
| 7 | CISPO | 2025 | r축 | [Swift docs](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/CISPO.html) |
| 8 | AGPO | 2025 | Â축(사후) | [2503.15952](https://arxiv.org/abs/2503.15952) |
| 9 | EDGE-GRPO | 2025 | Â축(사후) | [2507.21848](https://arxiv.org/html/2507.21848v1) |
| 10 | ProGRPO | 2025 | Â축(사후) | [2602.05281](https://arxiv.org/html/2602.05281) |
| 11 | SCAPPO (Wang et al.) | 2023 | Â축 | [ResearchGate](https://www.researchgate.net/publication/371886267) |
| 12 | GMPO | 2025 | 집계축 | [2507.20673](https://arxiv.org/html/2507.20673v1) |
| 13 | GSPO | 2025 | 집계축 | [2507.18071](https://arxiv.org/pdf/2507.18071) |
| 14 | λ-GRPO | 2025 | 집계축 | [openreview](https://openreview.net/pdf?id=0czAcXMBNO) |
| 15 | BAE | 2025 | 집계축 | [2602.10231](https://arxiv.org/html/2602.10231) |
| 16 | PSPO | 2025 | r축 | [2509.21282](https://arxiv.org/html/2509.21282) |
| 17 | P3O / SCOPIC | 2023 | r축 | - |
| 18 | F-GRPO | 2025 | 데이터축 | [2602.06717](https://arxiv.org/pdf/2602.06717) |
| 19 | GTPO | 2025 | Â축+집계 | [2602.09782](https://arxiv.org/html/2602.09782) |
| 20 | Search-R1 Collapse | 2025 | 분석 | [2512.04220](https://arxiv.org/html/2512.04220v1) |
| 21 | GDPO | 2025 | Â축 | [2601.05242](https://arxiv.org/pdf/2601.05242) |
