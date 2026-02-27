# GRPO 실패 메커니즘 종합 분석 — Â-axis Dampening 관점

> **Date**: 2026-02-13
> **Team**: 3-agent research team (math-analyst, code-analyst, lit-researcher) + team-lead synthesis
> **Objective**: GRPO의 구체적 실패 메커니즘을 규명하고, 우리 Â-axis dampening의 타당성 근거를 확립

---

## 1. Executive Summary

GRPO의 실패는 **group normalization의 std 나눗셈**에서 기원한다.
성공률이 높아지면 std → 0이 되어 소수 오답의 advantage가 극단적으로 증폭된다.

**핵심 공식** (Bessel 보정, verl 구현 기준):
```
k개 정답, G-k개 오답일 때:
  |Â_wrong| = (k/G) / sqrt(k(G-k) / (G(G-1)))

극단 케이스 (k = G-1, 하나만 오답):
  |Â_wrong| = (G-1) / sqrt(G)

G=16 → 3.75,  G=32 → 5.48,  G=64 → 7.88
```

이 현상은 **6개 이상의 독립 논문, 7개 이상의 GitHub issue, 다수의 실무자 보고**로 실증되었다.

---

## 2. 실패 메커니즘: 3축 교차 검증

### 2.1 수학적 분석 (math-analyst)

| G | k=G-1일 때 \|Â_wrong\| | k=G-2일 때 \|Â_wrong\| | k=G/2 (정상) |
|---|------------------------|------------------------|-------------|
| 4 | 1.50 | 1.00 | 1.00 |
| 8 | 2.47 | 1.63 | 1.00 |
| 16 | 3.75 | 2.42 | 1.00 |
| 32 | 5.48 | 3.50 | 1.00 |
| 64 | 7.88 | 5.02 | 1.00 |

**위험 구간** (G=16 기준):
- 안전: p < 0.85 → E[max|Â|] < 2.5
- 경고: 0.85 < p < 0.93 → E[max|Â|] 2.5~4.0
- 위험: 0.93 < p < 0.97 → E[max|Â|] 4.0~6.0
- 치명적: p > 0.97 → E[max|Â|] > 6.0

**임계점**: p > 1 - 1/G (G=16일 때 93.75%)

**GRPO의 이중 실패** (p=0.95, G=16):
- P(k=15) ≈ 34% → |Â| 폭발 (3.75)
- P(k=16) ≈ 44% → Â=0 (zero gradient)
- **정상 학습 가능 그룹: 22%에 불과**

### 2.2 코드 경로 분석 (code-analyst)

**실패 전파 경로**:
```
compute_grpo_outcome_advantage()     ← |Â| = 3.75 생성
  ↓ core_algos.py:375-438
data.batch["advantages"] = Â         ← 극단값 저장
  ↓ ray_trainer.py:237-245
policy_loss_fn(advantages=Â)          ← vanilla PPO: ratio × Â
  ↓ core_algos.py:1268-1367
  PPO clip: min(ratio×Â, clip(ratio)×Â)  ← ratio만 clip, Â는 무방비!
  ↓
loss.backward()                       ← gradient ∝ Â (3.75배 증폭)
  ↓ dp_actor.py:656-660
grad_clip(max_norm=1.0)              ← 이미 늦음, 상대적 비율 유지
```

**PPO 클리핑이 무력한 이유**:
- clip(ratio, 0.8, 1.2) × Â = 1.2 × 3.75 = 4.50 (여전히 극단적)
- 클리핑은 ratio의 범위를 제한하지, advantage의 크기를 제한하지 않는다

**다른 estimator와의 비교**:
| Estimator | Std 나눗셈 | k=G-1 시 \|Â\| | 안전성 |
|-----------|-----------|---------------|--------|
| GRPO | O (per-group) | 3.75 | ❌ |
| GAE | O (global whiten) | ~1.5 (batch 분산) | ⚠️ |
| RLOO | ✕ | 1.07 | ✅ |
| Dr.GRPO | ✕ | 0.94 | ✅ |
| REINFORCE++ | O (global batch) | ~1.5 | ✅ |

**핵심 발견**: RLOO와 Dr.GRPO는 std 나눗셈을 제거하여 |Â| ≤ ~1.07로 제한.

### 2.3 문헌적 실증 (lit-researcher)

#### 실험적 실패 사례

| 출처 | 현상 | 수치 |
|------|------|------|
| REINFORCE++ | 훈련 95%, 테스트 0% (catastrophic overfitting) | AIME-24 → AIME-25 |
| DAPO | 순수 GRPO 30점 → 수정 후 50점 | AIME 2024, Qwen2.5-32B |
| Search-R1 | 3단계 붕괴: step 60→120→200-300에서 reward=0 | Qwen2.5-3B/7B |
| EDGE-GRPO | 어려운 데이터셋의 80%가 zero gradient | DeepScaleR-Hard-1K |
| GTPO | Step 175에서 entropy 고갈, 성적 정체 | Qwen2.5-Math-7B |
| Dr.GRPO | 오답 응답 길이가 지속적으로 증가 | 7B 모델 |

#### GitHub Issues (실무자 보고)

| 프레임워크 | Issue | 현상 |
|-----------|-------|------|
| verl #2790 | log_prob NaN | 훈련 중 일부 샘플 log_prob NaN |
| verl #1197 | grad norm NaN | Actor gradient가 항상 NaN |
| verl #2738 | entropy crash | Entropy 급증 후 학습 crash |
| ms-swift #3136 | lr=0, grad_norm=NaN | 후반부 학습 완전 붕괴 |
| unsloth #2470 | VRAM 누수 | 600-700 step 후 OOM crash |

#### Search-R1 3단계 붕괴 궤적 (가장 상세한 기록)

```
Phase 1 (step 0-60):   reward ↑, likelihood 변화 없음 (조기 정체)
Phase 2 (step 60-120):  reward ↑, likelihood ↓ (점진적 쇠퇴)
Phase 3 (step 120+):    gradient magnitude 급등, entropy 폭발
                         → step 200-300에서 reward = 0 (완전 붕괴)
```

**핵심 경고**: "Reward가 상승하는 동안에도 correct response의 likelihood는 이미 하락 중"
→ **Reward는 후행 지표(lagging indicator)이며, likelihood 하락이 유일한 선행 경고**

#### 성공적 훈련의 공통점

| 모델 | 최종 성공률 | 위험 구간 진입 여부 |
|------|-----------|------------------|
| DeepSeekMath GSM8K | 88.2% | ❌ (93% 미만) |
| DeepSeek-R1 AIME | 77.9% | ❌ (85% 미만) |
| DAPO AIME | 50% | ❌ (위험 이전) |

**결론**: 성공적 GRPO 훈련은 모두 예측된 위험 구간 진입 전에 종료되었다.

---

## 3. 기존 해법과 우리 접근의 차이

### 3.1 기존 해법 분류

| 해법 | 축 | 접근 | 효과 | 부작용 |
|------|---|------|------|--------|
| Dr.GRPO | Â축 | std 나눗셈 **제거** | |Â| ≤ 1.0 | 난이도 자동조절 상실 |
| REINFORCE++ | Â축 | global batch std | |Â| 안정화 | Per-group 대비 정보 손실 |
| DAPO | 데이터축 | zero-var 그룹 **제거** | 극단 그룹 배제 | 데이터 낭비, oversampling 필요 |
| SAPO | r축 | importance ratio **dampening** | ratio 안정화 | Â는 여전히 극단적 |
| CISPO | r축 | ratio stop-gradient | gradient 차단 | 초기(ratio≈1)에는 무효 |
| EDGE-GRPO | Â축 | entropy 기반 advantage 수정 | zero-var 그룹 활용 | 추가 연산 |

### 3.2 우리의 접근: Â-axis Dampening

```python
# 기존 GRPO advantage 계산
Â = compute_grpo_outcome_advantage(...)  # |Â| = 3.75 가능

# 우리의 dampening
g = sigmoid(-β * (|Â| - threshold))      # |Â| < threshold → g≈1, |Â| > threshold → g→0
Â_dampened = Â * g                        # 극단값만 억제
```

**핵심 차별점**:
- Dr.GRPO는 std 나눗셈을 **완전히 제거** → 난이도별 가중치 조절 기능도 제거됨
- 우리는 std 나눗셈을 **유지**하되, 결과가 극단적일 때만 **부드럽게 억제**
- **난이도 자동조절 효과를 보존**하면서 꼬리 위험만 제거하는 것이 목표

**비유**: Dr.GRPO는 체온계를 버린 것, 우리는 해열제를 투여하는 것

### 3.3 우리 접근의 고유 가치 제안

1. **난이도 가중치 보존**: std 정규화는 쉬운/어려운 문제에 자동으로 다른 가중치를 부여한다.
   이 효과를 유지하면서 극단만 잘라내는 것은 Dr.GRPO가 포기한 기능이다.

2. **SAPO와의 직교성**: SAPO는 r축(importance ratio)을, 우리는 Â축(advantage)을 dampening.
   이론적으로 두 축은 독립이며 **동시 적용 가능** (SAPO × Â-dampening).

3. **점진적 억제**: clamping(torch.clamp)은 threshold에서 gradient가 갑자기 0이 된다.
   sigmoid 기반 dampening은 점진적으로 감소하여 **학습 신호를 최대한 보존**한다.

---

## 4. Â-axis Dampening의 수학적 성질 (team-lead 자체 분석)

에이전트들이 일반적 GRPO 실패에 집중한 관계로, 우리 dampening 함수의
구체적 성질은 team-lead가 직접 도출한다.

### 4.1 g(Â)의 형태

```
g(Â) = sigmoid(-β × (|Â| - τ))
     = 1 / (1 + exp(β × (|Â| - τ)))
```

| |Â| | β=3, τ=2.0 | β=5, τ=2.0 | β=3, τ=1.5 |
|-----|-------------|-------------|-------------|
| 0.5 | 0.989 | 0.999 | 0.953 |
| 1.0 | 0.953 | 0.993 | 0.818 |
| 1.5 | 0.818 | 0.924 | 0.500 |
| 2.0 | 0.500 | 0.500 | 0.182 |
| 2.5 | 0.182 | 0.076 | 0.047 |
| 3.0 | 0.047 | 0.007 | 0.012 |
| 3.75 | 0.005 | 0.000 | 0.001 |

### 4.2 Effective advantage: Â × g(Â)

핵심: dampening 후 effective advantage = Â × g(Â)

| |Â| 원래 | β=3, τ=2.0 일 때 | 억제율 |
|-----------|-------------------|--------|
| 0.5 | 0.495 | 1% |
| 1.0 | 0.953 | 5% |
| 1.5 | 1.227 | 18% |
| 2.0 | 1.000 | 50% |
| 2.5 | 0.455 | 82% |
| 3.0 | 0.141 | 95% |
| 3.75 | 0.019 | 99.5% |

**관찰**: |Â|=2.0 근처에서 effective advantage가 **최대값(~1.2)**을 가진다.
이후 급격히 감소. 즉 dampening은 effective advantage에 **암묵적 상한**을 설정한다.

이 상한은 약 `τ - 1/β` 부근에서 발생하며, `β=3, τ=2.0`일 때 ~1.2 수준이다.

### 4.3 Clamping vs Dampening

```
Clamping:   Â_eff = clamp(Â, -c, c)
            → |Â| > c에서 gradient = 0 (학습 신호 완전 소실)

Dampening:  Â_eff = Â × sigmoid(-β(|Â|-τ))
            → |Â| > τ에서 gradient ≠ 0 (점진적 감소, 소량의 학습 유지)
```

**차이점**:
- Clamping은 threshold 초과 시 **어떤 방향**인지는 알지만 **얼마나 나쁜지** 정보를 잃는다
- Dampening은 정보를 줄이되 완전히 잃지 않는다
- Clamping의 gradient 불연속은 최적화에 불안정을 유발할 수 있다

### 4.4 "반쪽 해결" 문제

우리 dampening은 |Â| 폭발(k=G-1 그룹의 34%)만 해결한다.
Â=0 소멸(k=G 그룹의 44%)은 해결하지 않는다.

**판단**: 이것은 의도된 설계이다.
- Â=0 소멸 문제는 **DAPO의 dynamic sampling** 영역이다
- 우리 dampening은 dynamic sampling과 **직교적으로 결합 가능**하다
- Pilot 단계에서는 dampening 단독 효과를 먼저 검증한다

### 4.5 Negative-only 변형

연구 메모에서 제안된 변형: 음의 advantage(오답)만 dampening, 양의 advantage는 유지.

**근거**:
- DAPO의 asymmetric clipping (ε_low=0.2, ε_high=0.28)과 같은 철학
- 양의 advantage(정답)는 "좋은 방향"이므로 억제할 필요 없음
- 음의 advantage(오답)의 극단적 크기가 gradient를 지배하는 것이 문제

**주의사항**:
- 양의 advantage도 극단적일 수 있다 (k=1, G=16 → |Â_correct| = 3.75)
- 이 경우 "하나만 맞은" 응답에 과도한 가중치 부여
- 완전한 해법은 양방향 dampening이 될 수 있으나, pilot에서는 negative-only 먼저 테스트

### 4.6 Search-R1 3단계 붕괴에 대한 효과 예측

| Phase | 현상 | Â-dampening 효과 |
|-------|------|-----------------|
| 1 (0-60) | reward ↑, likelihood 정체 | ⚠️ 효과 제한적. 이 단계의 문제는 |Â| 폭발이 아닌 학습 효율 |
| 2 (60-120) | likelihood 하락, reward 계속 ↑ | ✅ 극단적 Â에 의한 "과도한 오답 벌칙"을 완화하여 likelihood 하락 속도 감소 가능 |
| 3 (120+) | gradient spike → collapse | ✅✅ **핵심 효과 지점**. gradient spike의 원인인 |Â|=3.75 증폭을 직접 억제 |

**예측**: Phase 3의 gradient spike를 억제하여 collapse를 지연/방지할 수 있으나,
Phase 1-2의 근본 문제(GRPO estimator bias)는 해결하지 않는다.

---

## 5. Pilot 실험에 대한 시사점

### 5.1 관측해야 할 지표

| 지표 | 의미 | GRPO 실패 시 패턴 | Dampening 성공 시 기대 |
|------|------|------------------|---------------------|
| reward | 성공률 | 후행 지표, 늦게 하락 | 더 높은 최종값 |
| entropy | 정책 다양성 | step 175 부근 고갈 | 고갈 지연/방지 |
| likelihood (정답) | 올바른 응답 확률 | Phase 2에서 하락 시작 | 하락 속도 감소 |
| grad_norm | gradient 크기 | Phase 3에서 spike | spike 억제 |
| max\|Â\| | advantage 극단값 | 3.75+ 빈번 | τ 부근으로 제한 |
| Â 분포 | advantage 분포 | 극단적 bimodal | 더 집중된 분포 |

### 5.2 실험 설계 확인

- **모델**: Qwen3-4B-Base (RL effective accuracy ~45-60%, 점진적 |Â| 발생)
- **데이터**: GSM8K (binary {0,1}, 7,473 train)
- **G**: 16 (|Â|_max = 3.75, 93% 임계점에서 충분히 극단적)
- **비교**: vanilla GRPO vs grpo_dampened (same policy loss = vanilla)
- **dampening 파라미터**: τ=2.0, β=3.0 (|Â|>2.0에서 50% 억제, >3.0에서 95% 억제)

### 5.3 예상되는 3상 학습 역학

```
Phase 1 (accuracy 45-60%): 양 실험 동일. |Â|_max ≈ 1.0, dampening 거의 미작동.
Phase 2 (accuracy 60-85%): 차이 시작. |Â|_max ≈ 1.5-2.5, dampening 점진적 개입.
Phase 3 (accuracy 85-95%): 핵심 분기점.
  - Vanilla GRPO: |Â|_max → 3.75, gradient spike, entropy collapse 위험
  - grpo_dampened: |Â|_eff capped at ~1.2, gradient 안정, 학습 지속 기대
```

---

## 6. 리스크 및 미결 사항

### 6.1 우리 접근이 해결하지 않는 문제

1. **GRPO estimator bias**: REINFORCE++ 논문이 증명한 분자/분모 비독립성 편향.
   dampening은 크기를 줄이지만 편향 자체를 제거하지 않는다.

2. **Zero-gradient groups** (k=0 또는 k=G): Â=0이므로 dampening과 무관.
   이 문제는 별도의 dynamic sampling이 필요하다.

3. **Search-R1 Phase 1-2 문제**: likelihood degradation의 근본 원인이
   |Â| 폭발인지 다른 메커니즘(ratio drift, KL divergence)인지 불확실.

### 6.2 Pilot에서 검증해야 할 가설

1. **H1**: dampening이 Phase 3 collapse를 방지/지연한다
2. **H2**: dampening이 최종 성능(accuracy)을 개선한다
3. **H3**: dampening이 entropy collapse를 방지한다
4. **H4**: negative-only dampening이 양방향보다 효과적이다

### 6.3 실패 시나리오

- dampening이 학습 신호를 과도하게 줄여 **수렴 속도가 크게 저하**될 수 있다
- τ 설정이 부적절하면 dampening이 너무 일찍/늦게 작동할 수 있다
- GRPO bias가 dominant factor라면 dampening만으로는 부족할 수 있다

---

## 7. 참고 문헌

### 직접 인용 논문
1. **DAPO**: "DAPO: An Open-Source LLM RL System at Scale" (arxiv 2503.14476)
2. **REINFORCE++**: "Stabilizing Critic-Free Policy Optimization" (arxiv 2501.03262)
3. **Dr.GRPO**: "Understanding R1-Zero-Like Training" (arxiv 2503.20783)
4. **SAPO**: "Soft Adaptive Policy Optimization" (arxiv 2511.20347)
5. **CISPO**: "Clipped Importance Sampling Policy Optimization" (Swift docs)
6. **EDGE-GRPO**: "Entropy-Driven GRPO" (arxiv 2507.21848)
7. **GTPO**: "Flexible Entropy Control in RLVR" (arxiv 2602.09782)
8. **Search-R1 Collapse**: "On GRPO Collapse in Search-R1" (arxiv 2512.04220)
9. **GDPO**: (arxiv 2601.05242)
10. **Adaptive-Boundary-Clipping GRPO** (arxiv 2601.03895)

### 커뮤니티 리소스
- GRPO++ Tricks (Cameron Wolfe): cameronrwolfe.substack.com/p/grpo-tricks
- Qwen3-8b Training Report: dtianyou.com/en/notes/qwen3-8b-base-training/
- verl Issues: #514, #593, #742, #1197, #2302, #2738, #2790
- ms-swift Issue: #3136
- unsloth Issue: #2470
- HuggingFace open-r1 discussion: #20

---

## 부록 A: 전체 |Â| 테이블 (Bessel 보정, G=16)

| k (정답 수) | k/G | std | \|Â_correct\| | \|Â_wrong\| | max\|Â\| | 그룹 확률 (p=0.9) |
|------------|-----|-----|---------------|-------------|---------|------------------|
| 0 | 0.000 | 0.000 | 0 | 0 | 0 (vanishing) | 0.000 |
| 1 | 0.063 | 0.258 | 3.750 | 0.250 | 3.750 | 0.000 |
| 2 | 0.125 | 0.354 | 2.646 | 0.378 | 2.646 | 0.000 |
| 4 | 0.250 | 0.447 | 1.732 | 0.577 | 1.732 | 0.001 |
| 8 | 0.500 | 0.516 | 1.000 | 1.000 | 1.000 | 0.001 |
| 12 | 0.750 | 0.447 | 0.577 | 1.732 | 1.732 | 0.051 |
| 14 | 0.875 | 0.354 | 0.378 | 2.646 | 2.646 | 0.229 |
| 15 | 0.938 | 0.258 | 0.250 | 3.750 | 3.750 | 0.329 |
| 16 | 1.000 | 0.000 | 0 | 0 | 0 (vanishing) | 0.185 |

> 주: std는 Bessel 보정 (N-1), 그룹 확률은 Binomial(16, 0.9)
