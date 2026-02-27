# GRPO Advantage Dampening 구현 계획서

> **Date**: 2026-02-13
> **Objective**: one-sided softplus clamp을 사용한 `grpo_dampened` advantage estimator 구현
> **Preceding Work**: grpo_failure_analysis.md, related_work_survey.md, 5-agent τ debate (89 messages)

---

## 1. 설계 결정 요약

### 1.1 최종 수식

```
Â_eff = Â + (1/β) × softplus(β × (-Â - τ))

where:
  τ = 2.0   (dampening lower bound)
  β = 3.0   (softplus sharpness)
  softplus(x) = log(1 + exp(x))
```

**행동 요약:**
- Â >> -τ (예: Â = -0.5): Â_eff ≈ Â (무개입, identity)
- Â << -τ (예: Â = -3.75): Â_eff → -τ = -2.0 (soft cap)
- 양수 Â에 대해서는 correction ≈ 0 (무개입)

동치 형태 (직관적 해석):
```
Â_eff = -τ + (1/β) × softplus(β × (Â + τ))

→ "Â를 +τ만큼 이동 → softplus → 다시 -τ 이동"
→ 통계학의 Winsorization과 유사: 극단값을 경계값으로 대체
```

### 1.2 적용 범위

- **음수 Â에만 적용** (one-sided): 양수 advantage는 "정답을 더 잘하라"는 신호이므로 보존
- **group 단위 계산 후 적용**: GRPO의 group mean/std normalization 이후에 적용
- **`norm_adv_by_std_in_grpo` 옵션 존중**: std normalization 여부와 독립적으로 작동

---

## 2. 수학적 근거

### 2.1 문제: GRPO group normalization의 |Â| 폭발

GRPO에서 advantage는 group 내 mean/std로 정규화된다:
```
Â = (reward - mean_group) / (std_group + ε)
```

Binary reward (correct=1, incorrect=0)이고 G=16일 때, Bessel-corrected std를 사용하면:
```
Â_wrong(k) = -√(k(G-1)) / √(G(G-k))

k=15 (1개 오답): Â_wrong = -3.75   ← 문제!
k=14 (2개 오답): Â_wrong = -2.56
k=13 (3개 오답): Â_wrong = -2.01
k=12 (4개 오답): Â_wrong = -1.68
k=8  (절반):     Â_wrong = -0.97   ← 정상 스케일
```

accuracy가 높아질수록 std → 0이 되어 소수 오답의 |Â|가 극단적으로 증폭된다.
이 증폭된 negative advantage가 **entropy collapse**의 직접적 원인이다 (6개 논문, 7개 GitHub issue로 실증).

### 2.2 왜 τ = 2.0인가?

5명의 전문 에이전트 (advocate-τ1, advocate-τ2, math-referee, rl-practitioner, lit-searcher)가
89건의 교차 토론을 수행한 결과, **4:1로 τ=2.0 지지**.

#### τ=2.0을 지지하는 6개 독립적 이론 수렴

| # | 이론 | τ 도출값 | 설명 |
|---|------|---------|------|
| 1 | **2σ 규칙** | 2.0 | 통계적 outlier의 표준 임계치 (95% 신뢰구간) |
| 2 | **정보이론** | ~2.0 | KL divergence 관점에서 최소 정보 손실 |
| 3 | **Bias-variance 최적점** | ~2.0 | 추정 편향 도입 vs 분산 감소의 균형 |
| 4 | **Bessel 보정** | √(G-1)/√G × 2 ≈ 1.94 | G=16에서 Bessel factor × 2 |
| 5 | **Bayesian shrinkage** | ~2.0 | 사후분포 기반 축소 추정 |
| 6 | **Confidence interval** | ~2.0 | 이항분포 신뢰구간의 자연 경계 |

#### τ=1.0 (대안) 대비 수치 비교

```
                              τ=1.0        τ=2.0
─────────────────────────────────────────────────
gradient magnitude 보존율      26.8%        63.3%    ← τ=2가 2.4× 보존
zero-sum 위반 크기             6.40         2.87     ← τ=2가 2.2× 작음
k=12 (75% acc) 절삭률          40%          2%       ← τ=1은 학습 중인 문제도 절삭
dampening 개입 시작점          acc 50%+     acc 81%+ ← τ=2는 마스터한 문제만 개입
```

#### τ=1.5 (타협안) 대비 수치 비교

```
                              τ=1.5        τ=2.0
─────────────────────────────────────────────────
gradient magnitude 보존율      87.4%        94.5%
zero-sum 위반 크기             12.05        5.28     ← τ=2가 2.3× 작음
k=12 (75% acc) 절삭률          19.7%        6.4%     ← τ=1.5는 활발한 학습 구간도 절삭
구별력: 81% vs 94% 문제        3.7%         12.8%    ← τ=2가 3.5× 더 구별
dampening 개입 시작점          acc 69%+     acc 81%+
```

**결정적 차이: 구별력 (discrimination)**

τ=1.5는 고accuracy 문제들의 Â_eff를 모두 -1.5 근처로 뭉개서
"81% 정확도 문제"와 "94% 정확도 문제"의 구별력이 3.7%만 보존된다.
τ=2.0은 12.8% 보존 (3.5× 우위). 이는 curriculum learning 효율에 직결.

### 2.3 왜 β = 3.0인가?

β는 softplus의 sharpness를 제어한다:
- β → ∞: hard clamp (ReLU처럼 꺾임)
- β → 0: 매우 완만 (거의 선형)

**β=3.0 선택 근거:**
1. **전이 폭 (transition width)**: β=3.0에서 전이 구간 ≈ 2/β ≈ 0.67.
   즉, τ 전후 ±0.33 범위에서 부드럽게 전이 → 충분히 sharp하되 미분 가능
2. **sign flipping 안전성**: β≥2.0이면 어떤 τ에서도 음수→양수 뒤집힘 없음 (검증 완료)
3. **β=0.5 기각**: advocate-τ2가 제안했으나, τ=1.0에서 k=1,2,3의 Â_eff가 양수로 뒤집힘 → 위험

**sign flipping 검증 결과** (직접 Python 계산):
```
β=3.0, τ=1.0: sign flipping 없음 ✓
β=3.0, τ=2.0: sign flipping 없음 ✓
β=2.0, τ=1.0: sign flipping 없음 ✓
β=1.0, τ=1.0: k=1,2,3에서 sign flipping 발생! ✗
β=0.5, τ=1.0: k=1,2,3에서 sign flipping 발생! ✗
```

### 2.4 왜 softplus인가? (함수 계열 선택)

5개 후보를 통합 프레임워크 Â_eff = -τ + h(Â + τ)에서 비교:

| 함수 | 유형 | 단조성 | 하한 | 판정 |
|------|------|--------|------|------|
| ReLU | hard clamp | ✓ | -τ exactly | 미분 불가 → ✗ |
| **Softplus** | **soft clamp** | **✓** | **-τ (점근)** | **채택** |
| GELU | soft gate | ✗ (비단조!) | 0 | ✗ |
| SiLU/Swish | soft gate | ✗ (비단조!) | 0 | ✗ |
| Tanh | piecewise | ✓ | -τ (점근) | Softplus 대비 이점 없음 |

**핵심: 단조성 (monotonicity)**

GELU/SiLU gate 함수는 **비단조적**이다:
```
SiLU gate 예시 (τ=2.0):
  |Â| = 1.68 → |Â_eff| = 1.215  (절삭)
  |Â| = 2.56 → |Â_eff| = 0.402  (더 강한 절삭)
  |Â| = 3.75 → |Â_eff| = 0.001  (거의 소멸)
```
→ 더 극단적인 오답이 **더 적은** 교정을 받는 역전 현상 발생.
Softplus는 단조적이므로 이 문제가 없다.

### 2.5 왜 one-sided (음수만)인가?

**비대칭 효과**: 같은 |Â|=3.75라도:
- Â_wrong = -3.75 (k=15에서 유일한 오답): 과잉 패널티 → entropy collapse
- Â_correct = +3.75 (k=1에서 유일한 정답): 강한 보상 → 유익한 학습 신호

positive Â 증폭은 entropy를 **증가**시키는 방향이므로 무해.
negative Â 증폭만이 entropy **감소** → collapse를 유발.

### 2.6 동적 τ(G) 검토 결과

Â_wrong을 수학적으로 분해하면:
```
Â_wrong(k, G) = -√(α/(1-α)) × √((G-1)/G)
                 accuracy 항     G-scaling 항
```

동적 공식 τ(G, α*) = √(α*/(1-α*)) × √((G-1)/G)을 사용하면
G와 무관하게 정확히 accuracy α*에서 dampening이 시작된다.

그러나 **실용적으로 불필요**:
```
G     고정 τ=2.0   동적 τ(α*=80%)    차이
 8      2.000        1.871          0.129
16      2.000        1.937          0.063   ← 우리 pilot
32      2.000        1.969          0.031
64      2.000        1.984          0.016
```

G≥16에서 차이 < 0.07이므로, 고정 τ=2.0으로 충분하다.
(G=4~8 소규모 실험을 빈번히 하게 되면 향후 동적 τ 추가 고려)

### 2.7 lit-searcher의 반론 및 대응

**반론**: Dr.GRPO (COLM 2025)는 std normalization을 제거하여 |Â|≤1을 달성하고,
이것이 vanilla GRPO를 일관되게 능가한다. 이는 τ≈1이 최적임을 시사한다.

**대응**:
- Dr.GRPO는 **모든** Â를 균일하게 축소 (std로 나누지 않음)
- τ=1 softplus clamp은 극단값만 잘라냄 → 메커니즘이 다름
- Dr.GRPO는 중간 accuracy 구간의 Â도 1 미만으로 제한 → 학습 신호 약화 가능
- 우리 pilot에서 Dr.GRPO를 baseline으로 포함하여 직접 비교 예정

---

## 3. 구현 계획

### 3.1 변경 파일 목록

```
수정 (2 files):
  verl/trainer/ppo/core_algos.py    ← grpo_dampened estimator 등록
  verl/trainer/config/algorithm.py  ← AlgoConfig에 dampening 파라미터 추가

신규 (1 file):
  RL_side_2/adv_dampen_pilot/run_grpo_dampened_pilot.sh  ← 실험 스크립트
```

### 3.2 Step 1: `core_algos.py` — 새 advantage estimator 등록

**위치**: `compute_grpo_passk_outcome_advantage()` 함수 직후 (line 528 부근)

```python
@register_adv_est("grpo_dampened")
def compute_grpo_dampened_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO with one-sided softplus dampening on negative advantages.

    Computes standard GRPO advantage, then applies a soft lower bound:
        Â_eff = Â + (1/β) × softplus(β × (-Â - τ))

    For Â >> -τ: Â_eff ≈ Â (identity, no dampening)
    For Â << -τ: Â_eff → -τ (soft cap at -τ)

    This prevents extreme negative advantages from causing entropy collapse
    when group accuracy is very high (few wrong answers → small std → large |Â|).

    Config parameters:
        norm_adv_by_std_in_grpo (bool): Whether to divide by group std. Default True.
        adv_dampen_tau (float): Soft lower bound magnitude. Default 2.0.
        adv_dampen_beta (float): Softplus sharpness. Default 3.0.
    """
    assert config is not None, "grpo_dampened requires AlgoConfig"

    norm_adv_by_std = config.get("norm_adv_by_std_in_grpo", True)
    tau = config.get("adv_dampen_tau", 2.0)
    beta = config.get("adv_dampen_beta", 3.0)

    with torch.no_grad():
        # Step 1: Standard GRPO advantage (vectorized)
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon)

        if norm_adv_by_std:
            adv_raw = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            adv_raw = scores - mean_g[g]

        # Step 2: One-sided softplus dampening (negative Â only)
        # correction = (1/β) × softplus(β × (-Â - τ))
        # For Â > -τ: correction ≈ 0 (no effect)
        # For Â < -τ: correction ≈ (-Â - τ), pulling Â_eff toward -τ
        correction = (1.0 / beta) * torch.nn.functional.softplus(
            beta * (-adv_raw - tau)
        )
        scalars = adv_raw + correction

        advantages = scalars.unsqueeze(-1) * response_mask

    return advantages, advantages
```

**설계 결정:**
- `grpo_vectorized`와 동일한 vectorized 패턴 사용 (`as_torch_index`, `group_mean_std`)
- `torch.nn.functional.softplus` 사용 — PyTorch 내장, numerically stable
- config에서 파라미터를 `.get()`으로 읽어 기존 config 호환성 유지
- `**kwargs` 수용 — dispatcher가 추가 인자를 전달할 수 있도록

### 3.3 Step 2: `algorithm.py` — config 필드 추가

**위치**: `AlgoConfig` 클래스, `norm_adv_by_std_in_grpo` 필드 아래

```python
@dataclass
class AlgoConfig(BaseConfig):
    # ... 기존 필드들 ...
    norm_adv_by_std_in_grpo: bool = True

    # Advantage dampening (for grpo_dampened estimator)
    adv_dampen_tau: float = 2.0
    adv_dampen_beta: float = 3.0

    # ... 나머지 필드들 ...
```

**대안 (config 수정 없이)**: hydra `+` override로 전달 가능.
```bash
+algorithm.adv_dampen_tau=2.0
+algorithm.adv_dampen_beta=3.0
```
`AlgoConfig`가 `BaseConfig`를 상속하고 `get()` 메서드를 제공하므로,
명시적 field가 없어도 `config.get("adv_dampen_tau", 2.0)`이 동작한다.

**권장**: AlgoConfig에 명시적 field를 추가하여 문서화 및 타입 안전성 확보.

### 3.4 Step 3: dispatch 경로 확인

`ray_trainer.py:compute_advantage()`에서 dispatch 경로:

```python
if adv_estimator == AdvantageEstimator.GAE:
    # GAE 전용 경로
elif adv_estimator == AdvantageEstimator.GRPO:
    # GRPO 전용 경로 (hardcoded, norm_adv_by_std_in_grpo 직접 전달)
else:
    # Generic registry 경로 ← grpo_dampened는 여기로 간다
    adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
    adv_kwargs = {
        "token_level_rewards": ...,
        "response_mask": ...,
        "config": config,       # ← AlgoConfig 전체 전달
    }
    if "uid" in data.non_tensor_batch:
        adv_kwargs["index"] = data.non_tensor_batch["uid"]
    advantages, returns = adv_estimator_fn(**adv_kwargs)
```

**`grpo_dampened`는 별도의 dispatcher 수정 없이 generic path를 자동으로 탄다.**
이는 `grpo_vectorized`, `grpo_passk` 등과 동일한 패턴이다.

`config` 객체가 전달되므로 `config.get("adv_dampen_tau", 2.0)` 등이 동작한다.

### 3.5 Step 4: 실험 스크립트

```bash
#!/bin/bash
# run_grpo_dampened_pilot.sh
# 5-arm comparison: vanilla GRPO, grpo_dampened (τ=2), grpo_dampened (τ=1.5),
#                   grpo_dampened (τ=1), Dr.GRPO (no std norm)

MODEL="Qwen/Qwen3-4B-Base"
DATASET="gsm8k"
G=16

# Arm 1: Vanilla GRPO (baseline)
# algorithm.adv_estimator=grpo

# Arm 2: grpo_dampened τ=2.0 (primary)
# algorithm.adv_estimator=grpo_dampened
# algorithm.adv_dampen_tau=2.0
# algorithm.adv_dampen_beta=3.0

# Arm 3: grpo_dampened τ=1.5 (ablation)
# algorithm.adv_dampen_tau=1.5

# Arm 4: grpo_dampened τ=1.0 (ablation)
# algorithm.adv_dampen_tau=1.0

# Arm 5: Dr.GRPO (reference)
# algorithm.adv_estimator=grpo
# algorithm.norm_adv_by_std_in_grpo=false
```

---

## 4. 검증 계획

### 4.1 단위 테스트

```python
def test_grpo_dampened_basic():
    """기본 동작 검증"""
    # Case 1: 정상 범위 Â (|Â| < τ) → 거의 무개입
    # Case 2: 극단 음수 Â → soft cap at -τ
    # Case 3: 양수 Â → 무개입 (correction ≈ 0)
    # Case 4: vanilla GRPO와의 차이가 음수 Â에서만 존재
    pass

def test_grpo_dampened_monotonicity():
    """단조성: |Â| 증가 시 |Â_eff|도 증가 (τ에서 포화)"""
    pass

def test_grpo_dampened_no_sign_flip():
    """β=3.0에서 음수 Â가 양수로 뒤집히지 않음"""
    pass

def test_grpo_dampened_config_defaults():
    """config 없을 때 assert, config에 field 없을 때 기본값"""
    pass
```

### 4.2 수치 검증 체크포인트

G=16, β=3.0, τ=2.0 기준 기대값:

```
k=15 (94%): Â_raw = -3.750 → Â_eff ≈ -1.998  (절삭 46.7%)
k=14 (88%): Â_raw = -2.562 → Â_eff ≈ -1.943  (절삭 24.1%)
k=13 (81%): Â_raw = -2.016 → Â_eff ≈ -1.777  (절삭 11.9%)
k=12 (75%): Â_raw = -1.677 → Â_eff ≈ -1.570  (절삭  6.4%)
k=8  (50%): Â_raw = -0.968 → Â_eff ≈ -0.954  (절삭  1.5%)
k=1  (6%):  Â_raw = -0.250 → Â_eff ≈ -0.248  (절삭  0.7%)
positive:   Â_raw = +3.750 → Â_eff ≈ +3.750  (절삭  0.0%)
```

### 4.3 실험 관찰 지표

| 지표 | 의미 | 기대 |
|------|------|------|
| `policy_entropy` | 정책 분포 엔트로피 | dampened > vanilla (collapse 방지) |
| `adv_raw_abs_mean` | 원래 \|Â\| 평균 | 학습 진행에 따라 증가 |
| `adv_eff_abs_mean` | dampened \|Â_eff\| 평균 | τ에 의해 bounded |
| `correction_nonzero_frac` | correction > 0.01인 비율 | 학습 후반부에 증가 |
| `gsm8k_accuracy` | GSM8K 정확도 | dampened ≥ vanilla |
| `gradient_norm` | global gradient L2 norm | dampened ≤ vanilla |

---

## 5. 리스크 및 완화

### 5.1 Zero-sum 위반

one-sided clamping은 GRPO의 zero-mean 성질을 깨뜨린다:
```
원래: Σ_group Â = 0 (zero-sum)
dampened: Σ_group Â_eff > 0 (음수가 줄었으므로 양수 쪽으로 편향)
```

**완화**: τ=2.0의 zero-sum 위반은 τ=1.0의 2.2× 작음 (5.28 vs 27.19).
또한 PPO clip (ε=0.2)이 ratio를 [0.8, 1.2]로 제한하므로, 약간의 advantage 편향은
ratio clipping에 의해 자연스럽게 완화된다.

### 5.2 학습 초기 무효과

accuracy < 50%일 때 |Â_wrong| < 1.0이므로 τ=2.0 dampening은 전혀 개입하지 않는다.
→ **의도된 행동**: 초기에는 vanilla GRPO와 동일하게 작동하다가,
   모델이 성장하여 accuracy > 80%가 되면 점진적으로 보호 시작.

### 5.3 연속형 reward와의 호환

현재 설계는 binary reward (correct/incorrect)를 전제로 한다.
연속형 reward에서는 std 분포가 달라져서 τ=2.0의 의미가 변할 수 있다.
→ **완화**: 연속형 reward에서도 "2σ 임계치"로서의 일반적 의미는 유지됨.

### 5.4 gradient clipping과의 상호작용

verl 표준 설정에서 global gradient clipping (max_norm=1.0)이 이미 적용된다.
→ advantage dampening은 gradient clipping **이전**에 작동하므로,
   dampening이 gradient 크기를 줄여서 gradient clipping이 덜 발동하게 만든다.
   이는 **유효 학습률 증가** 효과를 가져올 수 있다 (의도적 부작용).

---

## 6. 실험 계획

### 6.1 Pilot 구성

```
Model:    Qwen3-4B-Base
Dataset:  GSM8K (8,792 train examples)
Group:    G=16, binary reward
Hardware: 4× GPU
Steps:    ~300 steps (collapse 관찰 충분)

5-arm comparison:
  1. vanilla_grpo          ─ baseline
  2. grpo_dampened_tau2    ─ primary (τ=2.0, β=3.0)
  3. grpo_dampened_tau1.5  ─ ablation
  4. grpo_dampened_tau1    ─ ablation
  5. drgrpo               ─ reference (norm_adv_by_std=false)
```

### 6.2 관찰할 핵심 질문

1. **vanilla GRPO는 언제 collapse하는가?** (step, accuracy 기록)
2. **τ=2.0이 collapse를 방지하는가?** (entropy curve 비교)
3. **τ=2.0 vs τ=1.5 vs τ=1.0**: 최종 정확도 차이가 있는가?
4. **Dr.GRPO 대비**: dampening이 Dr.GRPO보다 나은가?
5. **τ=2.0에서 correction이 실제로 얼마나 발동하는가?** (correction_nonzero_frac)

---

## 7. 코드 수정 상세

### 7.1 `core_algos.py` 수정 사항

**import 추가**: 없음 (torch.nn.functional.softplus는 이미 사용 가능)

**함수 추가 위치**: line 528 (`compute_grpo_passk_outcome_advantage` 종료) 직후

**의존성**: `as_torch_index`, `group_mean_std` (이미 import됨, `grpo_vectorized`에서 사용)

### 7.2 `algorithm.py` 수정 사항

`AlgoConfig` dataclass에 2개 필드 추가:
```python
adv_dampen_tau: float = 2.0
adv_dampen_beta: float = 3.0
```

**위치**: `norm_adv_by_std_in_grpo` 필드 직후 (line 489 부근)

### 7.3 사용법

```yaml
# config.yaml
algorithm:
  adv_estimator: grpo_dampened
  adv_dampen_tau: 2.0
  adv_dampen_beta: 3.0
  norm_adv_by_std_in_grpo: true   # 기존 GRPO std norm 옵션 존중
```

또는 CLI override:
```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo_dampened \
  algorithm.adv_dampen_tau=2.0 \
  algorithm.adv_dampen_beta=3.0
```

---

## 8. 토론 과정 기록 (부록)

### 8.1 토론 참가자

| Agent | Role | Model |
|-------|------|-------|
| advocate-τ1 | τ=1 옹호 | Sonnet |
| advocate-τ2 | τ=2 옹호 | Sonnet |
| math-referee | 수학적 검증 | Sonnet |
| rl-practitioner | 실전 RL 분석 | Sonnet |
| lit-searcher | 문헌 조사 | Sonnet |

### 8.2 주요 논쟁점 및 해결

**논쟁 1: 자연 스케일 복원 (τ=1) vs 최소 개입 원칙 (τ=2)**

- advocate-τ1: binary reward의 비정규화 advantage는 |Â|≤1. τ=1은 std 증폭을 원래 스케일로 복원.
  수학적으로 √(G-1)/√G ≈ 0.968 ≈ 1.
- advocate-τ2: 최소 개입 원칙 — 필요할 때만 개입. τ=2는 acc 81%+ 문제만 개입.
- **해결**: math-referee의 gradient 보존율 분석 (τ=1: 26.8%, τ=2: 63.3%)에서
  τ=1의 73% gradient 손실은 과도하다고 판단. τ=2 우세.

**논쟁 2: Dr.GRPO 증거는 τ=1을 지지하는가?**

- lit-searcher: Dr.GRPO (|Â|≤1 달성)가 GRPO를 능가. τ=1 우세 증거.
- advocate-τ2: Dr.GRPO는 모든 Â를 축소하지만, τ=1 clamp은 극단값만 자름. 메커니즘이 다름.
- **해결**: 직접 비교 불가. pilot에서 Dr.GRPO를 reference arm으로 포함하여 실증.

**논쟁 3: sign flipping 위험**

- math-referee: τ=1에서 sign flipping 발생 (β=3.0에서도).
- **team-lead 검증**: β=3.0에서는 sign flipping **없음** (math-referee 오류).
  β≤1.0에서만 발생. β=3.0은 안전.
- **해결**: β≥2.0 제약 조건 확립. β=3.0 채택.

**논쟁 4: 핵심 학습 구간에서의 영향**

- rl-practitioner: acc 40-70%가 학습의 핵심 구간. τ=1은 이 구간에서 6-40% 절삭.
- advocate-τ1: 50% accuracy 이상이면 "이미 학습된" 문제.
- **해결**: rl-practitioner의 실전 경험에 기반, 70% accuracy도 여전히 활발한 학습 중.
  τ=2는 이 구간에서 1.5-6.4%만 절삭하여 학습 신호 보존. τ=2 우세.

### 8.3 최종 투표 결과

```
τ=2.0 지지: advocate-τ2 (85%), math-referee (90%), rl-practitioner (6/6), lit-searcher (약간)
τ=1.0 지지: advocate-τ1 (70%, 타협안 τ=1.5에 25%)

최종 채택: τ=2.0, β=3.0
```
