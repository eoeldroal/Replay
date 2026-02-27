# GCISPO 기획안 — GSPO + CISPO 결합 정책 손실

> **목표**: GSPO의 시퀀스 수준 IS ratio 계산과 CISPO의 gradient-보존 클리핑을 결합하여,
> 토큰 수준 노이즈를 줄이면서도 clip boundary에서 gradient가 죽지 않는 정책 손실 함수를 구현한다.

> **상태**: 구현 완료. `core_algos.py`에 `@register_policy_loss("gcispo")` 등록됨.

---

## 1. 동기: 왜 결합하는가

### 1.1 기존 알고리즘의 한계

| 알고리즘 | 강점 | 약점 |
|----------|------|------|
| **Vanilla PPO** | 검증된 안정성, 듀얼 클리핑 | clip boundary에서 gradient=0, 토큰별 ratio 노이즈 |
| **GSPO** | 시퀀스 수준 ratio로 노이즈 감소 | 여전히 `max(unclipped, clipped)` 사용 → gradient 단절 |
| **CISPO** | `sg(clip(r))·log_prob`로 gradient 항상 살림 | 토큰별 ratio 사용 → 개별 토큰 노이즈에 취약 |

### 1.2 결합의 핵심 관찰

두 알고리즘의 혁신이 **서로 직교하는 축**에 있다:

```
축 1 (ratio 계산):     GSPO  — 시퀀스 수준 IS 기하평균
축 2 (gradient 전파):   CISPO — sg(clip) + log_prob 직접 최적화
```

따라서 자연스러운 결합이 가능하다:
- **GSPO에서**: ratio를 어떻게 계산할 것인가 (시퀀스 평균)
- **CISPO에서**: 클리핑 후 gradient를 어떻게 보존할 것인가 (stop-gradient)

---

## 2. 알고리즘 설계

### 2.1 수식 정의

**Step 1: 토큰별 log ratio**
```
neg_kl_t = log π_θ(y_t | x, y_{<t}) − log π_θ_old(y_t | x, y_{<t})
```
> CISPO는 여기에 clamp(min=-20, max=20)을 적용하지만, GSPO는 적용하지 않음.
> GCISPO는 GSPO를 따라 미적용 — 시퀀스 평균화가 안정화 효과를 제공하기 때문.

**Step 2: 시퀀스 수준 IS ratio (GSPO에서 가져옴)**
```
log s_i = (1/|y_i|) × Σ_t neg_kl_t × mask_t

s_i = exp(log s_i)    ← 시퀀스 전체의 정책 변화를 하나의 스칼라로 표현
```
> log 공간에서의 산술 평균. exp()를 씌우면 토큰별 ratio의 기하평균이 됨.

**Step 3: 시퀀스+토큰 결합 ratio (GSPO에서 가져옴)**
```
log r_{i,t} = sg[log s_i] + log π_θ(y_t) − sg[log π_θ(y_t)]

r_{i,t} = exp(clamp(log r_{i,t}, max=10))
```
> `log π_θ − sg[log π_θ]`: 값은 항상 0이지만 gradient는 ∂log_prob/∂θ를 보존하는 트릭.
> `.detach()`는 현재 값을 상수로 고정하므로, `log_prob - log_prob.detach()`는
> 수학의 `x - x = 0`이 아니라 `f(θ) - f(θ₀)` (현재 θ에서 값이 0이지만 기울기는 살아있음).
>
> **GCISPO에서의 역할**: 이 gradient 트릭 자체는 Step 4의 `.detach()`에 의해 무효화됨.
> 실질적 역할은 **값(value) 계산** — `combined_ratio = s_i`를 만들어 클리핑 판단 기준으로 사용.
> 코드 일관성을 위해 GSPO 원본 패턴을 그대로 유지.

**Step 4: 비대칭 클리핑 + stop-gradient (CISPO에서 가져옴)**
```
r̃_{i,t} = clamp(r_{i,t}, 1 − ε_low, 1 + ε_high)

r̃_{i,t}^{sg} = sg[r̃_{i,t}]    ← 전체 ratio를 상수로 취급
```
> CISPO는 토큰별 ratio를 클리핑하지만, GCISPO는 **시퀀스+토큰 결합 ratio**를 클리핑.

**Step 5: CISPO 스타일 손실**
```
L_{i,t} = −r̃_{i,t}^{sg} × A_{i,t} × log π_θ(y_t)
```
> gradient는 오직 log π_θ를 통해서만 흐른다. clip 여부와 무관하게 gradient가 살아 있다.
> log_prob의 **값**(-3.5 같은)은 gradient 계산에 관여하지 않음 — 미분값(∂log_prob/∂θ)만 사용됨.

**Step 6: 시퀀스 균등 집계 (GSPO에서 가져옴)**
```
L = (1/B) × Σ_i [ (1/|y_i|) × Σ_t L_{i,t} × mask_{i,t} ]
```
> seq-mean-token-mean: 긴 응답이 짧은 응답보다 과도한 영향을 주지 않도록.

**Step 7: KL 정규화 (CISPO에서 가져옴)**
```
L_total = L + λ_kl × KL_loss(π_θ || π_ref)
```
> 비대칭 클리핑(lower 무제한)의 안전장치. low_var_kl 사용.

### 2.2 Gradient 분석

| 조건 | ratio 상태 | gradient |
|------|-----------|----------|
| ratio ∈ [1−ε_low, 1+ε_high] | 클리핑 안 됨 | ∂L/∂θ = −r_{i,t} × A × ∂log_prob/∂θ |
| ratio > 1+ε_high | 상한 클리핑 | ∂L/∂θ = −(1+ε_high) × A × ∂log_prob/∂θ |
| ratio < 1−ε_low | 하한 클리핑 | ∂L/∂θ = −(1−ε_low) × A × ∂log_prob/∂θ |

**핵심**: 세 경우 모두 gradient ≠ 0. clip boundary에서도 학습이 계속된다.
비대칭 설정(ε_low=10)에서 하한 클리핑은 사실상 발생하지 않으므로,
확률 감소 방향은 자유롭고 증가 방향만 제약받는다.

### 2.3 GSPO/CISPO/GCISPO 비교

```
          GSPO                          CISPO                        GCISPO
          ────                          ─────                        ──────
ratio:    시퀀스+토큰 결합              토큰별                       시퀀스+토큰 결합 (GSPO)
clip:     max(unclipped, clipped)       sg(clipped) · log_prob       sg(clipped) · log_prob (CISPO)
clip범위: 대칭 타이트 (3e-4/4e-4)       비대칭 (low=10, high=0.2)    비대칭 (low=10, high=조율)
gradient: clip 밖에서 죽음              항상 살아있음                  항상 살아있음 (CISPO)
집계:     seq-mean-token-mean           token-mean                    seq-mean-token-mean (GSPO)
KL:       없음                          low_var_kl (0.001)            low_var_kl (CISPO)
```

---

## 3. 하이퍼파라미터 설계

### 3.1 클리핑 범위

| 파라미터 | GSPO 원본 | CISPO 원본 | GCISPO 초기값 | 근거 |
|----------|-----------|------------|--------------|------|
| `clip_ratio_low` | 0.0003 | 10 | **10** | CISPO 철학: 확률 감소 무제한 허용 |
| `clip_ratio_high` | 0.0004 | 0.2 | **0.05** | 시퀀스 ratio는 토큰 평균이라 분산 작음. GSPO(0.0004)보다 넓되 CISPO(0.2)보다 좁게 |

> `clip_ratio_high`가 핵심 조율 대상. 파일럿에서 clipfrac_upper를 관찰하여 조정.
> - clipfrac_upper > 50%: 너무 좁음 → 올리기
> - clipfrac_upper < 5%: 너무 넓음 → 내리기
> - 10~30% 범위가 이상적

### 3.2 KL 정규화

| 파라미터 | 값 | 근거 |
|----------|---|------|
| `use_kl_loss` | True | 비대칭 클리핑의 안전장치로 필수 |
| `kl_loss_coef` | 0.001 | CISPO 논문 기본값. 파일럿 후 조정 가능 |
| `kl_loss_type` | low_var_kl | CISPO 논문 추천. 일반 KL보다 분산이 낮아 안정적 |

### 3.3 기타

| 파라미터 | 값 | 근거 |
|----------|---|------|
| `loss_agg_mode` | seq-mean-token-mean | GSPO 철학: 시퀀스 길이 독립적 가중 |
| `adv_estimator` | grpo | 기존과 동일 (critic 없음) |
| `entropy_coeff` | 0 | 기존과 동일 |

---

## 4. 진단 로깅 설계

### 4.1 설계 원칙

- **자체 완결**: `core_algos.py` 내부에서 완전히 처리. 외부 파이프라인 무변경.
- **기존 무간섭**: 기존 `pg_metrics`(wandb 스칼라)와 별도 경로. 겹치지 않음.
- **제로 오버헤드**: `policy_diag_dir` 미설정 시 `record()` 첫 줄에서 즉시 반환.
- **원본 텐서 저장**: 평균/표준편차가 아닌 전체 리스트를 그대로 보존.
- **CPU 메모리 사용**: `record()` 내부에서 `.detach().cpu()`로 GPU→CPU 복사. 버퍼는 전부 CPU RAM에 존재.
- **세션 ID 기반 파일명**: 인스턴스 생성 시각(`YYYYMMDDTHHmmss`)을 파일명 prefix로 사용.
  재개/재실행 시 파일 충돌 없음. 같은 디렉토리에 다른 실험 데이터가 있어도 안전.
- **통합 호환**: vanilla/GSPO/CISPO/GCISPO 4개 알고리즘 모두에 동일한 로거 적용.

### 4.2 두 가지 로깅 경로

```
경로 1 (기존): core_algos.py → pg_metrics dict → dp_actor.py → ray_trainer.py → wandb
               (스칼라: clipfrac, ppo_kl, clipfrac_lower)

경로 2 (신규): core_algos.py → _PolicyDiagWriter → 디스크 (.pt 파일)
               (전체 텐서: token_ratios, seq_ratios, mask, clipped)
```

### 4.3 활성화 방법

`ActorConfig`의 config 인자로 제어 (환경변수가 아닌 실행 인자 방식).

**Hydra `+` prefix 필요**: `policy_diag_dir`와 `policy_diag_flush_freq`는 Python 데이터클래스
(`ActorConfig`)에는 정의되어 있지만, Hydra가 참조하는 기본 YAML 스키마에는 존재하지 않는다.
Hydra의 struct 모드는 YAML에 없는 키의 CLI 오버라이드를 거부하므로,
반드시 `+` prefix를 붙여 "새 키 추가"임을 명시해야 한다.

```
CLI 인자 전달
    │
    ▼
Hydra: YAML 기본 config + CLI 오버라이드 병합
    │
    ├─ YAML에 있는 키: 오버라이드 OK  (예: clip_ratio_high=0.05)
    ├─ YAML에 없는 키 + prefix 없음: ❌ struct 위반 에러
    └─ YAML에 없는 키 + "+" prefix: ✅ 새 키 추가 허용
    │
    ▼
DictConfig → ActorConfig 데이터클래스 변환 (여기서 policy_diag_dir 필드가 매칭됨)
```

```bash
# 실행 스크립트에서 (+ prefix 필수):
+actor_rollout_ref.actor.policy_diag_dir=/path/to/diag \
+actor_rollout_ref.actor.policy_diag_flush_freq=256 \
```

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `policy_diag_dir` | `None` | 저장 경로. None이면 비활성 |
| `policy_diag_flush_freq` | `32` | flush 주기 (micro-batch `record()` 호출 횟수 단위) |

> `policy_diag_flush_freq` 계산 예시: 현재 설정에서 매 training step당
> (mini_batch / micro_batch_per_gpu / n_gpus) = (16 / 2 / 4) = 2회의 `record()` 호출.
> flush_freq=256이면 약 128 step마다 1회 flush.

### 4.4 데이터 형식

`.pt` 파일 (PyTorch 텐서 딕셔너리):

```python
{
    "token_ratios": Tensor(N, seq_len)  float16   # 토큰별 IS ratio (π_θ / π_θ_old)
    "seq_ratios":   Tensor(N,)          float32   # 시퀀스별 IS ratio (기하평균)
    "mask":         Tensor(N, seq_len)  bool      # 유효 토큰 마스크
    "clipped":      Tensor(N, seq_len)  bool      # 클리핑 발생 여부 (선택 필드)
}
# N = flush 주기 동안 누적된 샘플 수
```

알고리즘별 필드 차이:

| 알고리즘 | token_ratios | seq_ratios | clipped 의미 |
|----------|-------------|-----------|--------------|
| vanilla | 토큰별 ratio | 자동계산 (기하평균) | `max(unclipped, clipped)`에서 clipped 선택됨 |
| gspo | 토큰별 ratio | 명시적 기하평균 | 위와 동일 |
| cispo | 토큰별 ratio | 자동계산 (기하평균) | `clamp`에 의해 잘린 토큰 |
| gcispo | 토큰별 ratio | 명시적 기하평균 | `clamp`에 의해 잘린 토큰 |

### 4.5 저장 구조

파일명 형식: `{session_id}_flush{count:04d}_rank{rank}.pt`

- `session_id`: `_PolicyDiagWriter` 인스턴스 생성 시각 (`YYYYMMDDTHHmmss`)
- `count`: 해당 세션 내 flush 순번 (1부터 시작)
- `rank`: 분산 학습의 rank 번호

```
logs/GCISPO_diag/
│
│  ── 1차 학습 (20260213T194023) ──
├── 20260213T194023_flush0001_rank0.pt
├── 20260213T194023_flush0001_rank1.pt
├── 20260213T194023_flush0001_rank2.pt
├── 20260213T194023_flush0001_rank3.pt
├── 20260213T194023_flush0002_rank0.pt
├── ...
│
│  ── 체크포인트 재개 (20260214T083015) ──
├── 20260214T083015_flush0001_rank0.pt
├── 20260214T083015_flush0001_rank1.pt
├── ...
```

> 세션 ID 기반이므로 재개/재실행/다른 실험 모두 파일 충돌이 발생하지 않는다.

### 4.6 종료 방식별 버퍼 보존

| 종료 방식 | `atexit` flush | 미저장 버퍼 |
|-----------|:-:|---|
| 정상 종료 / `Ctrl+C` / `kill` | O | 없음 |
| `kill -9` (SIGKILL) | X | 최대 flush_freq-1 회분 손실 |

### 4.7 JSONL 대응 관계

데이터 순서가 보존됨 (`DataProto.split()`이 순차 슬라이싱, 셔플 없음 — `protocol.py:925` 확인됨):

```
{session}_flush0001_rank0.pt의 sample[k]  ↔  rollout JSONL의 line[k]
{session}_flush0001_rank1.pt의 sample[k]  ↔  rollout JSONL의 line[64 + k]
{session}_flush0001_rank2.pt의 sample[k]  ↔  rollout JSONL의 line[128 + k]
{session}_flush0001_rank3.pt의 sample[k]  ↔  rollout JSONL의 line[192 + k]
```

---

## 5. 구현 상태

### 5.1 수정된 파일

| 파일 | 변경 내용 | 상태 |
|------|----------|------|
| `verl/trainer/ppo/core_algos.py` | `_PolicyDiagWriter` 클래스 + `compute_policy_loss_gcispo` 등록 + vanilla/gspo/cispo에 진단 hook | **완료** |
| `verl/workers/config/actor.py` | `policy_diag_dir`, `policy_diag_flush_freq` 필드 추가 | **완료** |
| `RL_side_2/run_qwen3-8b_4_gcispo.sh` | GCISPO 실행 스크립트 | **완료** |

> **dp_actor.py, ray_trainer.py 등 기존 파이프라인 코드는 수정하지 않음.**

### 5.2 core_algos.py 구현 구조

```python
# ── 모듈 상단 (line 90~) ──
class _PolicyDiagWriter:
    """통합 진단 로거. config.policy_diag_dir가 None이면 비활성."""
    def __init__(self, config=None):
        self.dir = getattr(config, "policy_diag_dir", None)
        self.freq = getattr(config, "policy_diag_flush_freq", 32)
        self._session_id = datetime.now().strftime("%Y%m%dT%H%M%S")  # 세션별 고유 ID
        ...
    def record(self, *, token_ratios, mask, seq_ratios=None, clipped=None): ...
    def _flush(self):
        # 파일명: {session_id}_flush{count}_rank{rank}.pt
        ...

# ── 각 손실 함수 내부 (lazy init) ──
@register_policy_loss("gcispo")
def compute_policy_loss_gcispo(..., config=None):
    ...
    if not hasattr(compute_policy_loss_gcispo, "_diag"):
        compute_policy_loss_gcispo._diag = _PolicyDiagWriter(config)
    compute_policy_loss_gcispo._diag.record(...)

# vanilla, gspo, cispo에도 동일한 패턴으로 hook 적용됨
```

### 5.3 실행 스크립트 (`run_qwen3-8b_4_gcispo.sh`)

GSPO / CISPO / GCISPO 하이퍼파라미터 통합 비교:

```
┌───────────────────┬──────────────┬──────────────┬──────────────┬──────────────────┐
│ Parameter         │ GSPO         │ CISPO        │ GCISPO       │ Note             │
├───────────────────┼──────────────┼──────────────┼──────────────┼──────────────────┤
│ loss_mode         │ gspo         │ cispo        │ gcispo       │                  │
│ loss_agg_mode     │ seq-mean-..  │ token-mean   │ seq-mean-..  │ from GSPO        │
│ clip_ratio_low    │ 0.0003       │ 10           │ 10           │ from CISPO       │
│ clip_ratio_high   │ 0.0004       │ 0.2          │ 0.05         │ tuned for seq IS │
│ clip_ratio_c      │ 10.0         │ --           │ --           │ GSPO only        │
│ use_kl_loss       │ False        │ True         │ True         │ from CISPO       │
│ kl_loss_coef      │ 0.0          │ 0.001        │ 0.001        │ from CISPO       │
│ kl_loss_type      │ --           │ low_var_kl   │ low_var_kl   │ from CISPO       │
└───────────────────┴──────────────┴──────────────┴──────────────┴──────────────────┘
```

GCISPO 전용 추가 인자 (`+` prefix 필수):

```bash
+actor_rollout_ref.actor.policy_diag_dir=/.../GCISPO_diag      # 진단 활성화
+actor_rollout_ref.actor.policy_diag_flush_freq=256             # flush 주기
```

---

## 6. 파일럿 분석 계획

### 6.1 Wandb 모니터링 (실시간)

학습 중 확인할 지표:

| 지표 | 정상 범위 | 이상 신호 |
|------|----------|----------|
| `actor/ppo_kl` | 0.001 ~ 0.05 | > 0.1: 정책 발산 위험 |
| `actor/pg_clipfrac` | 10~30% | > 50%: clip_ratio_high 너무 좁음 |
| 학습 reward (평균) | 상승 추세 | 급락: 정책 붕괴 |

### 6.2 사후 분석 (진단 .pt 파일)

#### 분석 1: 시퀀스 IS ratio 분포 변화

```python
import torch, matplotlib.pyplot as plt

early = torch.load("logs/GCISPO_diag/step_0005_rank0.pt")
late  = torch.load("logs/GCISPO_diag/step_0100_rank0.pt")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(early["seq_ratios"].numpy(), bins=50, alpha=0.7)
axes[0].set_title("Step 5: Seq IS Ratio Distribution")
axes[1].hist(late["seq_ratios"].numpy(), bins=50, alpha=0.7)
axes[1].set_title("Step 100: Seq IS Ratio Distribution")
plt.savefig("seq_ratio_evolution.png")
```

> 학습이 진행될수록 ratio 분포가 넓어지면 정상.
> 분포가 한쪽으로 치우치면 정책 붕괴 징후.

#### 분석 2: 토큰 수준 IS ratio — 정답/오답 비교

```python
import json

diag = torch.load("logs/GCISPO_diag/step_0050_rank0.pt")
with open("logs/GCISPO_pilot_rollout/50.jsonl") as f:
    rollout = [json.loads(line) for line in f]

correct_ratios, wrong_ratios = [], []
for i in range(len(rollout)):
    if i >= diag["mask"].shape[0]:
        break
    mask = diag["mask"][i]
    ratios = diag["token_ratios"][i][mask].float()
    if rollout[i]["score"] > 0:
        correct_ratios.append(ratios)
    else:
        wrong_ratios.append(ratios)

correct_all = torch.cat(correct_ratios)
wrong_all = torch.cat(wrong_ratios)

print(f"정답 응답 토큰: mean={correct_all.mean():.4f}, std={correct_all.std():.4f}")
print(f"오답 응답 토큰: mean={wrong_all.mean():.4f}, std={wrong_all.std():.4f}")
```

> 정답 응답의 ratio > 1 (확률 증가), 오답 응답의 ratio < 1 (확률 감소)이면 정상.

#### 분석 3: 특정 샘플의 토큰별 IS ratio 시각화

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/.../Qwen3-8B")

i = 5  # 분석할 샘플
mask = diag["mask"][i]
ratios = diag["token_ratios"][i][mask].float()
response_text = rollout[i]["output"]
token_ids = tokenizer.encode(response_text, add_special_tokens=False)
tokens = [tokenizer.decode([tid]) for tid in token_ids]

# ratio가 가장 높은/낮은 토큰 Top-10
top10 = ratios.topk(10)
bot10_vals, bot10_idx = ratios.topk(10, largest=False)

print(f"Score: {rollout[i]['score']}, Seq ratio: {diag['seq_ratios'][i]:.4f}")
print("\n=== 확률 증가 Top-10 ===")
for idx, val in zip(top10.indices, top10.values):
    pos = idx.item()
    if pos < len(tokens):
        context = "".join(tokens[max(0,pos-3):pos+1])
        print(f"  pos={pos:4d}  ratio={val:.4f}  token='{tokens[pos]}'  ctx='...{context}'")

print("\n=== 확률 감소 Top-10 ===")
for idx, val in zip(bot10_idx, bot10_vals):
    pos = idx.item()
    if pos < len(tokens):
        context = "".join(tokens[max(0,pos-3):pos+1])
        print(f"  pos={pos:4d}  ratio={val:.4f}  token='{tokens[pos]}'  ctx='...{context}'")
```

### 6.3 알고리즘 간 비교

다른 알고리즘 스크립트에도 진단을 활성화하여 동일 형식의 .pt 파일을 생성할 수 있음:

```bash
# GRPO baseline에 진단 추가 (+ prefix 필수):
+actor_rollout_ref.actor.policy_diag_dir=/.../GRPO_diag \

# GSPO에 진단 추가:
+actor_rollout_ref.actor.policy_diag_dir=/.../GSPO_diag \

# CISPO에 진단 추가:
+actor_rollout_ref.actor.policy_diag_dir=/.../CISPO_diag \
```

### 6.4 파일럿 결과에 따른 조율 방향

| 관찰 | 진단 | 조치 |
|------|------|------|
| clipfrac > 50% | clip_ratio_high 너무 좁음 | clip_ratio_high 올리기 (0.05 → 0.1) |
| clipfrac < 5% | clip_ratio_high 너무 넓음 | clip_ratio_high 내리기 (0.05 → 0.02) |
| seq_ratio 분산 급증 | 정책 불안정 | kl_loss_coef 올리기 (0.001 → 0.005) |
| 정답/오답 ratio 차이 없음 | 학습 신호 약함 | lr 올리기 또는 clip_ratio_high 넓히기 |
| 정답 토큰 ratio > 1.5 | 과도한 강화 | kl_loss_coef 올리기 |
| reward 정체 | 탐색 부족 | entropy_coeff > 0 시도 |

---

## 7. 참고 문헌

- GSPO 논문: https://arxiv.org/pdf/2507.18071
- CISPO 논문: https://arxiv.org/pdf/2506.13585
- DAPO 논문: https://arxiv.org/abs/2503.14476
- verl 코드: `verl/trainer/ppo/core_algos.py`

---

## 8. 핵심 파일 참조

| 컴포넌트 | 파일 | 상태 |
|----------|------|------|
| GCISPO 손실 함수 + 통합 진단 로거 | `verl/trainer/ppo/core_algos.py` | 완료 |
| ActorConfig 진단 필드 | `verl/workers/config/actor.py` | 완료 |
| GCISPO 실행 스크립트 | `RL_side_2/run_qwen3-8b_4_gcispo.sh` | 완료 |
| GRPO 베이스라인 | `RL_side_2/run_qwen3-8b_4.sh` | 기존 |
| GSPO 비교군 | `RL_side_2/run_qwen3-8b_4_gspo.sh` | 기존 |
| CISPO 비교군 | `RL_side_2/run_qwen3-8b_4_cispo.sh` | 기존 |
| 보상 함수 | `verl/utils/reward_score/math_dapo.py` | 기존 |
| Actor 학습 루프 | `verl/workers/actor/dp_actor.py` | 수정 없음 |
| 훈련 루프 | `verl/trainer/ppo/ray_trainer.py` | 수정 없음 |
| 기존 분석 문서 | `RL_side_2/AGENTS.md` | 기존 |
