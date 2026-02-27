# AGENTS.md — RL_side_2 실험 환경 가이드

> 이 문서는 GCISPO(GSPO + CISPO 결합) 프로젝트의 실험 환경, 알고리즘 이론, 코드베이스 함정을
> 코드 수준에서 분석한 결과물이다. 다른 에이전트나 협업자가 이 문서를 통해 빠르게 온보딩할 수 있도록 구성했다.
>
> **빠른 시작**: 새로 합류했다면 §13(알고리즘 이론) → §14(현재 실험) → §15~16(실수 방지) 순서로 읽을 것.
> **로그 분석**: §17(로그 구조 & 분석 가이드)를 참조. 경로 복붙은 §17.8, 분석 코드는 §17.10.
> **코드 탐색**: §15.5(빠른 파일 탐색 가이드)를 먼저 참조.
> **설계 근거**: `GCISPO_design.md`를 참조. verl 엔진의 일반적 구조는 `verl_engine_deep_dive.md`를 참조.
>
> **이하 §0~12는 초기 탐색 기록(Qwen3-8B 트랙 + Qwen2.5-3B-Instruct 초기 실험)이다.**
> **현재 활성 실험은 §14를 참조하라.**

---

## 0. 빠른 요약: 두 스크립트의 차이

| 항목 | GRPO 베이스라인 (`_4.sh`) | GSPO 변형 (`_4_gspo.sh`) |
|------|--------------------------|--------------------------|
| 진입점 | `python3 -m verl.trainer.main_ppo` | 동일 |
| 어드밴티지 추정 | `algorithm.adv_estimator=grpo` | 동일 (GRPO) |
| **정책 손실** | `vanilla` (기본값) | `gspo` |
| **손실 집계** | `token-mean` (기본값) | `seq-mean-token-mean` |
| **클리핑** | `clip_ratio_low/high=0.2` (기본값) | `0.0003 / 0.0004` |
| `clip_ratio_c` | `3.0` (기본값) | `10.0` (미사용) |
| KL 제어 | `use_kl_loss=False` | `use_kl_loss=False`, `kl_loss_coef=0.0` |
| 실험 이름 | `Baseline.test` | `GSPO.test` |

> **핵심**: 둘 다 GRPO 어드밴티지를 쓰지만, **정책 손실 함수**만 다르다.
> GSPO는 시퀀스 수준 importance ratio를 도입하여 더 안정적인 학습을 목표로 한다.

---

## 1. 훈련 루프 진입점 (`main_ppo.py`)

### 1.1 설정 파싱

```
verl/trainer/main_ppo.py
  → @hydra.main(config_name="ppo_trainer")
  → run_ppo(config)
  → Ray 클러스터 초기화
  → TaskRunner.remote().run(config)
  → RayPPOTrainer.fit()
```

- **Hydra + OmegaConf** 기반: CLI 인자가 YAML 설정을 덮어씀
- `+key=value` 구문(예: `+reward_model.reward_kwargs.max_resp_len`)은 기존 설정에 **새 키를 추가**
- 설정 파일 위치: `verl/trainer/config/ppo_trainer.yaml`

### 1.2 한 스텝의 실행 흐름

```
1. Generation   — vLLM으로 응답 생성 (rollout.n=4 → 프롬프트당 4개 응답)
2. Reward       — DAPO 보상 매니저로 점수 계산
3. Old Log Prob — actor가 torch.no_grad()로 old_log_prob 계산
4. Ref Log Prob — ref 모델이 ref_log_prob 계산 (KL용)
5. Advantage    — GRPO: 그룹 내 보상 정규화
6. Actor Update — PPO/GSPO 정책 손실로 역전파
```

### 1.3 학습 스텝 수 계산

```python
total_training_steps = len(train_dataloader) * total_epochs
# len(train_dataloader) = ceil(dataset_size / train_batch_size)
```

스크립트 기준 (25% 서브셋 3,529개, batch_size=64):
```
steps_per_epoch = 3529 // 64 = 55  (drop_last=True)
total_steps = 55 * 3 = 165
save_freq=58 → epoch 1 끝 근처에서 저장
test_freq=10 → 10 스텝마다 검증
```

---

## 2. GRPO 어드밴티지 추정기

> 파일: `verl/trainer/ppo/core_algos.py` (lines 266-330)

### 2.1 알고리즘

두 스크립트 모두 `algorithm.adv_estimator=grpo`를 사용한다. GRPO는 critic(밸류 네트워크) 없이 동작한다.

```python
# 1. 토큰 보상을 시퀀스 단위로 합산
scores = token_level_rewards.sum(dim=-1)  # shape: (batch_size,)

# 2. 프롬프트 UID로 그룹핑
# rollout.n=4이면, 같은 프롬프트에서 나온 4개 응답이 한 그룹

# 3. 그룹 내 정규화
advantage[i] = (score[i] - group_mean) / (group_std + eps)

# 4. 토큰 차원으로 브로드캐스트 (모든 토큰에 같은 어드밴티지)
advantages = scores.unsqueeze(-1) * response_mask
```

### 2.2 GAE와의 핵심 차이

| | GRPO | GAE |
|--|------|-----|
| critic 필요 | 아니오 | 예 |
| 어드밴티지 해상도 | 시퀀스당 스칼라 | 토큰별 |
| 베이스라인 | 그룹 평균 보상 | 밸류 함수 V(s) |
| 할인 인자 (gamma, lambda) | 미사용 | 사용 |

### 2.3 KL 관련 설정

두 스크립트 모두:
- `algorithm.use_kl_in_reward=False` → 보상에 KL 페널티 미적용
- `actor.use_kl_loss=False` → 정책 손실에 KL 항 미포함
- GSPO 스크립트는 추가로 `kl_loss_coef=0.0`을 명시 설정

→ **KL 제약 없는 순수 보상 기반 학습**

---

## 3. 정책 손실: Vanilla PPO vs GSPO

> 파일: `verl/trainer/ppo/core_algos.py`

### 3.1 Vanilla PPO 정책 손실 (GRPO 베이스라인)

```python
# 토큰 수준 importance ratio
ratio = exp(log_prob - old_log_prob)

# PPO 클리핑
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * clip(ratio, 1-0.2, 1+0.2)
pg_losses = max(pg_losses1, pg_losses2)

# 집계: token-mean (모든 토큰 균등 가중)
loss = sum(pg_losses * mask) / sum(mask)
```

### 3.2 GSPO 정책 손실 (lines 1253-1326)

GSPO는 **시퀀스 수준 importance ratio**와 **토큰 수준 그래디언트**를 결합한다.

```python
# Step 1: 시퀀스 수준 importance ratio (기하평균)
#   s_i(θ) = exp[ (1/|y_i|) * Σ_t log(π_θ / π_θ_old) ]
neg_kl = log_prob - old_log_prob                      # 토큰별 log ratio
seq_lengths = response_mask.sum(dim=-1).clamp(min=1)
neg_kl_seq = (neg_kl * response_mask).sum(dim=-1) / seq_lengths  # 시퀀스 평균

# Step 2: stop-gradient 결합 (핵심 트릭)
#   log(s_{i,t}) = sg[log(s_i)] + log(π_θ) - sg[log(π_θ)]
log_ratio = log_prob - log_prob.detach() + neg_kl_seq.detach().unsqueeze(-1)
ratio = exp(clamp(log_ratio, max=10.0))

# Step 3: 매우 타이트한 클리핑 (ε_low=0.0003, ε_high=0.0004)
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * clip(ratio, 1-0.0003, 1+0.0004)
pg_losses = max(pg_losses1, pg_losses2)

# Step 4: seq-mean-token-mean 집계
#   먼저 시퀀스 내 토큰 평균 → 그 다음 시퀀스 간 평균
per_seq = sum(pg_losses * mask, dim=-1) / seq_lengths  # 시퀀스 내 평균
loss = mean(per_seq)                                    # 시퀀스 간 평균
```

### 3.3 GSPO의 핵심 설계 의도

| 설계 요소 | 이유 |
|-----------|------|
| 시퀀스 수준 ratio | 개별 토큰이 아닌 전체 응답 수준에서 정책 변화를 측정 |
| stop-gradient 트릭 | 시퀀스 ratio는 방향만 제공, 그래디언트는 현재 토큰의 log_prob만 통과 |
| 타이트한 클리핑 (3e-4/4e-4) | 시퀀스 ratio가 이미 토큰 ratio의 평균이므로 더 좁은 범위가 적절 |
| seq-mean-token-mean | 긴 응답이 짧은 응답보다 과도한 영향을 주는 것을 방지 |
| Dual-clip 미사용 | `clip_ratio_c=10.0`이 설정되어 있지만 GSPO 코드에서 무시됨 |

> 논문: https://arxiv.org/pdf/2507.18071

---

## 4. DAPO 보상 매니저

> 파일: `verl/workers/reward_manager/dapo.py`
> 레지스트리: `verl/workers/reward_manager/registry.py`

### 4.1 보상 계산 흐름

```
1. 응답 디코딩 → response_str
2. data_source 필드로 보상 함수 라우팅 (예: "math_dapo")
3. compute_score(solution_str, ground_truth)
   → 정답 추출 (regex: "Answer:\s*..." 또는 \boxed{...})
   → 정규화 (단위 제거, LaTeX 처리, 쉼표 제거 등)
   → 비교: 정답이면 +1.0, 오답이면 -1.0
4. overlong 페널티 적용 (선택)
5. 최종 보상을 응답의 마지막 토큰 위치에 배치
```

### 4.2 Overlong Buffer 메커니즘

두 스크립트 모두 길이 페널티를 활성화한다:

```bash
+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True
+reward_model.reward_kwargs.overlong_buffer_cfg.len=$((MAX_RESPONSE_LENGTH / 2))  # 4096
+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
```

**계산 공식:**

```
expected_len = max_resp_len - buffer_len = 8192 - 4096 = 4096
exceed_len = actual_response_length - expected_len
overlong_reward = min(-exceed_len / buffer_len * penalty_factor, 0)
```

**예시:**
| 응답 길이 | exceed_len | overlong_reward | 최종 보상 (정답 시) |
|-----------|-----------|-----------------|-------------------|
| 3000 | -1096 | 0 (페널티 없음) | +1.0 |
| 5000 | 904 | -0.221 | +0.779 |
| 6000 | 1904 | -0.465 | +0.535 |
| 8192 | 4096 | -1.0 | 0.0 |

→ 4096 토큰까지는 페널티 없음. 이후 선형으로 증가하여 8192에서 -1.0 도달.

### 4.3 수학 보상 함수 세부사항

> 파일: `verl/utils/reward_score/math_dapo.py`

- `\boxed{}` 또는 `Answer:` 패턴으로 답 추출
- LaTeX 정규화: 분수, 제곱근, 단위 처리
- 숫자 비교: 쉼표 제거, 소수점 정규화
- **이진 보상**: 정답 = +1.0, 오답 = -1.0 (연속값 아님)

---

## 5. 데이터 파이프라인

### 5.1 Parquet 파일 구조

```python
# 필수 컬럼
"prompt"        # str 또는 [{"role": "user", "content": "..."}] 형식
# 선택 컬럼
"data_source"   # 보상 함수 라우팅 키 (예: "math_dapo")
"reward_model"  # {"ground_truth": "정답값"} 딕셔너리
"extra_info"    # {"index": 0, "tools_kwargs": {...}} 등
"images"        # 멀티모달 모델용 이미지 데이터
```

### 5.2 배치 크기 관계

```
train_batch_size = 64     (프롬프트 수)
rollout.n = 4             (프롬프트당 응답 수)
real_train_batch_size = 64 * 4 = 256  (실제 학습 샘플 수)
n_gpus = 4

GPU당 미니배치 = ppo_mini_batch_size / n_gpus = 16 / 4 = 4
GPU당 마이크로배치 = ppo_micro_batch_size_per_gpu = 2
gradient_accumulation_steps = 4 / 2 = 2
```

### 5.3 필터링 및 트렁케이션

- `filter_overlong_prompts=True`: 512 토큰 초과 프롬프트 제거 (데이터 로딩 시)
- `truncation='error'`: 최대 길이 초과 시 에러 발생 (안전 모드)

---

## 6. Actor/Rollout/Ref 아키텍처

### 6.1 세 컴포넌트의 관계

```
┌─────────────────────────────────────────────┐
│           actor_rollout_ref (Hybrid Engine)   │
│                                               │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │  Actor    │  │ Rollout  │  │ Ref Policy │  │
│  │ (FSDP2)  │  │ (vLLM)   │  │ (FSDP2)    │  │
│  │          │  │          │  │ offload=T   │  │
│  │ 학습     │  │ 생성     │  │ KL 계산    │  │
│  └──────────┘  └──────────┘  └────────────┘  │
│                                               │
│  동일한 GPU 프로세스에서 모드 전환하며 실행    │
└─────────────────────────────────────────────┘
```

### 6.2 FSDP2 전략

- PyTorch DTensor 기반의 최신 분산 학습 전략
- 파라미터, 그래디언트, 옵티마이저 상태를 GPU 간 샤딩
- 스크립트 설정:
  - Actor: `param_offload=False`, `optimizer_offload=False` (빠른 학습)
  - Ref: `param_offload=True` (메모리 절약, forward만 수행)

### 6.3 vLLM Rollout 설정

```yaml
rollout.name: vllm
rollout.tensor_model_parallel_size: 1  # 추론 시 TP=1
rollout.gpu_memory_utilization: 0.6    # GPU 메모리의 60%를 KV 캐시에 할당
rollout.n: 4                           # 프롬프트당 4개 응답 생성
```

### 6.4 Rollout Trace (디버깅)

```yaml
rollout.trace.backend: weave          # W&B Weave 연동
rollout.trace.token2text: true        # 토큰 ID → 텍스트 변환
rollout.trace.max_samples_per_step_per_worker: 1  # 워커당 스텝당 1개만 추적
```

→ 학습 중 생성된 응답을 Weave 대시보드에서 텍스트로 확인 가능

---

## 7. 핵심 파일 위치 참조

| 컴포넌트 | 파일 경로 |
|---------|----------|
| 훈련 진입점 | `verl/trainer/main_ppo.py` |
| 훈련 루프 | `verl/trainer/ppo/ray_trainer.py` |
| 핵심 알고리즘 (GRPO, GSPO, PPO) | `verl/trainer/ppo/core_algos.py` |
| Actor 학습 루프 | `verl/workers/actor/dp_actor.py` |
| Actor 설정 | `verl/workers/config/actor.py` |
| 알고리즘 설정 | `verl/trainer/config/algorithm.py` |
| DAPO 보상 매니저 | `verl/workers/reward_manager/dapo.py` |
| 보상 매니저 레지스트리 | `verl/workers/reward_manager/registry.py` |
| 수학 보상 함수 | `verl/utils/reward_score/math_dapo.py` |
| 데이터셋 | `verl/utils/dataset/rl_dataset.py` |
| Hydra 기본 설정 | `verl/trainer/config/ppo_trainer.yaml` |

---

## 8. 실험 재현 & 변형을 위한 팁

### 8.1 새 알고리즘으로 변형하려면

1. `core_algos.py`에 `@register_policy_loss("my_algo")` 데코레이터로 손실 함수 등록
2. 실행 스크립트에 `actor_rollout_ref.actor.policy_loss.loss_mode=my_algo` 추가
3. 필요시 `loss_agg_mode`, `clip_ratio_*` 조정

### 8.2 보상 함수를 바꾸려면

1. `verl/utils/reward_score/`에 새 보상 모듈 작성
2. `verl/utils/reward_score/__init__.py`의 `default_compute_score()`에 data_source 라우팅 추가
3. Parquet 데이터의 `data_source` 컬럼을 새 이름으로 설정

### 8.3 배치 크기 조정 가이드

```
OOM 발생 시:
  → ppo_micro_batch_size_per_gpu 줄이기 (2 → 1)
  → rollout.gpu_memory_utilization 줄이기 (0.6 → 0.4)
  → ref.fsdp_config.param_offload=True 확인

학습이 불안정할 때:
  → ppo_mini_batch_size 줄이기 (업데이트 단위 축소)
  → clip_ratio 조정 (GSPO: 3e-4~5e-4 범위)
  → rollout.n 늘리기 (그룹 내 분산 증가)

학습이 너무 느릴 때:
  → ppo_micro_batch_size_per_gpu 늘리기
  → rollout.gpu_memory_utilization 늘리기
  → actor.fsdp_config.param_offload=False
```

### 8.4 검증 주기 설정

```bash
trainer.test_freq=10       # 10 스텝마다 검증 (자주 — 초기 디버깅용)
trainer.save_freq=58       # 58 스텝마다 체크포인트 (약 1 에폭)
trainer.val_before_train=False  # 학습 전 검증 스킵
```

---

## 9. 두 스크립트의 수치 흐름 비교

### GRPO 베이스라인 (`run_qwen3-8b_4.sh`)

```
프롬프트 64개 → vLLM (n=4) → 256개 응답
  → DAPO: math_dapo 정답 검사 (+1/-1) + overlong 페널티
  → GRPO advantage: 그룹(4개) 내 정규화
  → Vanilla PPO loss: token-level ratio, clip(0.8, 1.2)
  → token-mean 집계
  → optimizer.step()
```

### GSPO 변형 (`run_qwen3-8b_4_gspo.sh`)

```
프롬프트 64개 → vLLM (n=4) → 256개 응답
  → DAPO: math_dapo 정답 검사 (+1/-1) + overlong 페널티
  → GRPO advantage: 그룹(4개) 내 정규화
  → GSPO loss: sequence-level ratio + token gradient
     clip(1-0.0003, 1+0.0004)   ← 매우 타이트
  → seq-mean-token-mean 집계   ← 시퀀스 균등 가중
  → optimizer.step()
```

---

## 10. Advantage Dampening 구현 시 발견한 코드베이스 특이점

> `adv_dampen_pilot/` 실험을 구현하면서 마주한 verl 코드베이스의 함정과 주의점을 정리한다.
> 향후 커스텀 advantage estimator나 보상 함수를 추가할 때 반드시 참고할 것.

### 10.1 AlgoConfig는 frozen dataclass

```python
# verl/trainer/config/algorithm.py
@dataclass(frozen=True)  # ← 생성 후 필드 수정 불가
class AlgoConfig:
    ...
```

**주의점:**
- `cfg.adv_dampen_tau = 2.0` 같은 직접 할당은 `FrozenInstanceError` 발생
- 테스트 시 반드시 생성자에서 모든 필드를 지정: `AlgoConfig(adv_dampen_tau=2.0, ...)`
- 또는 `config.get("field_name", default)` 패턴으로 딕셔너리처럼 접근 (실제 코드에서 사용하는 방식)

**실무 팁:** `AlgoConfig.get()` 메서드가 존재하며, OmegaConf의 DictConfig와 호환된다.
이것을 사용하면 frozen 여부에 관계없이 안전하게 값을 읽을 수 있다.


### 10.2 `_resolve_device()` — CPU 테스트 시 함정

```python
# verl/utils/groupwise.py:54
def _resolve_device(device):
    if device is not None:
        return torch.device(device) if isinstance(device, str) else device
    return get_torch_device()  # ← 문제의 원인
```

**문제:** `get_torch_device()`는 `torch.cuda` **모듈 자체**를 반환한다 (`torch.device` 인스턴스가 아님).
GPU 환경에서는 이후 `tensor.to(device=torch.cuda)`가 내부적으로 처리되지만,
CPU 환경에서는 `TypeError: to() received an invalid combination of arguments` 발생.

**해결:** 테스트 실행 시 `VERL_FORCE_DEVICE=cpu` 환경 변수를 설정하면
`_resolve_device()`가 `get_torch_device()` 대신 `torch.device("cpu")`를 반환한다.

```bash
VERL_FORCE_DEVICE=cpu python3 -c "from verl.utils.groupwise import group_mean_std; ..."
```


### 10.3 Advantage estimator 등록: enum vs string

```python
# core_algos.py — 두 가지 등록 방식
@register_adv_est(AdvantageEstimator.GRPO)       # enum (기존 내장 estimator)
@register_adv_est("grpo_dampened")                # string (커스텀 estimator)
```

**주의점:**
- `AdvantageEstimator` enum은 **immutable** — 새 값을 추가할 수 없음
- 커스텀 estimator는 반드시 **문자열 키**로 등록해야 함
- dispatch 경로가 다름:
  - `AdvantageEstimator.GRPO` → `ray_trainer.py:235` 전용 분기 (직접 호출)
  - 문자열 키 → `ray_trainer.py:248` 제네릭 `else` 분기 (`get_adv_estimator_fn()` 사용)

**실무 팁:** 제네릭 분기는 `inspect.signature()`로 함수 매개변수를 확인하고
`config`가 있으면 자동으로 전달한다. 커스텀 estimator에서 config가 필요하면
함수 시그니처에 `config: Optional[AlgoConfig] = None`을 반드시 포함할 것.


### 10.4 보상 함수 라우팅의 오염 위험

```python
# verl/utils/reward_score/__init__.py:44-47
if data_source == "openai/gsm8k":
    from . import gsm8k
    res = gsm8k.compute_score(solution_str, ground_truth)
    # ← format_score=0.0이 하드코딩됨
```

**주의점:**
- 이 라우터는 **모든** gsm8k 실험에서 공유됨
- `format_score`를 바꾸면 다른 실험에 영향
- 보상 함수를 커스터마이즈하려면 라우터를 수정하지 말고 **`custom_reward_function`** 메커니즘 사용

**안전한 방법:**

```python
# 별도 파일: reward_gsm8k_with_format.py
from verl.utils.reward_score.gsm8k import compute_score as _gsm8k_score

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return _gsm8k_score(solution_str, ground_truth, format_score=0.1)
```

```bash
# 실행 스크립트에서:
custom_reward_function.path=/path/to/reward_gsm8k_with_format.py
custom_reward_function.name=compute_score
```

**흐름:** `load_reward_manager()` → `get_custom_reward_fn(config)` → 외부 파일 로드
→ `DAPORewardManager(compute_score=커스텀_함수)` — `default_compute_score` 라우터를 완전히 우회.


### 10.5 GSM8K 형식 보상: Base 모델의 필수 요소

```python
# verl/utils/reward_score/gsm8k.py:29-31
if method == "strict":
    solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
    # ← "#### 72" 형식이 아니면 answer=None → reward=0
```

**주의점:**
- GSM8K의 strict 모드는 `#### <숫자>` 패턴을 요구
- **Instruct 모델**은 이 형식을 알고 있지만, **Base 모델**은 모름
- Base 모델로 GRPO 학습 시 `format_score=0.0`이면:
  - 초기에 거의 모든 응답이 reward=0 (형식 미충족)
  - 그룹 내 모든 보상이 0이면 std=0 → advantage=0 → 학습 신호 없음
  - 학습이 사실상 시작되지 않을 수 있음
- **`format_score=0.1`** 설정이 사실상 필수 (geo3k도 동일 패턴 사용)


### 10.6 DiagWriter 패턴: 함수 속성을 이용한 지연 초기화

```python
# core_algos.py — Policy Diag / Dampen Diag 공통 패턴
def compute_something(...):
    ...
    if not hasattr(compute_something, "_diag"):
        compute_something._diag = _SomeDiagWriter(config)
    compute_something._diag.record(...)
```

**주의점:**
- 함수 속성(`function._diag`)은 **프로세스 수명** 동안 유지됨
- 분산 학습에서 각 rank(GPU)가 독립된 인스턴스를 가짐
- `atexit.register(self._flush)`로 프로세스 종료 시 잔여 버퍼 자동 flush
- 테스트 간 격리가 필요하면 `delattr(func, '_diag')`로 수동 리셋

**설계 이유:** advantage estimator 함수는 `(Tensor, Tensor)` 반환 시그니처가 고정되어 있으므로,
진단 데이터를 반환값으로 전달할 수 없다. 함수 속성 + `.pt` 파일 사이드채널이 가장 비침습적인 해법.


### 10.7 compute_advantage() dispatch의 두 경로

```python
# ray_trainer.py:235-275
if adv_estimator == AdvantageEstimator.GRPO:    # 경로 A: 전용 분기
    advantages, returns = core_algos.compute_grpo_outcome_advantage(
        ..., norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
else:                                            # 경로 B: 제네릭 레지스트리
    adv_fn = core_algos.get_adv_estimator_fn(adv_estimator)
    adv_kwargs = {"token_level_rewards": ..., "response_mask": ..., "config": config}
    if "uid" in data.non_tensor_batch:
        adv_kwargs["index"] = data.non_tensor_batch["uid"]
    advantages, returns = adv_fn(**adv_kwargs)
```

**주의점:**
- 경로 A (GRPO 전용)는 `norm_adv_by_std_in_grpo`를 **직접 인자**로 전달
- 경로 B (제네릭)는 `config` 객체 전체를 전달하고, 함수 내부에서 `config.get()`으로 추출
- 커스텀 estimator에서 `norm_adv_by_std_in_grpo`가 필요하면 **config에서 읽어야 함**
  (`config.get("norm_adv_by_std_in_grpo", True)`)
- 경로 B는 `"uid"`가 `non_tensor_batch`에 있을 때만 `index`를 전달
  → estimator에서 `index`가 필수이면 누락 시 KeyError 가능


### 10.8 group_mean_std()의 Bessel 보정과 singleton 처리

```python
# verl/utils/groupwise.py:206-223
var = (s2 - s1*s1/count) / (count - 1).clamp_min(1.0)  # Bessel 보정
std = sqrt(clamp(var, min=eps))

# Singleton 그룹: mean=0, std=1로 강제
single = count <= 1.0
mean[single] = 0.0
std[single] = 1.0
```

**주의점:**
- 그룹 크기 1인 샘플(singleton)은 advantage = 0이 됨 (학습에 기여 안 함)
- `(count - 1)` 분모 → 그룹 크기 G=2일 때 std 추정이 불안정
- G가 클수록 (예: G=16) 안정적인 그룹 통계 → advantage dampening 효과도 안정적
- dampening의 τ=2.0은 G=16 기준으로 설계됨. G가 크게 달라지면 τ 재검토 필요


### 10.9 Hydra `+` 접두어 vs 일반 키

```bash
# + 없이: 기존 설정 키를 덮어씀 (키가 없으면 에러)
algorithm.adv_estimator=grpo_dampened

# + 있음: 새 키를 추가 (기존에 없어도 OK)
+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True
+actor_rollout_ref.actor.policy_diag_dir=/path/to/diag
```

**주의점:**
- `AlgoConfig`에 `adv_dampen_tau` 필드가 있으면 `+` 없이 설정 가능
- 그러나 `reward_kwargs`처럼 동적 딕셔너리에 추가하는 경우 `+` 필수
- `+`를 빼먹으면 `ConfigAttributeError: Key 'xxx' is not in struct` 발생
- 스크립트에서 `+actor_rollout_ref.actor.policy_diag_dir` 처럼 사용하는 이유:
  `ActorConfig`의 `policy_diag_dir` 필드가 기본 YAML에 없고 코드에서만 정의되어 있기 때문


### 10.10 `ray stop --force` — 절대 사용 금지

```bash
# 절대 하지 말 것:
ray stop --force   # ← 시스템 전체의 Ray 클러스터를 종료

# 올바른 방법: 개별 프로세스만 종료
kill -TERM <PID>   # 해당 학습 프로세스만 종료
```

**사고 사례:**
- dampened 파일럿 프로세스를 종료하려고 `ray stop --force` 실행
- 동일 머신에서 실행 중이던 CISPO 실험(GPU 0-3)도 같은 Ray 클러스터를 공유
- **CISPO 실험이 함께 종료됨** (step 115에서 강제 종료, 최대 5 step 분량 손실)

**원인:** verl의 모든 `main_ppo` 프로세스는 `ray.init()`으로 기존 Ray 클러스터에 자동 연결되거나
새 클러스터를 생성한다. `ray stop --force`는 **해당 노드의 모든 Ray 프로세스**를 종료시킨다.

**안전한 프로세스 종료 절차:**

```bash
# 1. 대상 프로세스 PID 확인
ps aux | grep main_ppo | grep "experiment_name" | grep -v grep

# 2. setsid으로 실행한 경우: 프로세스 그룹 종료
kill -TERM -<session_leader_PID>   # 음수 PID = 프로세스 그룹 전체

# 3. 개별 프로세스만 종료
kill -TERM <python_PID>

# 4. 종료 확인 후 잔여 GPU 프로세스 정리 (필요 시)
# Ray worker는 부모 종료 시 대부분 자동 정리되나, vLLM 프로세스가 남을 수 있음
nvidia-smi  # GPU 메모리 사용 확인
```

**복구:** CISPO는 `save_freq=5`로 체크포인트가 저장되어 있었고,
`resume_mode: auto` 설정 덕분에 동일 스크립트를 다시 실행하면 `global_step_115`부터 자동 재개.


### 10.11 `group_mean_std()` vs dict-based 그룹 계산 — device 호환성 함정

```python
# 방법 A: group_mean_std() — GPU 실행 시 device 버그 있음
from verl.utils import group_mean_std
g = as_torch_index(index, device=scores.device)
mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon)
# ↑ device=None 시 _resolve_device() → get_torch_device() → torch.cuda (모듈!)
# → scores.to(device=torch.cuda) → TypeError

# 방법 B: dict-based loop — compute_grpo_outcome_advantage와 동일 패턴 (안전)
id2score = defaultdict(list)
for i in range(bsz):
    id2score[index[i]].append(scores[i])
for uid in id2score:
    scores_tensor = torch.stack(id2score[uid])
    id2mean[uid] = torch.mean(scores_tensor)
    id2std[uid] = torch.std(scores_tensor)
```

**사고 사례:**
- `compute_grpo_dampened_outcome_advantage`에서 방법 A를 사용
- GPU 실행 시 `TypeError: to() received an invalid combination of arguments — got (device=module, ...)`
- 원인: `get_torch_device()`가 `torch.device("cuda")`가 아닌 `torch.cuda` **모듈**을 반환
- 기존 `compute_grpo_outcome_advantage`(방법 B)는 `group_mean_std`를 사용하지 않아 면역이었음
- `compute_grpo_vectorized_outcome_advantage`도 방법 A를 사용하지만, 실전에서 사용된 적 없어 버그 미발견

**교훈:**
- 새 advantage estimator를 작성할 때 실전 검증된 기존 함수의 패턴을 따를 것
- `group_mean_std()`를 사용하려면 반드시 `device=scores.device`를 명시 전달
- 또는 dict-based 방법 B를 사용하여 device 문제를 근본적으로 회피


### 10.12 CISPO 재개(resume)와 로그 중복 리스크

**확인된 사실 (2026-02-14 기준):**
- `CISPO.test` 체크포인트는 `global_step_115`까지 저장되어 있음
  (`checkpoints/Exp.Custom.Algo/CISPO.test/latest_checkpointed_iteration.txt = 115`)
- 로그에는 step 116 출력이 보이지만, `save_freq=5`이므로 다음 저장 시점(120) 전에 중단되면
  최신 저장 체크포인트는 115에서 멈출 수 있음
- `resume_mode=auto`는 `trainer.default_local_dir`에서 `latest_checkpointed_iteration.txt`를 읽어 자동 재개
  (코드: `verl/trainer/ppo/ray_trainer.py::_load_checkpoint`)

**중요 리스크: 동일 출력 경로 재사용**
- 아래 경로를 같은 값으로 재실행하면 step 번호가 같은 파일이 덮어써질 수 있음:
  - `trainer.rollout_data_dir` (예: `.../CISPO_test_rollout/116.jsonl`)
  - `trainer.validation_data_dir`
- 체크포인트 경로(`trainer.default_local_dir`)도 동일하면 기존 실험 위에 이어서 저장됨

**실무 권장안 (충돌 최소화):**
1. 재개 체크포인트는 명시적으로 고정:
   `trainer.resume_mode=resume_path`, `trainer.resume_from_path=.../global_step_115`
2. 출력 경로 분리:
   `trainer.rollout_data_dir`, `trainer.validation_data_dir`를 새 폴더로 지정
3. 저장 경로 분리:
   `trainer.default_local_dir`를 새 경로로 지정 (원본 실험 보존)

**"로그 덮어써도 상관없나?"에 대한 판단:**
- "최신 상태만 보면 된다"는 목적이면 기술적으로 가능
- 하지만 연구 관점(재현/비교/ablation)에서는 과거 샘플(JSONL) 증거가 사라지므로 비권장
- 특히 step-by-step 품질 드리프트를 사후 분석할 때 손실이 큼

**W&B 동작**
- `wandb.init(project=..., name=experiment_name, ...)` 호출 구조상, 재실행 시 새 run ID가 생성됨
  (기존 run 위에 로그를 덮어쓰지 않음)
- 즉 `experiment_name`이 같아도 W&B에서는 "동명이인 run"이 병렬로 쌓임
- 헷갈림 방지를 위해 재개 실험은 `trainer.experiment_name`에 접미사 권장
  (예: `CISPO.resume_g115`)


---

*이 문서는 verl 코드베이스 분석을 기반으로 자동 생성되었습니다.*
*관련 참조: `verl_engine_deep_dive.md`, `verl_memory_management.md`*
*Section 10은 adv_dampen_pilot 구현 과정에서 발견한 사항을 추가 정리한 것입니다.*

---

## 11. 최근 추가로 확정한 사실 (GCISPO/CISPO + parse_success 로깅)

### 11.1 `GCISPO.pilot.v2` 로그 경로 매핑 (중요)

- `experiment_name`은 v2지만, 실제 dump 경로는 v1 스타일 폴더를 사용했다.
  - `wandb/run-20260214_142412-u3jfs85c/files/config.yaml`
  - `trainer.experiment_name: GCISPO.pilot.v2`
  - `trainer.rollout_data_dir: .../logs/GCISPO_pilot_rollout`
  - `trainer.validation_data_dir: .../logs/GCISPO_pilot_validation`
- 즉, 이번 v2 정성/정량 분석 대상은 아래 폴더로 고정한다.
  - `logs/GCISPO_pilot_rollout` (현재 162개 jsonl)
  - `logs/GCISPO_pilot_validation` (현재 17개 jsonl)
- `logs/GCISPO_pilot_v2_rollout`, `logs/GCISPO_pilot_v2_validation`는 존재하지 않는다.

### 11.2 로그 타입과 필드 구조

- 텍스트 정성 분석의 본체는 `rollout/validation`의 `*.jsonl`이다.
- `rollout` 샘플 키:
  - `input`, `output`, `gts`, `score`, `step`, `acc`, `pred`, `overlong_reward`, `overlong`
- `validation` 샘플 키:
  - `input`, `output`, `gts`, `score`, `step`, `acc`, `pred`, `overlong_reward`, `overlong`, `reward`
- `GCISPO_diag`는 JSONL이 아니라 `.pt` 바이너리이며, 샘플 키는 아래와 같다.
  - `token_ratios`, `clipped`, `seq_ratios`, `mask`
  - 예시 shape: `token_ratios (512, 8192)`, `seq_ratios (512,)`

### 11.3 `parse_success_rate` 네이티브 지원 여부

- 현재 VERL PPO 기본 메트릭에 `parse_success_rate`라는 고정 항목은 없다.
- 파싱 성공 여부는 보통 보상 함수가 만든 `pred`를 통해 간접 판정한다.
  - 예: `math_dapo`에서 파싱 실패 시 `pred="[INVALID]"` 처리
  - 따라서 post-hoc 기준은 `mean(pred != "[INVALID]")`

### 11.4 보상 함수와 W&B 로깅의 실제 연결 구조

- 보상 함수가 dict를 반환하면, `NaiveRewardManager`가 `reward_extra_info`로 수집한다.
  - `verl/workers/reward_manager/naive.py`
- PPO 학습 루프는 `Tracking`을 통해 metrics dict를 W&B로 전송한다.
  - `verl/trainer/ppo/ray_trainer.py`
  - `verl/utils/tracking.py`
- 핵심: 보상 함수에서 `wandb.log()`를 직접 호출하는 구조가 아니다.

### 11.5 train vs val에서 reward extra metric 반영 차이

- `train`
  - 현재 기본 구현은 reward extra 키를 전부 자동 로깅하지 않는다.
  - 사실상 `format_score`, `ndcg`만 평균으로 별도 로깅한다.
- `validation`
  - reward extra 키를 넓게 수집해 `process_validation_metrics`로 집계한다.
  - 단, 문자열 값 키는 집계에서 제외된다.

### 11.6 실무 적용 결론 (`parse_success_rate`를 쓰려면)

1. 보상 함수 dict에 `parse_success`(0/1)를 넣는다.
2. validation에서는 거의 자동으로 메트릭 집계/로깅 가능하다.
3. training에서 실시간으로 보려면 `ray_trainer.py`의 train-side 로깅 키에
   `parse_success`를 추가하거나, numeric reward extra key 자동 집계로 일반화한다.

---

## 12. 2026-02-16 운영 기준 (Qwen2.5-3B-Instruct 트랙)

이 섹션은 현재 실제 운용 중인 스크립트 기준의 "실행 전 점검표 + 해석 기준"이다.
기존 섹션(0~11)은 코드베이스 분석 기록으로 유지하고, 실제 실험 운영은 본 섹션을 우선 참조한다.

### 12.1 현재 실험 스크립트 (활성 트랙)

- `RL_side_2/run_qwen25_3b_instruct_gcispo.sh`
- `RL_side_2/run_qwen25_3b_instruct_cispo.sh`

공통:
- 모델: `actor_rollout_ref.model.path=/home/work/DDAI_revised/verl/data/models/Qwen2.5-3B-Instruct`
- 데이터:
  - train: `data/dapo-math-17k-processed/en_verl_25pct.parquet`
  - val: `data/aime-2024-repo/data/aime-2024.parquet`
- 응답 길이:
  - `MAX_RESPONSE_LENGTH=8192`
  - `rollout.max_model_len=MAX_RESPONSE_LENGTH+1024` (vLLM KV cache 안전장치)
- 보상:
  - `reward_manager=dapo`
  - `answer_score_cfg.mode=one_or_zero` (정답 보상 0/1)
  - `overlong_buffer_cfg.mode=binary`, `binary_combine_mode=plus_one_or_zero`

### 12.2 GCISPO vs CISPO 현재 차이

GCISPO (`run_qwen25_3b_instruct_gcispo.sh`):
- `policy_loss.loss_mode=gcispo`
- `loss_agg_mode=seq-mean-token-mean`
- `clip_ratio_low=0.005`, `clip_ratio_high=0.005`
- `use_kl_loss=False`
- validation sampling override:
  - `val_kwargs.n=32`
  - `val_kwargs.do_sample=True`
  - `val_kwargs.temperature=1.0`
  - `val_kwargs.top_p=1.0`
  - `val_kwargs.top_k=-1`

CISPO (`run_qwen25_3b_instruct_cispo.sh`):
- `policy_loss.loss_mode=cispo`
- `clip_ratio_low=10`, `clip_ratio_high=0.2`
- `use_kl_loss=True`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl`
- validation은 기본 설정(별도 `val_kwargs` override 없음)

### 12.3 `top_k=-1` 해석 (vLLM)

- vLLM 문맥에서 `top_k=-1`은 **top-k 컷오프 비활성화**를 의미한다.
- 즉 어휘를 top-k로 자르지 않고, 샘플링 제어는 `temperature/top_p`로만 수행된다.
- 참고: HF rollout에서는 일반적으로 `top_k=0`이 같은 의미로 쓰인다.

### 12.4 실험명/로그 경로 네이밍 규칙 (resume 충돌 방지)

반드시 아래 네 항목을 동일 접미사로 함께 변경한다.

1. `trainer.experiment_name`
2. `trainer.rollout_data_dir`
3. `trainer.validation_data_dir`
4. `+actor_rollout_ref.actor.policy_diag_dir`

권장 접미사 예시:
- `_3_clip_0.005`
- `_4_lr_5e7`
- `_ablation_no_kl`

### 12.5 재실행 전 점검 체크리스트

1. 기존 프로세스 정리
   - `pgrep -af "python3 -m verl.trainer.main_ppo"`
   - 대상 PID만 `kill -TERM <pid>` (공유 Ray 환경에서 `ray stop --force` 금지)
2. 새 실험명/로그 경로 충돌 여부 확인
   - 같은 `experiment_name` 재사용 시 자동 resume/혼합 위험
3. vLLM 길이/메모리 확인
   - `rollout.max_model_len` 미설정 시 모델 config max length(예: 262144)를 따라가 KV cache OOM 가능
4. 실행 후 헤더 로그 검증
   - `loss_mode`, `clip_ratio`, `use_kl_loss`, `val_kwargs.*`가 의도대로 반영됐는지 확인

### 12.6 `reward/acc_mean` 해석 규칙

- `reward/acc_mean`은 해당 스텝에서 수집된 응답들의 `acc`(0/1) 평균이다.
- 즉 "배치 전체 응답 단위 평균 정확도"이며, prompt 단위 `pass@k`와 동일하지 않다.
- 따라서 쉬운/어려운 샘플 구분 없이 평균으로 합쳐진다.

실무 권장:
- 평균(`reward/acc_mean`)은 추세용으로 보고,
- 정밀 비교는 rollout jsonl 기반으로 prompt 난이도/문항별 지표를 별도 산출한다.

### 12.7 SSH/터미널 종료 내구 실행 표준

```bash
cd /home/work/DDAI_revised/verl
nohup setsid bash RL_side_2/run_qwen25_3b_instruct_gcispo.sh \
  > logs/gcispo_qwen25_3b_instruct_3_clip_0.005_nohup.log 2>&1 < /dev/null &
echo $! > logs/gcispo_qwen25_3b_instruct_3_clip_0.005.pid
disown
```

```bash
cd /home/work/DDAI_revised/verl
nohup setsid bash RL_side_2/run_qwen25_3b_instruct_cispo.sh \
  > logs/cispo_qwen25_3b_instruct_3_clip_0.005_nohup.log 2>&1 < /dev/null &
echo $! > logs/cispo_qwen25_3b_instruct_3_clip_0.005.pid
disown
```

모니터링:
- `pgrep -af "python3 -m verl.trainer.main_ppo.*qwen25_3b_instruct_3_clip_0.005"`
- `tail -f logs/gcispo_qwen25_3b_instruct_3_clip_0.005_nohup.log`
- `tail -f logs/cispo_qwen25_3b_instruct_3_clip_0.005_nohup.log`

---

## 13. PPO → CISPO → GSPO → GCISPO 알고리즘 이론 레퍼런스

> 이 섹션은 GCISPO를 이해하기 위한 필수 배경 지식을 압축 정리한 것이다.
> 상세 이론은 Notion (`RL side Project - GCISPO` 하위 페이지)과 `GCISPO_design.md`를 참조하라.

### 13.1 Vanilla PPO: 비관적 선택 + Gradient 소멸

**목적함수:**
```
L_PPO = -min( r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t )
```

여기서 `r_t = π_θ / π_old` (importance sampling ratio)

**핵심 메커니즘: `min`이 만드는 비관적 하한**

| | A > 0 (좋은 행동) | A < 0 (나쁜 행동) |
|--|--|--|
| r > 1+ε | **클리핑 활성** → gradient=0 | 클리핑 무력화 → weight 무제한 |
| r < 1-ε | 클리핑 무력화 → weight 무제한 | **클리핑 활성** → gradient=0 |

- **대각선**에서만 보호가 작동 (gradient 소멸로 trust region 유지)
- **반대 대각선**은 무방비 (하나의 토큰이 gradient를 지배할 수 있음)
- Off-policy 라운드가 쌓일수록 더 많은 토큰이 clip 경계를 넘어 **gradient death** 발생
- Dual-clip은 A<0 방향의 부분적 패치이나 구조적 해결은 아님

### 13.2 CISPO: "어떻게 클리핑하느냐"를 바꾸다

> 논문: https://arxiv.org/pdf/2506.13585 (MiniMax, 2025)

**목적함수:**
```
L_CISPO = -sg[clamp(r_t, max=1+ε_high)] · A_t · log π_θ(a_t|s_t)
```

**PPO와의 두 가지 구조적 차이:**

| 변화 | PPO | CISPO | 효과 |
|------|-----|-------|------|
| **Gradient 경로** | ratio를 통해 흐름 → clip 밖에서 gradient=0 | `sg(ratio)` + 별도 `log_prob` → 항상 gradient ≠ 0 | **Gradient death 해결** |
| **min의 역할** | `min(r×A, clip(r)×A)` = 비관적 선택 | `clamp(r, max=ε)` = ratio 상한 cap | **A 부호 무관하게 bounded** |

**하한 클리핑 제거가 안전한 이유:**
- 실제 설정: `clip_ratio_low=10` → `clamp(r, 1-10, 1+0.2)` = `clamp(r, -9, 1.2)`
- ratio = exp(·) > 0 이므로 하한 -9는 **절대 활성화되지 않음**
- PPO의 `min`이 없으므로, 상한 clamp이 **A 부호와 무관하게 항상 작동**
- gradient weight가 항상 `(0, 1+ε_high]` 범위로 bounded

**CISPO의 핵심 이점 (Case: A<0, r=1.5, ε=0.2):**

| | PPO | CISPO |
|--|--|--|
| 계산 | min(-15, -12) = **-15** (cap 무력화) | clamp(1.5, max=1.2) × (-10) = **-12** (cap 작동) |
| weight | 15 (**무제한**) | 12 (**bounded**) |

→ 하나의 토큰이 전체 gradient를 장악하는 것을 방지

### 13.3 GSPO: "무엇을 클리핑하느냐"를 바꾸다

> 논문: https://arxiv.org/pdf/2507.18071 (Qwen Team, 2025)

**핵심 아이디어: 토큰별 ratio → 시퀀스 기하평균 ratio**

```
s_i(θ) = exp( (1/|y_i|) × Σ_t log(r_{i,t}) )  ← 기하평균
```

**기하평균의 효과:**

| 길이 | 단순 곱 (평균 ratio=1.01) | 기하평균 |
|------|--------------------------|----------|
| 100 | ≈ 2.7 | 1.01 |
| 1000 | ≈ 21,000 | 1.01 |

- 시퀀스 길이에 불변, 이상 토큰 1개의 영향을 평균으로 희석
- MoE 라우팅 변화, 정밀도 차이에도 강건

**Combined Ratio 트릭 (gradient 전파):**

```
log r_combined = sg[log s_i] + log π_θ − sg[log π_θ]
                  ↑ 값=log s_i, grad=0    ↑ 값=0, grad=∂logπ/∂θ
```

→ 값은 시퀀스 ratio, gradient는 토큰별 ∂log π/∂θ를 통해 흐름

**GSPO가 해결하지 못한 것:**
- PPO의 `min`(비관적 선택)을 **그대로 유지** → gradient death 잔존
- 시퀀스 단위 클리핑이므로 **All-or-Nothing**: s_i가 경계를 넘으면 500개 토큰 전부 gradient=0
- 클리핑 범위가 극도로 좁아 (ε ≈ 0.0004), 더 많은 시퀀스가 경계를 넘음

### 13.4 GCISPO: 두 혁신의 직교적 결합

```
GSPO가 바꾼 축:  "무엇을 클리핑하느냐" (토큰 → 시퀀스)
CISPO가 바꾼 축: "어떻게 클리핑하느냐" (비관적 min → ratio cap + stop-gradient)
→ 독립적(직교) 축이므로 자연스럽게 결합 가능
```

| | 토큰 단위 ratio | 시퀀스 단위 ratio |
|--|--|--|
| **비관적 min (gradient death)** | PPO/GRPO | GSPO |
| **sg[clamp] (gradient 보존)** | CISPO | **GCISPO** |

**GCISPO 알고리즘 6단계:**

```python
# Step 1: 토큰별 log ratio (GSPO 방식 — clamp 미적용)
neg_kl = log_prob - old_log_prob

# Step 2: 시퀀스 기하평균 IS ratio (← GSPO)
log_s_i = mean(neg_kl * mask) / seq_len
s_i = exp(log_s_i)

# Step 3: Combined ratio 트릭 (← GSPO)
r_combined = exp(clamp(sg[log_s_i] + log_prob - sg[log_prob], max=10))

# Step 4: 비대칭 클리핑 + stop-gradient (← CISPO)
r̃ = sg[clamp(r_combined, 1-ε_low, 1+ε_high)]

# Step 5: CISPO 스타일 손실 — gradient는 log_prob만 통과
L = -r̃ · A · log_prob

# Step 6: 집계 — 기본 seq-mean-token-mean (← GSPO), 실험에 따라 token-mean도 가능
loss = agg_loss(L, mode=loss_agg_mode)
```

> **주의 (집계 모드)**: GCISPO 함수의 기본값은 `seq-mean-token-mean`이지만,
> §14의 현재 실험에서는 CISPO baseline과의 공정 비교를 위해 `token-mean`으로 통일했다.
> 집계 모드는 `ActorConfig.loss_agg_mode`로 스크립트에서 제어 가능하다.

> **주의 (gradient 트릭)**: GSPO의 gradient 트릭(Step 3의 `log_prob - sg[log_prob]`)은 GCISPO에서
> 사실상 **무효화**됨. Step 4의 `.detach()`가 ratio를 통한 모든 gradient를 차단하기 때문.
> 실제 gradient는 Step 5의 별도 `log_prob`을 통해 제공됨. 코드 일관성을 위해 유지.

**GSPO/CISPO/GCISPO 비교 요약:**

```
          GSPO                          CISPO                        GCISPO
          ────                          ─────                        ──────
ratio:    시퀀스+토큰 결합              토큰별                       시퀀스+토큰 결합 (GSPO)
clip:     max(unclipped, clipped)       sg(clipped) · log_prob       sg(clipped) · log_prob (CISPO)
clip범위: 대칭 타이트 (3e-4/4e-4)       비대칭 (low=10, high=0.2)    비대칭 (low=10, high=조율)
gradient: clip 밖에서 죽음              항상 살아있음                  항상 살아있음 (CISPO)
집계:     seq-mean-token-mean           token-mean                    seq-mean-token-mean (GSPO)
KL:       없음                          low_var_kl (0.001)            선택적 (실험 변수)
```

### 13.5 Gradient 흐름 비교 (핵심 정리)

| 알고리즘 | clip 경계 안 | clip 경계 밖 | 실질 효과 |
|----------|------------|------------|-----------|
| **PPO** | ∂L/∂θ = −r·A·∂r/∂θ | ∂L/∂θ = **0** | 경계 밖 토큰은 학습에서 탈락 |
| **GSPO** | ∂L/∂θ = −s·A·∂logπ/∂θ | ∂L/∂θ = **0** | All-or-Nothing: 시퀀스 전체 탈락 |
| **CISPO** | ∂L/∂θ = −r·A·∂logπ/∂θ | ∂L/∂θ = −(1+ε)·A·∂logπ/∂θ | 항상 gradient 존재, weight bounded |
| **GCISPO** | ∂L/∂θ = −r·A·∂logπ/∂θ | ∂L/∂θ = −(1+ε)·A·∂logπ/∂θ | 시퀀스 ratio 안정성 + gradient 보존 |

### 13.6 Notion 문서 참조

| 문서 | URL | 내용 |
|------|-----|------|
| PPO clipping 이해하기 | `30d7fe6aad8c80e1922fdca3df8e143d` | PPO의 min/clamp/gradient 메커니즘 상세 |
| CISPO | `30d7fe6aad8c80ccb761dc578be8532b` | CISPO의 구조적 차이, 하한 제거 안전성, self-braking |
| GSPO | `30f7fe6aad8c80aabbeef1460d143554` | 시퀀스 기하평균 ratio, combined ratio 트릭 |
| GCISPO 기획안 | `RL_side_2/GCISPO_design.md` | 구현 설계, 진단 로깅, 하이퍼파라미터 근거 |

---

## 14. 2026-02-23 현재 활성 실험 (high off-policy 트랙)

> 이전 섹션(12)의 실험들과 달리, 이 트랙은 **full dataset** (dapo-math-17k-processed 전체)을
> 사용하며, rollout.n=16, train_batch_size=512의 대규모 설정으로 진행된다.

### 14.1 실험 스크립트 3개

| # | 파일명 | 알고리즘 | GPU |
|---|--------|----------|-----|
| ① | `main_run_qwen25_3b_instruct_cispo_3_high_off_policy.sh` | CISPO (baseline) | 0,1,2,3 |
| ② | `main_run_qwen25_3b_instruct_gcispo_high_off_0.01_nokl_token_mean.sh` | GCISPO | 4,5,6,7 |
| ③ | `main_run_qwen25_3b_instruct_gcispo_high_off_0.0003~4_nokl_token_mean.sh` | GCISPO | 0,1,2,3 |

> ⚠️ **①과 ③은 GPU 0-3을 공유** → 동시 실행 불가.
> ①과 ②는 GPU가 겹치지 않으므로 동시 실행 가능.

### 14.2 공통 설정

```yaml
모델:        Qwen2.5-3B-Instruct
데이터:      dapo-math-17k-processed (train), aime-2024 (val)
배치:        train_batch_size=512, mini_batch=16, micro_batch=4/gpu
응답 길이:   max_response_length=8192
롤아웃:      vLLM, n=16 (train), n=32 (val), temperature=1.0
전략:        FSDP2, GPU 4장, lr=1e-6
에폭:        10, save_freq=15, test_freq=10
어드밴티지:  GRPO (KL in reward=False)
진단:        policy_diag 활성 (flush_freq=32)
```

### 14.3 핵심 차이점 비교

| 설정 | ① CISPO (baseline) | ② GCISPO 0.01 | ③ GCISPO 0.0003~4 |
|------|-------------------|---------------|-------------------|
| **loss_mode** | `cispo` | `gcispo` | `gcispo` |
| **clip_ratio_low** | `10` | `10` | `0.0003` |
| **clip_ratio_high** | `0.2` | `0.01` | `0.0004` |
| **loss_agg_mode** | `token-mean` | `token-mean` | `token-mean` |
| **use_kl_loss** | `True` (coef=0.001, low_var_kl) | `False` | `False` |
| **reward 표기** | `mode=one_or_zero` + `binary/plus_one_or_zero` | `weight=1.0` | `weight=1.0` |

> **중요 관찰 1**: ②③ GCISPO의 `loss_agg_mode`는 `token-mean`이다.
> GCISPO_design.md의 기본 권장값 `seq-mean-token-mean`과 다르다.
> 이는 CISPO baseline과의 공정 비교를 위해 집계 방식을 통일한 것이다.
>
> **중요 관찰 2**: ②와 ③은 **서로 다른 클리핑 전략**을 테스트하는 별개 실험이다.
> - ② (`low=10, high=0.01`): CISPO 스타일 하한(비활성) + 시퀀스 ratio에 맞춘 중간 상한
> - ③ (`low=0.0003, high=0.0004`): GSPO 원본값 기반 양방향 타이트 클리핑
>
> **중요 관찰 3**: ②와 ③은 GPU 할당도 다르다 (② GPU 4-7, ③ GPU 0-3).
> ①과 ③은 GPU 0-3을 공유하므로 동시 실행 불가.

### 14.4 Reward 설정: 두 가지 표기법 (결과 동일)

| | ① CISPO | ②③ GCISPO |
|--|---------|-----------|
| 정답 보상 | `mode=one_or_zero` → 1 or 0 | `weight=1.0` → 1×1.0 or 0×1.0 |
| 길이 패널티 | `mode=binary, combine=plus_one_or_zero` | `weight=1.0` |

두 표기법은 `_map_answer_score()`와 `_compute_length_reward_components()` 내부에서
**동일한 binary(0/1) × weight 계산**을 수행한다.

- `answer_score_cfg`: 정답이면 1.0, 오답이면 0.0
- `overlong_buffer_cfg`: 7168(=8192-1024) 토큰 이내면 +1.0, 초과면 0.0

> 코드 위치: `verl/workers/reward_manager/dapo.py`
> - `_map_answer_score()` (lines 231-242)
> - `_compute_length_reward_components()` (lines 46-59)

### 14.5 clip_ratio의 실제 유효 범위 (직관 형성용)

세 실험의 `clip_ratio_low/high`가 ratio에 어떤 범위를 만드는지 직관적으로 파악:

```
① CISPO  (low=10, high=0.2):       clamp(r, 1-10, 1+0.2)     = clamp(r, -9.0, 1.2)
  → ratio는 exp(·) > 0이므로 실질 범위: (0, 1.2]
  → 상한만 작동. 매우 넓은 범위 — 토큰 ratio(0.5~2.0)의 대부분을 허용

② GCISPO (low=10, high=0.01):      clamp(r, 1-10, 1+0.01)    = clamp(r, -9.0, 1.01)
  → 실질 범위: (0, 1.01]
  → 상한만 작동. ①보다 20배 타이트한 상한 — 시퀀스 ratio에 맞춤
  → CISPO 스타일 비대칭 (하한 비활성 + 상한 cap)

③ GCISPO (low=0.0003, high=0.0004): clamp(r, 1-0.0003, 1+0.0004) = clamp(r, 0.9997, 1.0004)
  → 실질 범위: [0.9997, 1.0004]
  → 극도로 타이트 — GSPO 원본 수준
  → 상한/하한 모두 활성 가능 (거의 대칭)
```

> **역대 GCISPO clip_ratio 변천** (혼동 방지):
>
> | 실험 | clip_ratio_low | clip_ratio_high | 맥락 |
> |------|---------------|-----------------|------|
> | GCISPO_design.md 권장값 | 10 | 0.05 | CISPO 하한 스타일 + 시퀀스 ratio 맞춤 |
> | §12 실험 (`run_qwen25_3b_instruct_gcispo.sh`) | 0.005 | 0.005 | 대칭 파일럿 |
> | §14 실험 ② (`main_run_*_0.01_*.sh`) | 10 | 0.01 | CISPO 스타일 하한 + 타이트 상한 |
> | §14 실험 ③ (`main_run_*_0.0003~4_*.sh`) | 0.0003 | 0.0004 | GSPO 원본값 채택, 양방향 타이트 |
>
> 각 실험이 **다른 clip_ratio를 사용**하므로, 결과 비교 시 어떤 값인지 반드시 확인할 것.

### 14.6 실험의 핵심 가설과 해석 주의사항

```
① CISPO baseline:   토큰 수준 IS ratio + 넓은 비대칭 클리핑(10/0.2) + KL 정규화
② GCISPO 0.01:      시퀀스 수준 IS ratio + 중간 비대칭 클리핑(10/0.01) + KL 없음
③ GCISPO 0.0003~4:  시퀀스 수준 IS ratio + 타이트 대칭 클리핑(0.0003/0.0004) + KL 없음

핵심 질문:
  - ②: CISPO 스타일 비대칭(하한 비활성)을 유지하되, 상한만 타이트하게 하면?
  - ③: GSPO처럼 양방향 극도로 타이트하게 하면?
  - ①↔②↔③ 비교로 클리핑 전략의 효과를 탐색
```

> ⚠️ **Confounding Variables 경고**:
> ①과 ②③ 사이에는 **3개 이상의 변수**가 동시에 변경되었다:
>
> 1. **loss_mode**: cispo → gcispo (ratio 계산 방식)
> 2. **clip_ratio**: (10/0.2) → (10/0.01) 또는 (0.0003/0.0004) (클리핑 범위)
> 3. **use_kl_loss**: True → False (KL 정규화)
>
> 따라서 ①↔②③ 성능 차이의 원인을 단독으로 귀인할 수 없다.
> 다만, **②↔③ 비교**는 clip_ratio 전략만 다르므로 (loss_mode, KL 동일),
> 클리핑 범위의 효과를 더 직접적으로 비교할 수 있다.

### 14.7 W&B 프로젝트/실험명

| # | experiment_name | project |
|---|-----------------|---------|
| ① | `CISPO_qwen25_3b_instruct_3_high_off_policy_v2` | `Exp.Custom.Algo` |
| ② | `GCISPO_qwen25_3b_instruct_6_high_off_0.01_nokl_token_mean` | `Exp.Custom.Algo` |
| ③ | `GCISPO_qwen25_3b_instruct_6_high_off_0.0003_0.0004_nokl_token_mean` | `Exp.Custom.Algo` |

---

## 15. 하기 쉬운 실수 모음 & 빠른 참조

> 이 섹션은 새 에이전트나 협업자가 빠르게 훑어볼 수 있도록
> 코드베이스에서 반복적으로 발생하는 함정을 한 곳에 모았다.

### 15.1 실행 & 인프라 관련

| 실수 | 결과 | 올바른 방법 |
|------|------|------------|
| `ray stop --force` 실행 | **동일 머신의 모든 실험이 종료됨** | `kill -TERM <PID>` 으로 개별 프로세스만 종료 |
| GPU 번호 겹치는 스크립트 동시 실행 | OOM 또는 NCCL 충돌 | `CUDA_VISIBLE_DEVICES` 확인 후 실행 |
| `rollout.max_model_len` 미설정 | 모델 config max (예: 262144) 적용 → KV cache OOM | 반드시 `$((MAX_RESPONSE_LENGTH + 1024))` 설정 |
| 같은 `experiment_name`으로 재실행 | 자동 resume/로그 혼합 위험 | 새 접미사 추가 (예: `_v2`, `_ablation`) |
| Hydra `+` prefix 누락 | `ConfigAttributeError: Key 'xxx' is not in struct` | YAML에 없는 키는 반드시 `+key=value` |

### 15.2 알고리즘 & 하이퍼파라미터 관련

| 실수 | 결과 | 올바른 방법 |
|------|------|------------|
| GCISPO에 CISPO의 clip_ratio 사용 (low=10, high=0.2) | 시퀀스 ratio가 좁은 분포라 클리핑이 거의 안 됨 | 시퀀스 ratio 분포에 맞게 타이트하게 설정 (예: 0.0003~0.05) |
| GSPO에 CISPO의 clip_ratio 사용 (low=10, high=0.2) | 위와 동일 | GSPO 기본값: low=0.0003, high=0.0004 |
| 파일명의 숫자를 실제 하이퍼파라미터로 착각 | 잘못된 실험 해석 | **반드시 스크립트 내부 값 확인** |
| `loss_agg_mode` 무시 | 긴 응답이 짧은 응답보다 과도한 영향 (token-mean) 또는 그 반대 | 비교 실험 시 집계 모드 통일 |
| KL loss 비활성화 후 넓은 clip_ratio 사용 | 정책 발산 위험 | KL 없으면 타이트한 클리핑으로 보상, 또는 KL 활성화 |

### 15.3 보상 함수 관련

| 실수 | 결과 | 올바른 방법 |
|------|------|------------|
| `mode=one_or_zero`와 `weight=1.0`을 다른 보상이라고 오해 | 실험 비교 오류 | 내부 코드 확인 — 둘 다 binary(0/1)×weight |
| `overlong_buffer_cfg.len` 계산 실수 | 의도와 다른 길이 임계값 | `expected_len = max_resp_len - buffer_len` 확인 |
| Base 모델에서 `format_score=0.0` | 초기 reward 전부 0 → 학습 신호 없음 | Base 모델은 `format_score=0.1` 필수 |
| `data_source` 컬럼과 보상 함수 라우팅 불일치 | 잘못된 보상 함수 호출 | Parquet의 `data_source`와 `__init__.py` 라우터 매칭 확인 |

### 15.4 진단 & 분석 관련

| 실수 | 결과 | 올바른 방법 |
|------|------|------------|
| `.pt` 진단 파일을 JSONL로 읽으려 함 | 파싱 에러 | `torch.load()` 사용 |
| 진단 파일의 session_id를 step 번호로 착각 | 파일 매핑 오류 | session_id는 생성 시각 (YYYYMMDDTHHmmss) |
| `kill -9`로 학습 종료 | 진단 버퍼 flush 안 됨 (최대 flush_freq-1 분 손실) | `kill -TERM` 또는 `Ctrl+C` 사용 |
| 진단 `.pt`와 rollout JSONL의 샘플 대응 실수 | 잘못된 분석 | rank별 오프셋 고려: rank0의 sample[k] ↔ JSONL line[k], rank1 ↔ line[64+k] 등 |

### 15.5 빠른 파일 탐색 가이드

```
알고리즘 코드를 찾고 싶다면:
  → verl/trainer/ppo/core_algos.py
  → 검색: @register_policy_loss("cispo"), @register_policy_loss("gcispo")

보상 함수를 찾고 싶다면:
  → verl/workers/reward_manager/dapo.py (DAPO 매니저)
  → verl/utils/reward_score/ (개별 보상 함수)

설정 스키마를 찾고 싶다면:
  → verl/workers/config/actor.py (ActorConfig — clip_ratio, loss_mode 등)
  → verl/trainer/config/algorithm.py (AlgoConfig — adv_estimator 등)
  → verl/trainer/config/ppo_trainer.yaml (Hydra 기본 YAML)

학습 루프를 찾고 싶다면:
  → verl/trainer/ppo/ray_trainer.py (메인 루프 — fit(), _training_step())
  → verl/workers/actor/dp_actor.py (Actor 학습 — update_policy())
  → verl/workers/utils/losses.py (손실 함수 dispatch — get_policy_loss_fn())

실험 스크립트를 찾고 싶다면:
  → RL_side_2/main_run_*.sh (현재 활성 실험)
  → RL_side_2/run_*.sh (이전 실험)

설계 문서를 찾고 싶다면:
  → RL_side_2/GCISPO_design.md (GCISPO 기획안)
  → RL_side_2/verl_engine_deep_dive.md (verl 엔진 구조)
  → RL_side_2/verl_memory_management.md (메모리 관리)
```

### 15.6 실험 네이밍 규칙 (4개 경로 동시 변경 필수)

새 실험을 만들 때 아래 4개 경로를 **동일 접미사로 함께 변경**해야 한다.
하나라도 빠지면 이전 실험 데이터와 혼합되거나 덮어써질 위험이 있다.

```bash
trainer.experiment_name='GCISPO_qwen25_3b_instruct_7_...'
trainer.rollout_data_dir='.../logs/GCISPO_qwen25_3b_instruct_rollout_7_...'
trainer.validation_data_dir='.../logs/GCISPO_qwen25_3b_instruct_validation_7_...'
+actor_rollout_ref.actor.policy_diag_dir='.../logs/GCISPO_qwen25_3b_instruct_diag_7_...'
```

---

## 16. 개념적으로 정확하게 알아야 하는 사실들

> 이 섹션은 표면적으로 비슷해 보이지만 실제로는 다른 개념들,
> 또는 직관적으로 잘못 이해하기 쉬운 사실들을 정리한다.

### 16.1 PPO의 `min`과 CISPO의 `min(r, ε)`은 완전히 다른 연산이다

**PPO**: `min(r_t × A_t, clip(r_t) × A_t)` — **A를 곱한 후** 두 후보를 비교하는 **비관적 선택**
- A의 부호가 어떤 항이 선택되는지를 결정 (대각선 패턴)
- 선택된 항이 상수이면 gradient = 0

**CISPO**: `min(r_t, ε_high)` — ratio 자체에 상한을 거는 **값 cap** (= `torch.clamp(r, max=ε_high)`)
- A와 무관하게 ratio가 항상 ε_high 이하로 bounded
- 이후 sg()로 상수화하고 별도 log_prob을 곱하므로 gradient 항상 살아있음

> **핵심**: 둘 다 "min"이라고 불리지만, PPO의 min은 "두 목적함수 후보 비교"이고,
> CISPO의 min은 "하나의 값에 상한 씌우기"이다. 수학적으로 완전히 다른 연산.

### 16.2 CISPO에서 하한 클리핑 제거가 안전한 이유는 "self-braking" 때문이 아니다

**잘못된 설명**: "A<0일 때 ratio가 줄어들면 gradient도 자연 감쇠하니까 안전하다"
→ 이는 **동일 샘플이 반복 처리되는 이상적 상황**에서만 성립. 미니배치 간에는 보장 안 됨.

**올바른 설명**: CISPO는 PPO의 비관적 선택(min)을 **제거**했으므로,
상한 `clamp(r, max=1+ε_high)`이 **A의 부호와 무관하게** 직접 작동한다.
따라서 gradient weight가 어떤 경우에도 `(0, 1+ε_high]` 범위로 bounded되며,
하한 클리핑이 없어도 안전하다.

→ 안전성의 근거는 "자연 감쇠"가 아니라 **"비관적 선택 제거로 인한 구조적 bounding"**이다.

### 16.3 GCISPO에서 GSPO의 gradient 트릭은 사실상 무효화된다

GSPO의 `log_prob - sg[log_prob]` 트릭은 gradient를 ratio를 통해 흘리기 위한 것이다:

```python
# GSPO: 이 트릭으로 gradient가 combined_ratio를 통해 흐름
log_ratio = log_prob - log_prob.detach() + neg_kl_seq.detach().unsqueeze(-1)
ratio = exp(log_ratio)
# → ratio에 대해 미분하면 ∂logπ/∂θ가 살아있음
```

그러나 GCISPO에서는 Step 4에서 `clipped_ratio.detach()`가 **ratio를 통한 모든 gradient를 차단**한다:

```python
# GCISPO: .detach()가 ratio 경로의 gradient를 모두 끊음
clipped_ratio_sg = clipped_ratio.detach()  # ← 여기서 차단
pg_losses = -clipped_ratio_sg * advantages * log_prob  # gradient는 이 log_prob에서만
```

→ GCISPO에서 gradient는 **오직 Step 5의 별도 `log_prob`**을 통해서만 흐른다.
→ GSPO의 gradient 트릭은 **값(value) 계산에만 기여**하고, gradient 전파에는 무관하다.
→ 코드에서 유지하는 이유는 일관성이며, 제거해도 동작은 동일하다.

### 16.4 시퀀스 ratio의 분포가 토큰 ratio보다 훨씬 좁다

이것이 **clip_ratio가 알고리즘마다 완전히 다른 범위**인 근본적 이유이다:

```
토큰 ratio 분포:   0.5 ~ 2.0     → ε ≈ 0.2   (PPO, CISPO)
시퀀스 ratio 분포: 0.999 ~ 1.001 → ε ≈ 0.0004 (GSPO, GCISPO)
```

- 시퀀스 ratio = 토큰 ratio의 **기하평균** → 평균화에 의해 1 근처로 수렴
- CISPO의 ε=0.2를 GCISPO에 그대로 사용하면 **클리핑이 거의 발생하지 않아** 사실상 무제한 업데이트
- GSPO의 ε=0.0004를 CISPO에 사용하면 **거의 모든 토큰이 클리핑되어** 학습이 극도로 느려짐
- **반드시 ratio의 실제 분포 범위에 맞게 ε를 조정**해야 한다

### 16.5 `log_prob`의 "값"과 "gradient"를 구분해야 한다

CISPO/GCISPO의 손실 `L = -sg[r̃] · A · log_prob`에서:

```python
# log_prob의 "값" (예: -3.5)은 gradient 계산에 관여하지 않는다
# gradient는 ∂log_prob/∂θ를 통해 흐르며, 이것은 log_prob의 값과 다른 양이다

# 비유: f(x) = 2·x 에서 f(3) = 6이지만, df/dx = 2이다
# 마찬가지로 log_prob의 값이 -3.5이어도, ∂log_prob/∂θ는 별개의 벡터이다

# 실용적 의미:
# - sg[r̃]·A는 gradient의 "스케일링 팩터"로만 작용
# - gradient의 "방향"은 ∂log_prob/∂θ가 결정
# - 따라서 클리핑이 걸려도 gradient 방향은 변하지 않고, 크기만 bounded
```

### 16.6 GRPO의 advantage는 시퀀스 단위 스칼라이다

```
GRPO advantage = (score_i - group_mean) / (group_std + ε)
```

- 하나의 시퀀스 내 모든 토큰이 **동일한 advantage 값**을 가짐
- 이것이 GSPO/GCISPO의 "All-or-Nothing" 문제와 결합되면:
  - 시퀀스 ratio가 클리핑 경계를 넘으면, 500개 토큰 전부가 **동일 advantage인데 동시에 gradient=0** (GSPO)
  - GCISPO는 이 문제를 해결: 클리핑 경계 밖에서도 gradient가 살아있으므로 "전체 탈락" 없음

### 16.7 `use_kl_loss=False`이면서 `kl_loss_coef=0.001`은 모순이 아니다

②③ GCISPO 스크립트에서:

```bash
actor_rollout_ref.actor.use_kl_loss=False      # ← KL 손실 비활성화
actor_rollout_ref.actor.kl_loss_coef=0.001      # ← 설정만 해 놓고 사용 안 함
actor_rollout_ref.actor.kl_loss_type=low_var_kl  # ← 마찬가지
```

- `use_kl_loss=False`가 최우선. 나머지 KL 관련 설정은 **무시됨**
- 코드에서 `if config.use_kl_loss:` 분기 안에서만 `kl_loss_coef`를 참조
- 이런 패턴은 "나중에 KL을 켤 때 값만 바꾸면 되도록" 미리 설정해 놓은 것

### 16.8 `algorithm.use_kl_in_reward`와 `actor.use_kl_loss`는 다른 KL이다

| 설정 | 역할 | 적용 위치 |
|------|------|----------|
| `algorithm.use_kl_in_reward=False` | 보상에 KL 페널티 추가 여부 | advantage 계산 전, 보상 값 자체를 변경 |
| `actor.use_kl_loss=True` | 정책 손실에 KL 정규화 항 추가 여부 | 손실 함수에 `λ·KL(π_θ \|\| π_ref)` 추가 |

- 전자는 RLHF 초기(InstructGPT)의 방식: `r' = r - β·KL`
- 후자는 CISPO의 방식: `L_total = L_policy + λ·KL_loss`
- 세 스크립트 모두 전자는 `False`, 후자만 선택적으로 사용

### 16.9 보상 합산 구조: answer + overlong은 독립적으로 더해진다

```python
# dapo.py의 실제 보상 계산 흐름 (simplified):
reward = 0.0
reward += mapped_answer_score    # 정답이면 +1.0, 오답이면 0.0
reward += overlong_reward        # 7168 이내면 +1.0, 초과면 0.0
# → 가능한 최종 보상: 0.0, 1.0, 2.0
```

| 정답 여부 | 길이 초과 여부 | 최종 보상 |
|----------|-------------|----------|
| 정답 (1.0) | 미초과 (1.0) | **2.0** |
| 정답 (1.0) | 초과 (0.0) | **1.0** |
| 오답 (0.0) | 미초과 (1.0) | **1.0** |
| 오답 (0.0) | 초과 (0.0) | **0.0** |

> **주의**: 이전 실험(§4.2)의 overlong은 **선형 감소 + 음수 페널티** 방식이었지만,
> 현재 high off-policy 트랙(§14)의 overlong은 **이진(0/1) + 덧셈** 방식이다.
> 같은 `overlong_buffer_cfg`라도 mode에 따라 동작이 완전히 다르다.

### 16.10 CISPO와 GCISPO의 수치 안정화 전략이 다르다

**CISPO**: `neg_kl = clamp(log_prob - old_log_prob, min=-20, max=20)` → 토큰별 ratio에 바로 exp() 적용
- clamp 없으면 `exp(20+)` = overflow 가능 → 직접 clamp 필수

**GCISPO**: `neg_kl = log_prob - old_log_prob` (clamp 없음) → 시퀀스 평균 후 exp()
- 시퀀스 평균화가 자체적으로 안정화 효과 제공
- 대신 `log_combined_ratio`에 `clamp(max=10.0)` 적용 (GSPO와 동일)

> 코드를 수정할 때 CISPO의 `clamp(min=-20, max=20)`을 GCISPO에 추가하거나,
> GCISPO의 미적용을 CISPO에 따라가면 **의도와 다른 동작**이 될 수 있다.
> 각 알고리즘의 안정화 지점이 다르다는 것을 인식할 것.

### 16.11 PPO에서 `max(pg_losses1, pg_losses2)` = `−min(objective1, objective2)`

코드를 읽을 때 가장 혼동되는 부분:

```python
# 코드 (최소화 대상):
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * clipped_ratio
pg_loss = torch.max(pg_losses1, pg_losses2)  # ← max!

# 수학 (최대화 대상):
J = min(r·A, clip(r)·A)  # ← min!
```

- **부호가 뒤집혀 있다**: `max(-a, -b) = -min(a, b)`
- 코드는 loss(minimize) 관점, 수학은 objective(maximize) 관점
- 따라서 코드의 `max`는 수학의 `min`(비관적 선택)과 동일

---

## 17. 로그 구조 & 빠른 분석 가이드 (2026-02-23 기준)

> 이 섹션은 실험 로그를 처음 접하는 에이전트가 "어디에 무엇이 있고, 어떻게 읽는지"를
> 5분 안에 파악할 수 있도록 작성되었다.

### 17.1 로그 디렉토리 레이아웃 (공통 패턴)

모든 실험은 아래 4개 디렉토리 + 1개 nohup 로그를 생성한다:

```
verl/
├── logs/
│   ├── {PREFIX}_rollout_{SUFFIX}/          ← 학습 rollout 데이터
│   │   └── {step}.jsonl                    (매 step 생성, 14~427MB)
│   ├── {PREFIX}_validation_{SUFFIX}/       ← 검증 데이터
│   │   └── {step}.jsonl                    (매 test_freq step, 59~744MB)
│   ├── {PREFIX}_diag_{SUFFIX}/             ← 정책 진단 텐서
│   │   └── {timestamp}_flush{N}_rank{R}.pt (flush_freq마다, 4.1MB 고정)
│   └── {nohup_name}.log                    ← 콘솔 stdout/stderr
│
├── checkpoints/Exp.Custom.Algo/{EXPERIMENT_NAME}/
│   ├── latest_checkpointed_iteration.txt   ← 최신 step 번호 (cat으로 확인)
│   └── global_step_{N}/
│       ├── data.pt                         ← 학습 메타 상태
│       └── actor/
│           ├── model_world_size_4_rank_{0-3}.pt   (~3GB/shard)
│           ├── optim_world_size_4_rank_{0-3}.pt   (~9GB/shard)
│           ├── extra_state_world_size_4_rank_{0-3}.pt
│           ├── fsdp_config.json
│           └── huggingface/               ← tokenizer, config.json 등
│
└── wandb/
    ├── latest-run -> run-YYYYMMDD_HHMMSS-{run_id}/
    └── run-YYYYMMDD_HHMMSS-{run_id}/
        ├── files/
        │   ├── config.yaml                ← 실험 설정 전체 (experiment_name 포함)
        │   ├── wandb-summary.json         ← 최종 메트릭 요약
        │   └── output.log                 ← W&B 캡처 로그
        ├── run-{run_id}.wandb             ← 바이너리 이벤트 로그
        └── logs/debug.log
```

### 17.2 파일별 읽기 방법

| 파일 패턴 | 포맷 | 읽기 방법 | 주요 필드 |
|-----------|------|----------|----------|
| `rollout/{step}.jsonl` | JSONL | `pd.read_json(path, lines=True)` | `input`, `output`, `gts`, `score`, `acc`, `pred`, `overlong_reward`, `overlong` |
| `validation/{step}.jsonl` | JSONL | 동일 | 위 + `reward` |
| `diag/{ts}_flush{N}_rank{R}.pt` | PyTorch | `torch.load(path, weights_only=True)` | `token_ratios (B,T)`, `clipped (B,T)`, `seq_ratios (B,)`, `mask (B,T)` |
| `checkpoints/.../data.pt` | PyTorch | `torch.load()` | 학습 step, epoch 등 메타 정보 |
| `wandb/.../config.yaml` | YAML | `yaml.safe_load()` | 실험 전체 설정 (experiment_name으로 실험 식별) |
| `wandb/.../wandb-summary.json` | JSON | `json.load()` | 최종 메트릭 (acc, loss 등) |
| `*.log` (nohup) | 텍스트 | `tail -f` / `grep` | 실시간 학습 진행 로그 |

### 17.3 현재 실험별 로그 현황 (2026-02-23 스냅샷)

| 항목 | ① CISPO baseline | ② GCISPO 0.01 | ③ GCISPO 0.0003~4 |
|------|-----------------|---------------|-------------------|
| **상태** | **완료** (step 270) | **진행 중** | **진행 중** |
| **기간** | Feb 21 16:28→Feb 23 01:46 (~33h) | Feb 22 18:39→진행중 | Feb 23 04:02→진행중 |
| **Rollout** | 270개, 5.6GB | 34개, 4.3GB | 53개, 1.8GB |
| **Validation** | 27개, 2.3GB | 3개, 2.6GB | 5개, 1.2GB |
| **Diagnostics** | 17,280개, 68GB | 2,176개, 8.6GB | 3,392개, 14GB |
| **Checkpoints** | 18 snap, 622GB | 2 snap, 70GB | 3 snap, 104GB |
| **총 용량** | ~698GB | ~85GB | ~121GB |
| **nohup log** | `cispo_3_high_off_policy_v2.log` (13MB) | `GCISPO_6_high_off_0.01_nokl_token_mean.log` (1.6MB) | `GCISPO_qwen25_3b_instruct_6_high_off_0.0003_0.0004_nokl_token_mean.out` (2.4MB) |

### 17.4 실험 → W&B run 매핑

| 실험 | W&B run directory | run ID | 상태 |
|------|-------------------|--------|------|
| ① CISPO | `run-20260221_162021-sck622jk` | `sck622jk` | 완료 (config.yaml, summary 존재) |
| ② GCISPO 0.01 | 미확인 | — | 진행 중 (config.yaml 미완성 가능) |
| ③ GCISPO 0.0003~4 | 미확인 | — | 진행 중 |

> **W&B run 식별 방법**: `wandb/run-*/files/config.yaml` 내 `trainer.experiment_name` 필드로 매칭.
> 진행 중인 실험은 `latest-run` 심링크(`→ run-20260223_035519-dh44kq20`)가 가리키는
> run이 현재 활성 실험일 가능성이 높다.

### 17.5 로그 파일 네이밍 패턴

```
Rollout/Validation 파일:
  {global_step}.jsonl
  → step 번호가 곧 파일명. test_freq=10이면 validation에 10.jsonl, 20.jsonl, ...

Diagnostics 파일:
  {실험시작시각}_flush{순번4자리}_rank{GPU번호}.pt
  예: 20260221T162807_flush0001_rank0.pt
  → 타임스탬프는 실험 시작 시각 (고정), flush 순번만 증가
  → rank 0~3 (4-GPU FSDP)
  → flush 1회 = 4파일 (rank당 1개)

Checkpoint 디렉토리:
  global_step_{N}/
  → save_freq=15이므로 15, 30, 45, ... 간격
```

### 17.6 빠른 실험 상태 확인 원라이너

```bash
# 최신 체크포인트 step 확인
cat checkpoints/Exp.Custom.Algo/{EXPERIMENT_NAME}/latest_checkpointed_iteration.txt

# rollout 진행 상태 (파일 수 = 완료 step 수)
ls logs/{PREFIX}_rollout_{SUFFIX}/ | wc -l

# validation 진행 상태
ls logs/{PREFIX}_validation_{SUFFIX}/ | wc -l

# 진단 flush 수 (4로 나누면 flush 횟수)
ls logs/{PREFIX}_diag_{SUFFIX}/ | wc -l

# 현재 실행 중인 실험 프로세스 확인
pgrep -af "python3 -m verl.trainer.main_ppo"

# 실시간 학습 로그 모니터링
tail -f logs/{nohup_name}.log

# 디스크 사용량 확인
du -sh logs/{PREFIX}_*/
du -sh checkpoints/Exp.Custom.Algo/{EXPERIMENT_NAME}/
```

### 17.7 로그 분석 시 주의사항

| 주의사항 | 설명 |
|---------|------|
| **rollout JSONL 크기가 후반에 급증** | 응답 길이가 학습 중 변화하기 때문. 초기 14~36MB → 후기 200~427MB |
| **diagnostics `.pt`를 JSONL로 읽으면 에러** | 반드시 `torch.load()` 사용 |
| **diag의 session_id ≠ step 번호** | 타임스탬프는 실험 시작 시각, flush 순번은 누적 카운터 |
| **rank별 diag 파일은 서로 다른 데이터 슬라이스** | rank0의 sample[k] ↔ 전체 배치의 line[k], rank1 ↔ line[batch/4 + k] 등 |
| **`kill -9`로 종료 시 diag 버퍼 유실** | 최대 flush_freq-1 스텝분 손실. `kill -TERM` 사용 |
| **checkpoint 1개 = ~35GB** | save_freq를 너무 낮게 잡으면 디스크 급속 소모 |
| **W&B 진행 중 실험은 config.yaml 불완전 가능** | 실험 완료 후 최종 기록됨. 진행 중에는 `latest-run` 심링크로 추정 |

### 17.8 실험별 전체 경로 매핑 (복붙용)

```bash
# ① CISPO baseline (완료)
ROLLOUT=logs/CISPO_qwen25_3b_instruct_rollout_3_high_off_policy_v2
VALID=logs/CISPO_qwen25_3b_instruct_validation_3_high_off_policy_v2
DIAG=logs/CISPO_qwen25_3b_instruct_diag_3_high_off_policy_v2
CKPT=checkpoints/Exp.Custom.Algo/CISPO_qwen25_3b_instruct_3_high_off_policy_v2
LOG=logs/cispo_3_high_off_policy_v2.log

# ② GCISPO 0.01 (진행 중)
ROLLOUT=logs/GCISPO_qwen25_3b_instruct_rollout_6_high_off_0.01_nokl_token_mean
VALID=logs/GCISPO_qwen25_3b_instruct_validation_6_high_off_0.01_nokl_token_mean
DIAG=logs/GCISPO_qwen25_3b_instruct_diag_6_high_off_0.01_nokl_token_mean
CKPT=checkpoints/Exp.Custom.Algo/GCISPO_qwen25_3b_instruct_6_high_off_0.01_nokl_token_mean
LOG=logs/GCISPO_6_high_off_0.01_nokl_token_mean.log

# ③ GCISPO 0.0003~4 (진행 중)
ROLLOUT=logs/GCISPO_qwen25_3b_instruct_rollout_6_high_off_0.0003_0.0004_nokl_token_mean
VALID=logs/GCISPO_qwen25_3b_instruct_validation_6_high_off_0.0003_0.0004_nokl_token_mean
DIAG=logs/GCISPO_qwen25_3b_instruct_diag_6_high_off_0.0003_0.0004_nokl_token_mean
CKPT=checkpoints/Exp.Custom.Algo/GCISPO_qwen25_3b_instruct_6_high_off_0.0003_0.0004_nokl_token_mean
LOG=logs/GCISPO_qwen25_3b_instruct_6_high_off_0.0003_0.0004_nokl_token_mean.out
```

> **주의**: ②는 `.log` 확장자, ③은 `.out` 확장자. 내용 형식은 동일 (stdout/stderr 캡처).
>
> **②번 clip_ratio 확인 완료**: nohup 로그로 검증한 결과, ②번 실험은 파일명대로
> `clip_ratio_low=10, clip_ratio_high=0.01`로 실행되었다 (§14.3 참조).
> ③번 실험은 `clip_ratio_low=0.0003, clip_ratio_high=0.0004`이며, **②와 ③은 다른 클리핑 전략**이다.

### 17.9 diag flush 순번 ↔ step 매핑

diagnostics `.pt` 파일의 flush 순번은 step 번호가 **아니다**. 매핑 관계:

```
flush_freq = 32  (스크립트의 policy_diag_flush_freq)
mini_batch_per_step = ppo_mini_batch_size / micro_batch_per_gpu / n_gpus
                    = 16 / 4 / 4 = 1  (현재 설정 기준)

→ flush 1회 = 32 미니배치 = 32 step (mini_batch_per_step=1일 때)
→ flush N번째 파일 ≈ step (N × flush_freq)
```

그러나 이 계산은 **ppo_mini_batch_size와 micro_batch 설정에 따라 변한다**.
정확한 매핑이 필요하면 nohup 로그에서 `step` 출력 타이밍과 diag 파일 타임스탬프를 대조한다:

```bash
# diag 파일의 마지막 수정 시각 확인
ls -lt logs/{PREFIX}_diag_{SUFFIX}/ | head -5

# nohup 로그에서 해당 시각 전후의 step 번호 찾기
grep "step:" logs/{nohup_name}.log | tail -10
```

### 17.10 자주 쓰는 분석 레시피

```python
# ── 1. 학습 곡선 (step별 정확도 추이) ──
import pandas as pd, glob, os

rollout_dir = "logs/CISPO_qwen25_3b_instruct_rollout_3_high_off_policy_v2"
records = []
for f in sorted(glob.glob(f"{rollout_dir}/*.jsonl")):
    step = int(os.path.basename(f).replace(".jsonl", ""))
    df = pd.read_json(f, lines=True)
    records.append({"step": step, "acc_mean": df["acc"].mean(), "n": len(df)})
curve = pd.DataFrame(records).sort_values("step")

# ── 2. validation pass@k 근사 ──
val_dir = "logs/CISPO_qwen25_3b_instruct_validation_3_high_off_policy_v2"
df = pd.read_json(f"{val_dir}/270.jsonl", lines=True)
# prompt별 그룹핑 (같은 input → n=32개 응답)
pass_at_1 = df.groupby("input")["acc"].mean().mean()  # 근사 pass@1

# ── 3. diag ratio 분포 확인 ──
import torch
diag = torch.load("logs/.../flush0100_rank0.pt", weights_only=True)
seq_ratios = diag["seq_ratios"]  # shape: (B,)
print(f"seq ratio: mean={seq_ratios.mean():.6f}, std={seq_ratios.std():.6f}")
print(f"range: [{seq_ratios.min():.6f}, {seq_ratios.max():.6f}]")
clipped_frac = diag["clipped"].float().mean()
print(f"clipped fraction: {clipped_frac:.4f}")
```

> **주의**: rollout JSONL이 200MB+인 경우 `pd.read_json()` 시 수 GB RAM을 소모할 수 있다.
> 대량 파일 분석 시 chunked 읽기 또는 필요 컬럼만 추출하는 것을 권장한다.

---

*이 문서는 verl 코드베이스 분석 및 실험 운영 경험을 기반으로 작성/갱신되었습니다.*
*마지막 업데이트: 2026-02-23 (rev.5) — ② clip_ratio 오류 수정 (0.0003/0.0004→10/0.01), §14.3 3열 비교표로 개편, §14.5-14.6 ②③ 구분 반영, §17 로그 구조 가이드*
