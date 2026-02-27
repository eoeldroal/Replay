# verl GPU 메모리 관리 심층 분석

> 대상: Qwen3-8B (~8.55B params), 4× H100 80GB, FSDP2, vLLM (TP=1), 현재 실험 설정 기준

---

## 1. HybridEngine: 하나의 GPU에서 3개의 역할

verl의 HybridEngine은 **actor(학습)**, **rollout(생성)**, **ref(참조 정책)**를 동일한 GPU 프로세스에 공존시킨다. 이들은 동시에 실행되지 않으며, **단계별로 모드를 전환**하면서 GPU 메모리를 시분할 공유한다.

```
┌──────────────────────────────────────────────────────────────────┐
│                     하나의 GPU (H100 80GB)                        │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │  FSDP Actor      │  │  vLLM Engine │  │  FSDP Ref Model  │    │
│  │  (학습 모델)       │  │  (생성 엔진)  │  │  (참조 모델)      │    │
│  │                  │  │              │  │                  │    │
│  │  weights (상주)   │  │  sleep/wake  │  │  param_offload   │    │
│  │  optimizer (상주) │  │  로 전환      │  │  =True (CPU)     │    │
│  └──────────────────┘  └──────────────┘  └──────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### 1.1 각 컴포넌트의 메모리 관리 전략

| 컴포넌트 | 상주/비상주 | 관리 메커니즘 | 메모리 제어 파라미터 |
|----------|:---------:|-------------|-------------------|
| **FSDP Actor** | GPU 상주 | FSDP2 sharding | `param_offload`, `optimizer_offload` |
| **vLLM Rollout** | 비상주 (필요 시만) | `sleep()` / `wake_up()` | `gpu_memory_utilization`, `enable_sleep_mode` |
| **FSDP Ref** | CPU 상주 | `param_offload=True` | `ref.fsdp_config.param_offload` |

현재 설정에서:
- **Actor**: `param_offload=False`, `optimizer_offload=False` → weights + optimizer가 항상 GPU에 상주
- **Rollout (vLLM)**: `enable_sleep_mode=True` (디폴트) → 학습 중에는 GPU 메모리 반납
- **Ref**: `param_offload=True` → 필요할 때만 GPU로 올라옴, 평소에는 CPU

---

## 2. vLLM의 Sleep/Wake 메커니즘

### 2.1 Sleep Level

vLLM 버전에 따라 sleep의 깊이가 달라진다:

| Sleep Level | 조건 | 해제 대상 | 해제 메모리 (8B 기준) |
|:-----------:|------|----------|---------------------|
| **Level 1** | vLLM < 0.8.5 또는 NPU | KV cache만 | ~31 GB |
| **Level 2** | vLLM ≥ 0.8.5 | KV cache + 모델 가중치 | ~48 GB |

```python
# verl/third_party/vllm/__init__.py
if vs.parse(package_version) >= vs.parse("0.8.5"):
    VLLM_SLEEP_LEVEL = 2  # 가중치까지 해제
```

### 2.2 모드 전환 흐름

```
                  rollout_mode()                          trainer_mode()
                  ─────────────                           ──────────────
  (학습 중)  ──→  vLLM wake_up   ──→  generate  ──→  vLLM sleep     ──→  (학습 중)
                  ├ weights 복구                          ├ weights 해제
                  ├ FSDP→vLLM 동기화                      ├ KV cache 해제
                  └ KV cache 복구                         └ empty_cache()
```

코드 위치:
- `rollout_mode()`: `fsdp_workers.py:678-759`
- `trainer_mode()`: `fsdp_workers.py:761-773`
- `release()` (sleep): `vllm_rollout.py:252-255`
- `resume()` (wake_up): `vllm_rollout.py:243-250`

### 2.3 Sleep Level 2에서 가중치를 해제해도 되는 이유

vLLM의 모델 가중치는 FSDP actor의 가중치와 **별도의 복사본**이다. 학습 후 `update_weights()`로 FSDP의 최신 가중치를 vLLM에 다시 동기화하므로, 학습 중에는 vLLM 가중치가 GPU에 있을 필요가 없다.

---

## 3. 컴포넌트별 메모리 상세 산출

### 3.1 FSDP Actor (GPU 상주)

Qwen3-8B: 8.55B 파라미터, 4 GPU FSDP2 sharding, bf16

| 항목 | 계산 | GPU당 |
|------|------|-------|
| Model weights (sharded bf16) | 8.55B × 2B / 4 GPU | **4.3 GB** |
| Adam momentum `exp_avg` (sharded fp32) | 8.55B × 4B / 4 GPU | **8.6 GB** |
| Adam variance `exp_avg_sq` (sharded fp32) | 8.55B × 4B / 4 GPU | **8.6 GB** |
| Gradients (sharded bf16) | 8.55B × 2B / 4 GPU | **4.3 GB** |
| **소계 (학습 시 피크)** | | **~25.7 GB** |

> **주의**: Gradients는 backward 중에만 존재. Optimizer states는 첫 `optimizer.step()` 이후에만 존재.

### 3.2 vLLM Rollout (비상주, wake 시)

| 항목 | 계산 | GPU당 |
|------|------|-------|
| Model weights (full copy, bf16, TP=1) | 8.55B × 2B | **17.1 GB** |
| KV cache | `total × util - weights` | **가변** |
| **소계 (util=0.6)** | 80 × 0.6 = 48 GB | **~48 GB** |

`gpu_memory_utilization`은 전체 GPU 메모리의 비율이다. vLLM은 이 예산 내에서 weights를 빼고 남은 부분을 KV cache로 할당한다.

```
vLLM 예산 = 80GB × 0.6 = 48 GB
KV cache  = 48 - 17.1 = ~30.9 GB
```

### 3.3 FSDP Ref (CPU 상주, `param_offload=True`)

| 항목 | 위치 | GPU당 |
|------|------|-------|
| Model weights | **CPU** | 0 GB |
| Optimizer | 없음 (추론 전용) | 0 GB |
| 임시 forward용 weights | GPU (compute 시만) | ~4.3 GB (임시) |

`compute_ref_log_prob` 실행 시에만 CPU → GPU로 올라왔다가 완료 후 다시 내려간다.

---

## 4. 1 Iteration의 메모리 타임라인

아래는 한 iteration(= 한 training step) 동안 각 GPU에서의 메모리 점유 변화이다.

### 4.1 첫 번째 iteration (optimizer states 미생성)

```
                      GPU 메모리 (80 GB)
                 0          20          40          60          80
Phase            |           |           |           |           |
─────────────────┼───────────┼───────────┼───────────┼───────────┤
                 │           │           │           │           │
① rollout_mode() │▓▓         │                                   │
   FSDP wt only  │4.3GB      │           │           │           │
                 │           │           │           │           │
② generate       │▓▓ ████████████████████████████████│           │
   FSDP + vLLM   │4.3│       vLLM 48GB                │  Free 28GB│
                 │           │           │           │           │
③ trainer_mode() │▓▓         │           │           │           │
   vLLM sleep    │4.3GB      │           Free ~76 GB │           │
                 │           │           │           │           │
④ compute_log_prob│▓▓▒▒       │           │           │           │
   +ref on GPU    │4.3│ref 4.3│           │           │           │
                 │           │           │           │           │
⑤ update_actor   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░  │           │
   (첫 step)     │wt │**optim 생성 17.1GB**│grad│activ│           │
                 │4.3│      17.1          │4.3 │가변  │           │
                 │           │           │           │           │

▓ = FSDP weights    ▒ = Ref weights (임시)
█ = vLLM            ░ = Gradients + Activations
```

### 4.2 두 번째 iteration 이후 (optimizer states 존재)

```
                      GPU 메모리 (80 GB)
                 0          20          40          60          80
Phase            |           |           |           |           |
─────────────────┼───────────┼───────────┼───────────┼───────────┤
                 │           │           │           │           │
① rollout_mode() │▓▓▓▓▓▓▓▓▓▓▓▓           │           │           │
   FSDP wt+optim │   21.4 GB              │           │           │
                 │           │           │           │           │
② generate       │▓▓▓▓▓▓▓▓▓▓▓▓████████████████████████████████ │
   FSDP + vLLM   │  21.4 GB  │         vLLM 48GB               │F│
                 │           │           │           │       Free│
                 │           │           │           │       9GB │
③ trainer_mode() │▓▓▓▓▓▓▓▓▓▓▓▓           │           │           │
   vLLM sleep    │  21.4 GB               │  Free ~59 GB         │
                 │           │           │           │           │
④ compute_log_prob│▓▓▓▓▓▓▓▓▓▓▓▓▒▒         │           │           │
   +ref on GPU    │  21.4 GB  │ref 4.3    │           │           │
                 │           │           │           │           │
⑤ update_actor   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░│           │
                 │wt+optim 21.4│         grad│activation│           │
                 │              │         4.3 │  가변     │           │
```

> **핵심 관찰**: 두 번째 iteration의 ② generate에서 여유가 ~28 GB → ~9 GB로 급감한다. 이는 첫 번째 `optimizer.step()`에서 Adam states(~17.1 GB)가 생성되기 때문이다.

---

## 5. Adam Optimizer States의 Lazy 생성

PyTorch의 Adam/AdamW는 성능 최적화를 위해 `exp_avg`(1차 모멘텀)와 `exp_avg_sq`(2차 모멘텀)를 **첫 번째 `optimizer.step()` 호출 시점에 생성**한다.

```python
# PyTorch Adam._init_group() 내부 (간략화)
if len(state) == 0:
    state['step'] = torch.tensor(0.)
    state['exp_avg'] = torch.zeros_like(p)      # ← 여기서 최초 할당
    state['exp_avg_sq'] = torch.zeros_like(p)    # ← 여기서 최초 할당
```

### 5.1 타임라인 상의 영향

| 시점 | Optimizer States | GPU당 점유 |
|------|:----------------:|:---------:|
| init ~ 첫 rollout | 미생성 (0 GB) | 0 GB |
| 첫 `update_actor()` 중 `optimizer.step()` | **생성** | **+17.1 GB** |
| 이후 모든 iteration | 상주 (해제 안 됨) | 17.1 GB |

### 5.2 실무적 함의

- 첫 번째 rollout에서의 메모리 여유는 **착시**이다. `nvidia-smi`로 첫 rollout을 관찰하면 ~28 GB가 남아 보이지만, 이는 optimizer states가 아직 생성되지 않았기 때문이다.
- 두 번째 iteration부터 FSDP 점유가 4.3 GB → 21.4 GB로 급증한다.
- `gpu_memory_utilization` 결정 시 **반드시 두 번째 iteration 이후의 상황**을 기준으로 해야 한다.

---

## 6. gpu_memory_utilization 결정 공식

### 6.1 제약 조건

rollout 중 GPU 메모리가 다음을 만족해야 한다:

```
FSDP(weights + optimizer) + vLLM(weights + KV cache) + overhead < 전체 GPU 메모리
```

정리하면:

```
vLLM 예산 < 전체 GPU - FSDP 점유 - overhead
gpu_memory_utilization < (전체 GPU - FSDP 점유 - overhead) / 전체 GPU
```

### 6.2 현재 설정 계산 (Qwen3-8B, 4× H100 80GB)

```
전체 GPU         = 80 GB
FSDP 점유        = 21.4 GB  (weights 4.3 + optimizer 17.1)
overhead         = ~3 GB    (CUDA context, PyTorch, 임시 버퍼)
─────────────────────────────
vLLM 가용        = 80 - 21.4 - 3 = 55.6 GB
최대 utilization = 55.6 / 80 ≈ 0.695
```

### 6.3 권장 값

| util 값 | vLLM 예산 | KV cache | 여유 | 평가 |
|:-------:|:---------:|:--------:|:----:|:----:|
| **0.60** | 48.0 GB | 30.9 GB | 10.6 GB | 안전 (현재) |
| **0.65** | 52.0 GB | 34.9 GB | 6.6 GB | **권장** |
| 0.70 | 56.0 GB | 38.9 GB | 2.6 GB | 위험 |
| 0.75 | 60.0 GB | 42.9 GB | -1.4 GB | OOM |

> 0.65 권장 이유: KV cache가 4 GB 늘어 rollout throughput이 향상되면서, `rollout_mode()` 전환 시의 일시적 피크(`state_dict()` → `full_tensor()` 중 임시 버퍼)를 감안한 안전 마진(~6.6 GB)이 확보된다.

### 6.4 rollout_mode() 전환 시의 일시적 피크

`fsdp_workers.py:699,730-734`에서 FSDP → vLLM 가중치 동기화 시, 각 파라미터 텐서를 `full_tensor()`로 all-gather한다:

```python
per_tensor_param = (
    (name, param.to(device).full_tensor() if isinstance(param, DTensor) else param)
    for name, param in params.items()
)
```

generator 패턴이므로 한 번에 하나의 파라미터만 un-shard되지만, 큰 레이어(예: embedding 151936×4096 = ~1.2 GB)의 임시 피크가 발생할 수 있다.

---

## 7. param_offload / optimizer_offload 트레이드오프

### 7.1 offload=True 시 메모리 이득

| 설정 | rollout 중 FSDP GPU 점유 | vLLM 가용 | 최대 util |
|------|:-----------------------:|:---------:|:---------:|
| offload=False (현재) | 21.4 GB | 55.6 GB | ~0.69 |
| param_offload=True | 17.1 GB (optim만) | 59.9 GB | ~0.75 |
| 둘 다 True | ~0 GB | ~77 GB | ~0.90 |

### 7.2 offload=True 시 CPU↔GPU 전송 비용

`param_offload=True`일 때, 1 iteration 내에서 FSDP 모델이 CPU↔GPU를 오가는 횟수:

| 단계 | 함수 | 방향 |
|------|------|:----:|
| `rollout_mode()` (line 684) | `load_fsdp_model_to_gpu` | CPU → GPU |
| weight sync 전 (line 722) | `offload_fsdp_model_to_cpu` | GPU → CPU |
| `compute_log_prob` (line 1015) | `load_fsdp_model_to_gpu` | CPU → GPU |
| compute_log_prob 후 (line 1055) | `offload_fsdp_model_to_cpu` | GPU → CPU |
| `update_actor` (line 917) | `load_fsdp_model_to_gpu` | CPU → GPU |
| update_actor 후 (line 950) | `offload_fsdp_model_to_cpu` | GPU → CPU |

**iteration당 최소 6회 전송** (3회 load + 3회 offload)

### 7.3 전송 시간 추산 (H100, PCIe Gen5)

```
FSDP sharded weights (bf16): ~4.3 GB per GPU
PCIe Gen5 실효 대역폭: ~50 GB/s
1회 전송: 4.3 / 50 ≈ 0.09초
6회: ~0.5초/iteration (param만)

optimizer까지 offload 시:
  optimizer states: ~17.1 GB per GPU
  추가 전송: ~0.34초 × 2 = ~0.68초
  총: ~1.2초/iteration
```

### 7.4 판단 기준

```
offload 이득 = generation 속도 향상 (더 큰 KV cache)
offload 비용 = CPU↔GPU 전송 시간 (iteration당 고정)

판단: offload 이득 > offload 비용  → offload 활성화
```

| 모델 크기 | FSDP GPU 점유 비율 | offload 권장 |
|:---------:|:------------------:|:----------:|
| ~8B (현재) | ~27% (21.4/80) | **불필요** — 점유 비율 낮아 KV cache 여유 충분 |
| ~32B | ~50%+ | **권장** — vLLM 공간 부족 |
| ~70B+ | ~70%+ | **필수** — offload 없이는 vLLM 불가 |

현재 Qwen3-8B 설정에서는 **offload=False + util=0.65가 최적**이다.

---

## 8. 각 페이즈별 메모리 상세

### 8.1 Generation (Rollout)

```
실행: vLLM (wake 상태)
GPU 점유:
  ├─ FSDP actor weights + optimizer: 21.4 GB (상주)
  ├─ vLLM model weights: 17.1 GB (wake_up으로 복구)
  ├─ vLLM KV cache: ~30.9 GB (util=0.6 기준)
  └─ overhead: ~3 GB
  ──────────────
  합계: ~72.4 GB / 80 GB (여유 ~7.6 GB)
```

모드 전환:
1. `rollout_mode()` → FSDP state_dict() 추출 → vLLM wake_up(weights) → update_weights → vLLM wake_up(kv_cache)
2. 생성 완료 후 `trainer_mode()` → vLLM sleep → empty_cache

### 8.2 Compute Log Prob (Old Policy)

```
실행: FSDP Actor (torch.no_grad)
GPU 점유:
  ├─ FSDP actor weights: 4.3 GB (all-gather 시 일시적으로 전체 크기)
  ├─ FSDP optimizer states: 17.1 GB (상주)
  ├─ Forward 중간 텐서: 수 GB (no_grad이므로 activation 미저장)
  └─ vLLM: 0 GB (sleeping)
  ──────────────
  합계: ~25 GB 내외 (메모리 여유 충분)
```

`log_prob_micro_batch_size_per_gpu=32`로 설정 가능하며, no_grad이므로 activation 미저장으로 batch 크게 잡아도 안전하다.

### 8.3 Compute Ref Log Prob

```
실행: FSDP Ref (torch.no_grad, param_offload=True)
GPU 점유:
  ├─ FSDP actor weights + optimizer: 21.4 GB (상주)
  ├─ FSDP ref weights (CPU→GPU 로드): 4.3 GB (임시)
  ├─ Forward 중간 텐서: 수 GB
  └─ vLLM: 0 GB (sleeping)
  ──────────────
  합계: ~30 GB 내외
```

Ref 모델은 `param_offload=True`이므로 `compute_ref_log_prob` 호출 시에만 GPU에 올라왔다가 완료 후 즉시 CPU로 내려간다.

### 8.4 Update Actor (Backward — 메모리 피크)

```
실행: FSDP Actor (gradient ON)
GPU 점유:
  ├─ FSDP actor weights: 4.3 GB
  ├─ FSDP optimizer states: 17.1 GB
  ├─ Gradients: 4.3 GB (backward 중 생성)
  ├─ Activations: ppo_micro_batch_size에 비례 (피크 지점)
  └─ vLLM: 0 GB (sleeping)
  ──────────────
  고정: ~25.7 GB + Activations (가변)
```

**OOM이 가장 자주 발생하는 지점이다.** Micro batch 루프 안에서 forward → backward가 쌍으로 실행되므로, 동시에 존재하는 activation은 1개 micro batch 분량:

```
activation ≈ micro_batch_size × seq_len × hidden_dim × num_layers × dtype_size
```

`ppo_micro_batch_size_per_gpu`를 줄이면 activation 크기가 줄어들어 OOM을 해소할 수 있다 (학습 결과는 동일, gradient accumulation으로 보상).

### 8.5 페이즈별 피크 메모리 비교

```
80 GB ┬─────────────────────────────────────────────────────────
      │
      │  ████████████████████████████████████████████████████ ← ② rollout (~72 GB)
      │  ██████████████████████████████████████████████ ← ⑤ update_actor (~60 GB*)
      │  ████████████████████ ← ④ ref_log_prob (~30 GB)
      │  █████████████████ ← ③ compute_log_prob (~25 GB)
      │  ████████████████ ← ⑥ trainer_mode 복귀 (~21 GB)
      │
 0 GB ┴─────────────────────────────────────────────────────────
         ①     ②       ③        ④         ⑤          ⑥

* update_actor의 피크는 micro_batch_size에 따라 크게 달라짐
```

---

## 9. free_cache_engine과 enable_sleep_mode

### 9.1 두 설정의 관계

| 설정 | 디폴트 | 역할 |
|------|:-----:|------|
| `free_cache_engine` | `True` | rollout 종료 후 vLLM의 weights/KV cache를 해제할지 결정 |
| `enable_sleep_mode` | `True` | vLLM 엔진 초기화 시 sleep mode 기능 자체를 활성화할지 결정 |

`free_cache_engine=True`이면 `trainer_mode()`에서 `self.rollout.release()` → `inference_engine.sleep(level=N)` 호출.
`free_cache_engine=False`이면 vLLM이 항상 GPU를 점유하여 학습과 생성 메모리가 동시에 존재해야 한다 (사실상 OOM 위험).

### 9.2 enable_sleep_mode=False일 때의 차이

```python
# vllm_async_server.py:270-273
if not self.config.enable_sleep_mode:
    from verl.utils.device import set_expandable_segments
    set_expandable_segments(True)
```

sleep mode가 비활성화되면 `expandable_segments`를 켜서 PyTorch의 CUDA 메모리 할당자가 동적으로 메모리를 확장할 수 있게 한다. 이 경우 vLLM과 학습이 동적으로 메모리를 공유하지만, 예측 불가능한 OOM이 발생할 수 있어 권장되지 않는다.

---

## 10. 실전 설정 가이드 요약

### 10.1 Qwen3-8B / 4× H100 80GB 권장 설정

```bash
# FSDP offload — 8B 모델은 불필요
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

# Ref 모델은 offload 권장 (verl 공식 가이드)
actor_rollout_ref.ref.fsdp_config.param_offload=True

# vLLM 메모리 — 0.65 권장 (optimizer states 생성 후 여유 ~6.6 GB)
actor_rollout_ref.rollout.gpu_memory_utilization=0.65

# OOM 방지 — micro batch로 activation 크기 제어
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4

# Gradient checkpointing — activation 메모리 대폭 절감
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

### 10.2 모델 크기별 전략 가이드

| 모델 크기 | param_offload | optimizer_offload | util 범위 | 핵심 전략 |
|:---------:|:------------:|:-----------------:|:---------:|----------|
| ≤8B | False | False | 0.6–0.65 | offload 불필요, 공존 가능 |
| ~32B | True | False | 0.7–0.8 | param만 offload로 KV cache 확보 |
| ~70B+ | True | True | 0.8–0.9 | 전체 offload 필수 |

### 10.3 메모리 모니터링 체크리스트

1. **첫 iteration 결과를 신뢰하지 말 것** — optimizer states 미생성으로 메모리 여유가 과대 보고됨
2. **2~3번째 iteration의 rollout 중 peak**을 기준으로 util 결정
3. `nvidia-smi -l 1`로 실시간 모니터링하며 peak 확인
4. WandB의 `perf/max_memory_allocated_gb`는 **update_actor 페이즈만** 반영함에 주의
