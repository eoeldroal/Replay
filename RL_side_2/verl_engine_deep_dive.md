# verl 강화학습 엔진 심층 분석

## 1. 학습 파이프라인 구조

verl의 RLHF/GRPO 학습 루프는 하나의 스텝 안에서 다음 페이즈를 순차적으로 실행한다.

```
Generation (rollout) → Reward → Log Prob → Ref Policy → Values → Advantage → Actor Update → Critic Update
```

### 1.1 각 페이즈의 역할과 실행 방식

| 페이즈 | 실행 주체 | 역할 | backward 여부 |
|--------|-----------|------|:---:|
| Generation | vLLM (자체 스케줄러) | 프롬프트로부터 응답 생성 | 없음 |
| Reward | 보상 함수 또는 RM | 생성된 응답에 점수 부여 | 없음 |
| Log Prob | FSDP actor (`torch.no_grad()`) | PPO ratio 계산용 old_log_prob 산출 | 없음 |
| Ref Policy | FSDP ref 모델 | KL 페널티 계산용 ref_log_prob 산출 | 없음 |
| Values | FSDP critic | 밸류 추정 (GAE 계산용) | 없음 |
| Advantage | 드라이버 (CPU) | GAE/GRPO 어드밴티지 계산 | 없음 |
| Actor Update | FSDP actor (gradient ON) | PPO 정책 손실 계산 + backprop | **있음** |
| Critic Update | FSDP critic (gradient ON) | 밸류 손실 계산 + backprop | **있음** |

### 1.2 워커 공유 구조

`generate_sequences`, `compute_log_prob`, `update_actor`는 모두 **같은 `actor_rollout_wg` 워커 그룹** — 즉 동일한 GPU 프로세스에서 실행된다. Hybrid Engine 구조에서 rollout 모드와 trainer 모드를 전환하며 동작한다.

---

## 2. PPO의 3단계 배치 계층 구조

PPO에는 세 가지 배치 개념이 있으며, 각각 존재하는 이유가 다르다.

### 2.1 계층 정의

| 계층 | 파라미터 | 결정하는 것 | 영향 |
|------|---------|-----------|------|
| **Batch** | `data.train_batch_size` | 한 스텝에서 rollout으로 수집하는 전체 데이터량 | 데이터 수집 규모 |
| **Mini Batch** | `actor.ppo_mini_batch_size` | 한 번의 `optimizer.step()`에 사용되는 샘플 수 | **수렴/최적화에 영향** |
| **Micro Batch** | `actor.ppo_micro_batch_size_per_gpu` | 한 번의 forward+backward에 GPU에 올리는 샘플 수 | **메모리에만 영향** |

### 2.2 코드 구조 (`dp_actor.py`)

```python
mini_batches = data.split(ppo_mini_batch_size)           # 배치 → 미니배치 분할

for _ in range(ppo_epochs):                               # PPO epoch 반복
    for mini_batch in mini_batches:                       # 미니배치 순회
        micro_batches = mini_batch.split(ppo_micro_batch_size_per_gpu)

        optimizer.zero_grad()                              # gradient 초기화
        for micro_batch in micro_batches:                 # 마이크로배치 순회
            outputs = forward(micro_batch)                #   forward (activation 저장)
            loss.backward()                               #   backward (activation 소비, gradient 누적)
        optimizer.step()                                  # 미니배치 하나당 1번 step
```

### 2.3 예시: batch=128, mini_batch=32, micro_batch=8 (1 GPU)

```
전체 배치 128개
├── mini_batch ①: 32개 ──→ optimizer.step() ①
│   ├── micro 8개: forward → backward (gradient 누적)
│   ├── micro 8개: forward → backward (gradient 누적)
│   ├── micro 8개: forward → backward (gradient 누적)
│   └── micro 8개: forward → backward (gradient 누적)
│
├── mini_batch ②: 32개 ──→ optimizer.step() ②
│   └── (micro 4회 반복)
│
├── mini_batch ③: 32개 ──→ optimizer.step() ③
└── mini_batch ④: 32개 ──→ optimizer.step() ④
```

### 2.4 Mini Batch vs Micro Batch의 핵심 차이

- **Mini batch 크기를 줄이면**: step 횟수 증가 → gradient 분산 변화 → **학습 역학이 바뀜**
- **Micro batch 크기를 줄이면**: accumulation 횟수 증가 → 수학적으로 동일한 gradient → **시간만 증가**

Micro batch는 순수하게 **GPU 메모리 제약을 우회**하기 위한 장치이다.

---

## 3. PPO 업데이트의 두 번의 Forward Pass

PPO 알고리즘의 클리핑 목적함수는 다음과 같다:

```
ratio = exp(log_π_θ(a|s) - log_π_old(a|s))
L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

이를 위해 동일한 데이터에 대해 **같은 `_forward_micro_batch` 함수를 2번 호출**한다.

### 3.1 1차 Forward: `compute_log_prob`

```python
# dp_actor.py:471
with torch.no_grad():                          # gradient 추적 OFF
    outputs = self._forward_micro_batch(...)    # activation 저장 안 됨
```

- **목적**: `old_log_prob` 계산 (업데이트 이전 정책 기준의 고정값)
- **배치 제어**: `log_prob_micro_batch_size_per_gpu`
- **메모리 특성**: activation이 저장되지 않아 가벼움

### 3.2 2차 Forward: `update_policy`

```python
# dp_actor.py:581
outputs = self._forward_micro_batch(...)        # gradient 추적 ON → activation 자동 저장
...
loss.backward()                                 # activation 소비하며 gradient 계산
```

- **목적**: 현재 파라미터 기준의 `new_log_prob` 계산 → PPO loss → backprop
- **배치 제어**: `ppo_micro_batch_size_per_gpu`
- **메모리 특성**: PyTorch autograd가 backward를 위해 중간 텐서를 자동 보관 → **메모리 피크 발생 지점**

### 3.3 on-policy 최적화: forward 1회만으로 충분한 경우

`dp_actor.py:543,591-594`에 중요한 분기가 있다:

```python
on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

if on_policy:
    old_log_prob = log_prob.detach()     # 방금 구한 값을 상수로 사용 (ratio ≈ 1)
else:
    old_log_prob = model_inputs["old_log_probs"]  # 사전 계산된 고정값 사용
```

미니배치가 1개이고 epoch도 1이면, 업데이트 forward에서 나온 `log_prob`을 `detach()`하여 바로 `old_log_prob`으로 사용한다. 이 경우 **별도의 `compute_log_prob` 페이즈가 사실상 불필요**하며, forward가 1회로 줄어든다. `ppo_mini_batch_size = train_batch_size`이고 `ppo_epochs = 1`인 설정이 이에 해당한다.

### 3.4 왜 2번 돌리는가 (on_policy=False일 때)

`old_log_prob`은 업데이트 **시작 전** 파라미터(θ₀) 기준값이어야 한다. `ppo_epochs > 1`이거나 미니배치가 여러 개이면 Actor Update 중 파라미터가 계속 갱신되므로, 2차 forward의 `new_log_prob`은 반복마다 달라진다. 반면 1차에서 구한 `old_log_prob`은 θ₀ 기준의 상수로 고정되어 재사용된다.

---

## 4. 미니배치 간 Off-Policy 문제와 PPO 클리핑

### 4.1 미니배치를 거치며 발생하는 off-policy

`old_log_probs`는 전체 배치에 대해 **1번만 계산**되고, 모든 미니배치와 PPO epoch에서 고정값으로 재사용된다. 그런데 각 미니배치의 `optimizer.step()` 후 파라미터가 변하므로, **미니배치 2번째부터는 엄밀히 off-policy**이다.

```
compute_log_prob(전체 배치)  →  old_log_probs (θ₀ 기준, 1회 계산)

update_policy:
  mini_batch ①: forward(θ₀) → ratio = π_θ₀/π_old(θ₀) ≈ 1.0  →  step → θ₁
  mini_batch ②: forward(θ₁) → ratio = π_θ₁/π_old(θ₀) ≠ 1.0  →  step → θ₂
  mini_batch ③: forward(θ₂) → ratio = π_θ₂/π_old(θ₀)         →  step → θ₃
  mini_batch ④: forward(θ₃) → ratio = π_θ₃/π_old(θ₀)         →  step → θ₄
                                        ↑           ↑
                                    계속 변함      고정값
```

### 4.2 PPO 클리핑이 이를 제어하는 방식

ratio가 `[1-ε, 1+ε]` 범위를 벗어나면 gradient가 차단되어, 한 스텝 안에서 정책이 너무 크게 변하는 것을 방지한다:

```
L = min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)
```

미니배치마다 `old_log_probs`를 재계산하면 더 정확하겠지만, 매번 추가 forward가 필요하여 계산 비용이 2배가 된다. PPO는 클리핑으로 이 비용 없이 안정성을 확보하는 것이 핵심 설계이다. 이것은 버그가 아니라 **PPO 원논문의 의도된 설계**이다.

---

## 5. 3가지 정책과 Off-Policy 보정 (Rollout Correction)

### 5.1 verl이 구분하는 3가지 정책의 log probability

verl은 off-policy 보정을 위해 3가지 정책의 log probability를 구분한다:

| 정책 | 키 이름 | 계산 주체 | 시점 |
|------|---------|----------|------|
| π_rollout | `rollout_log_probs` | vLLM (생성 시) | 응답 생성 시점 |
| π_old | `old_log_probs` | FSDP actor (`no_grad`) | 업데이트 직전 |
| π_θ | `log_probs` | FSDP actor (gradient ON) | 업데이트 중 (매 forward마다 변화) |

### 5.2 vLLM의 rollout_log_probs 반환

`vllm_async_server.py:531-534`에서 vLLM은 생성 시점의 log probability를 함께 반환할 수 있다:

```python
token_ids = final_res.outputs[0].token_ids
log_probs = None
if sampling_params.logprobs is not None:
    log_probs = [logprobs[token_ids[i]].logprob
                 for i, logprobs in enumerate(final_res.outputs[0].logprobs)]
```

이 값은 `agent_loop.py:742`에서 `rollout_log_probs`라는 키로 배치에 포함된다:

```python
if inputs[0].response_logprobs is not None:
    optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
```

### 5.3 On-Policy vs Off-Policy 시나리오

**표준 on-policy (현재 설정)**: rollout 직후 바로 업데이트하므로 π_rollout ≈ π_old. `rollout_log_probs`는 불필요하고, `old_log_probs`만으로 충분하다.

**Fully async / one-step-off policy**: rollout과 업데이트 사이에 파라미터가 갱신되므로 π_rollout ≠ π_old. IS(Importance Sampling) 보정에 `rollout_log_probs`가 필요하다. `use_rollout_log_probs=True`와 `rollout_correction` 설정을 활성화해야 한다.

### 5.4 Bypass Mode

`rollout_correction.bypass_mode=True` 설정 시, `old_log_probs = rollout_log_probs`로 대체하여 3-정책 체계를 2-정책으로 단순화한다 (`rollout_corr_helper.py:922-945`). 이 경우 `compute_log_prob` 페이즈를 건너뛸 수 있어 계산 비용이 줄어든다.

---

## 6. Activation 메모리와 Micro Batch Size

### 6.1 Activation이란

Forward pass에서 각 레이어가 backward 시 gradient 계산을 위해 보관하는 **중간 hidden state 텐서**이다. `torch.no_grad()` 없이 forward를 실행하면 PyTorch autograd가 자동으로 저장한다.

```
activation ≈ micro_batch_size × seq_len × hidden_dim × num_layers × dtype_size
```

별도의 "activation 계산 단계"가 존재하는 것이 아니라, gradient 추적이 켜진 상태의 forward pass에서 **자동으로 저장되는 부산물**이다.

### 6.2 forward와 backward가 micro_batch 루프 안에서 쌍으로 실행됨

`loss.backward()`가 micro_batch 루프 **안**에 있다는 것이 핵심이다. 한 micro_batch의 activation이 backward로 소비된 후에야 다음 micro_batch의 forward가 시작된다. 따라서 **동시에 메모리에 존재하는 activation은 항상 1개 micro_batch 분량**뿐이다.

```
micro_batch=32 (accumulation 2회):
  [forward 32개] activation 32개분 ████████████████ 피크
  [backward     ] activation 소비   ░░░░░░░░░░░░░░░ 해제
  [forward 32개] activation 32개분 ████████████████ 피크
  [backward     ] activation 소비   ░░░░░░░░░░░░░░░ 해제
  [step         ] in-place 갱신

micro_batch=16 (accumulation 4회):
  [forward 16개] activation 16개분 ████████ 피크 (절반)
  [backward     ] 해제              ░░░░░░░
  [forward 16개] ████████
  [backward     ] ░░░░░░░
  [forward 16개] ████████
  [backward     ] ░░░░░░░
  [forward 16개] ████████
  [backward     ] ░░░░░░░
  [step         ] in-place 갱신
```

### 6.3 Backprop 시 GPU 메모리 구성

```
┌─────────────────────────────────────────────┐
│  ① Model Parameters        ← 고정           │
│  ② Optimizer States         ← 고정           │
│  ③ Gradients                ← 고정           │
│  ④ Activations              ← micro_batch_size에 비례  │
└─────────────────────────────────────────────┘
```

- ①②③은 배치 크기와 무관하게 고정이다.
- ④만이 `micro_batch_size`에 정비례하여 변한다.
- OOM은 대부분 backward 시 발생한다. 아직 소비되지 않은 activation과 이미 생성된 gradient가 **동시에 존재**하는 순간이 피크이기 때문이다.
- `optimizer.step()`은 이미 할당된 메모리 위에서 in-place 연산만 수행하므로 추가 메모리가 거의 없다. (예외: 첫 번째 step에서 Adam의 `exp_avg`, `exp_avg_sq`가 최초 할당됨)

### 6.4 전체 메모리 감소량은 정비례가 아님

activation 메모리 자체는 micro_batch_size에 정비례하지만, **전체 GPU 메모리**는 고정 비용(param + optimizer + gradient)이 있으므로 정비례보다 완만하게 감소한다.

```
전체 메모리 = 고정 부분 + 가변 부분(activation)

예시 (Qwen3-8B, 4 GPU FSDP 기준 개략치):
  고정 ≈ ~15GB
  micro_batch=32: 15GB + ~45GB = 60GB  ← OOM
  micro_batch=16: 15GB + ~22GB = 37GB  ← 통과
  micro_batch=8:  15GB + ~11GB = 26GB
```

### 6.5 Gradient Accumulation의 원리

`ppo_micro_batch_size_per_gpu`를 줄이면 gradient accumulation 횟수가 늘어나지만, **수학적으로 동일한 gradient**가 계산된다.

```
micro_batch=32, 1회:
  grad = ∂Loss(x₁...x₃₂) / ∂w

micro_batch=16, 2회:
  grad₁ = ∂Loss(x₁...x₁₆) / ∂w    ← 계산 후 activation 해제
  grad₂ = ∂Loss(x₁₇...x₃₂) / ∂w   ← 새로 계산
  grad = grad₁ + grad₂              ← 수학적으로 동일
```

### 6.6 vLLM Generation과의 독립성

vLLM의 generation 결과물은 **토큰 ID (정수 배열)** 뿐이다. vLLM 내부의 KV cache와 activation은 생성 완료 후 전부 해제된다. PPO 업데이트에 필요한 activation은 FSDP 모델이 **새로 forward를 돌려서 생성**하는 것이므로, generation 배치 크기(`train_batch_size × rollout.n`)와 PPO micro batch size는 완전히 독립적이다.

```
vLLM 결과:      256 × 8 × 8192 × 4byte ≈ 64MB        (토큰 ID, 정수)
PPO activation: 32 × 8192 × 4096 × 36 × 2byte ≈ 수십 GB (hidden state, float16)
```

---

## 7. 페이즈별 배치 제어 파라미터

| 페이즈 | 파라미터 | backward | OOM 위험 |
|--------|---------|:---:|:---:|
| Generation | vLLM `gpu_memory_utilization` | 없음 | vLLM 자체 관리 |
| Log Prob | `rollout.log_prob_micro_batch_size_per_gpu` | 없음 | 낮음 |
| Ref Policy | `ref.log_prob_micro_batch_size_per_gpu` | 없음 | 낮음 |
| **Actor Update** | **`actor.ppo_micro_batch_size_per_gpu`** | **있음** | **높음** |
| Critic Update | `critic.ppo_micro_batch_size_per_gpu` | 있음 | 높음 |

Backprop OOM 대응 시 `ppo_micro_batch_size_per_gpu`를 줄이는 것이 정확한 타겟이다.

---

## 8. 네이티브 로깅 시스템

### 8.1 지원 백엔드 (9종)

`verl/utils/tracking.py`의 `Tracking` 클래스가 통합 인터페이스를 제공한다.

| 백엔드 | 설정값 | 비고 |
|--------|--------|------|
| Console | `"console"` | 기본 터미널 출력 |
| WandB | `"wandb"` | 인터랙티브 테이블, 프록시 지원 |
| MLflow | `"mlflow"` | 키 자동 정제, 비동기 로깅 |
| TensorBoard | `"tensorboard"` | 실시간 메트릭 |
| SwanLab | `"swanlab"` | 클라우드/로컬 모드 |
| ClearML | `"clearml"` | 모델 관리 통합 |
| TrackIO | `"trackio"` | 통합 실험 추적 |
| File | `"file"` | JSONL 파일 |
| VEMLP WandB | `"vemlp_wandb"` | Volcano Engine 플랫폼용 |

설정 예: `trainer.logger='["console","wandb"]'`

### 8.2 자동 수집 메트릭 (설정 불필요)

**페이즈별 타이밍** — `marked_timer` 컨텍스트 매니저로 모든 페이즈의 실행 시간을 자동 측정:
- `timing_s/gen`, `timing_s/reward`, `timing_s/old_log_prob`, `timing_s/update_actor` 등
- `timing_per_token_ms/{phase}` (토큰당 시간)

**MFU (Model FLOPs Utilization)** — `FlopsCounter`가 모델별 이론 FLOPS 대비 실제 사용률 추정:
- `perf/mfu/actor`, `perf/mfu/actor_infer`, `perf/mfu/critic`

**처리량**:
- `perf/throughput` (GPU당 초당 토큰 수)
- `perf/time_per_step`, `perf/total_num_tokens`

### 8.3 GPU 메모리 메트릭의 한계

WandB에 기록되는 GPU 메모리 메트릭은 **Actor Update 페이즈에서만** 수집된다:

```python
# fsdp_workers.py:936-938 (update_actor 메서드 내부)
metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
```

다른 페이즈(generation, log_prob 등)는 `log_gpu_memory_usage()`로 Python DEBUG 로그에만 출력하며, WandB에는 기록되지 않는다. `engine_workers.py:139-140`의 메모리 메트릭은 주석 처리되어 있다.

**페이즈별 GPU 메모리 비교 추적 기능은 현재 네이티브로 지원되지 않는다.**

### 8.4 페이즈별 메모리 추적 구현 시 주의사항

`reset_peak_memory_stats()`를 각 페이즈 시작 전에 호출하면 기존 `perf/max_memory_allocated_gb`의 의미가 변경된다 (전체 피크 → 해당 페이즈만의 피크). 기존 로직을 건드리지 않으려면, `memory_allocated()` (현재 할당량 스냅샷)을 페이즈 진입/종료 시점에 읽는 방식이 안전하다.

### 8.5 모델 응답 로깅 (모델 진화 추적)

**학습 롤아웃 저장**: `trainer.rollout_data_dir` 설정 시 매 스텝 `{step}.jsonl` 파일 생성

**검증 응답 저장**: `trainer.log_val_generations=N` 설정 시 검증마다 N개 샘플을 WandB 테이블 등에 기록

**롤아웃 뷰어**: `python scripts/rollout_viewer.py <dir>` — 저장된 JSONL 탐색용 TUI 도구

### 8.6 프로파일러

`global_profiler.tool` 설정으로 활성화:

| 도구 | 용도 |
|------|------|
| `torch` | PyTorch 네이티브 프로파일러 |
| `nsys` | NVIDIA Nsight Systems |
| `torch_memory` | CUDA 메모리 스냅샷 |
| `npu` | Ascend NPU용 |

---

## 9. 실행 스크립트 설정 가이드

### 9.1 Backprop OOM 대응

```bash
# ppo_micro_batch_size_per_gpu를 줄인다 (학습 결과 동일, 시간만 증가)
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16   # 32 → 16
```

### 9.2 모델 진화 추적 활성화

```bash
trainer.rollout_data_dir='./rollout_logs'
trainer.validation_data_dir='./validation_logs'
trainer.log_val_generations=10
```

### 9.3 Off-Policy 보정 활성화 (비동기 학습 시)

```bash
actor_rollout_ref.actor.use_rollout_log_probs=True
# rollout_correction 관련 설정 추가 필요
```
