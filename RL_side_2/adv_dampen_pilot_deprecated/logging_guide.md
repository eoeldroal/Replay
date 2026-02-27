# Advantage Dampening Pilot — Logging Guide

이 문서는 파일럿 실험에서 수집되는 모든 로깅 채널과 데이터를 정리한다.


## 1. 로깅 채널 전체 지도

파일럿 실험은 **4개의 독립된 로깅 채널**을 통해 데이터를 수집한다.

```
                        학습 루프
                           │
         ┌─────────────────┼─────────────────────┐
         │                 │                      │
    [A] wandb          [B] Rollout Trace      [C] Policy Diag
    실시간 스칼라       토큰 단위 생성 기록    IS ratio 진단
    (console 병행)     (.jsonl, weave)        (.pt 파일)
         │                 │                      │
         │                 │                      │
    compute_data_metrics   rollout worker     compute_policy_loss_vanilla
    metric_utils.py        vllm engine       core_algos.py:_PolicyDiagWriter
                                                  │
                                            ─ ─ ─ ┤
                                                  │
                                          [D] Dampening Diag
                                          advantage 댐핑 진단
                                          (.pt 파일)
                                                  │
                                       compute_grpo_dampened_outcome_advantage
                                       core_algos.py:_DampenDiagWriter
```


## 2. 채널별 상세

### [A] wandb + console — 실시간 스칼라 메트릭

**설정:**
```
trainer.logger='["console","wandb"]'
trainer.project_name='AdvDampen.Pilot'
trainer.experiment_name='grpo_dampened_tau2'  # 또는 'grpo_vanilla'
```

**매 학습 스텝마다 기록되는 메트릭:**

| 카테고리 | 메트릭 키 | 설명 |
|---------|----------|------|
| **Advantage** | `critic/advantages/mean` | 배치 내 유효 토큰 advantage 평균 |
| | `critic/advantages/max` | 배치 내 최대 advantage |
| | `critic/advantages/min` | 배치 내 최소 advantage |
| **보상** | `critic/score/mean`, `max`, `min` | 시퀀스별 보상 통계 |
| | `critic/rewards/mean`, `max`, `min` | token_level_rewards 합산 통계 |
| | `critic/returns/mean`, `max`, `min` | 리턴 값 통계 |
| **정책 손실** | `actor/pg_loss` | 정책 그래디언트 손실 |
| | `actor/pg_clipfrac` | PPO 클리핑 비율 (상한) |
| | `actor/pg_clipfrac_lower` | PPO 클리핑 비율 (하한) |
| | `actor/ppo_kl` | 정책 KL divergence |
| | `actor/entropy` | 정책 엔트로피 |
| | `actor/grad_norm` | 그래디언트 노름 |
| **분산 프록시** | `variance_proxy/expected_a_squared` | E[A^2] — 그래디언트 분산 추정 |
| | `variance_proxy/proxy2_total_power` | E[A^2 * W(t)] |
| **응답 길이** | `response_length/mean`, `max`, `min` | 응답 토큰 수 통계 |
| | `response_length/clip_ratio` | max_response_length 도달 비율 |
| **처리량** | `perf/total_time`, `perf/tokens_per_sec` | 학습 속도 |

**소스:** `verl/trainer/ppo/metric_utils.py` → `compute_data_metrics()`,
`compute_variance_proxy_metrics()`, `compute_timing_metrics()`

**주의:** wandb에는 dampening 전용 메트릭이 없다. dampening 효과는 [D] Dampening Diag에서 분석한다.


### [B] Rollout Trace — 토큰 단위 생성 기록

**설정:**
```
actor_rollout_ref.rollout.trace.backend=weave
actor_rollout_ref.rollout.trace.token2text=true
actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=1
```

**기록 내용:**
- 각 학습 스텝에서 워커당 1개 샘플의 전체 프롬프트+응답 텍스트
- 토큰별 log probability
- 생성 메타데이터 (길이, 종료 조건 등)

**용도:** 모델이 어떤 텍스트를 생성하는지 정성적으로 확인.
특히 `#### <숫자>` 형식을 학습하는 과정을 추적할 수 있다.


### [C] Policy Diagnostics — IS ratio 진단 (.pt 파일)

**설정:**
```
+actor_rollout_ref.actor.policy_diag_dir=/.../AdvDamp_{dampened,vanilla}_diag
+actor_rollout_ref.actor.policy_diag_flush_freq=256
```

**저장 위치:**
- dampened: `/home/work/DDAI_revised/verl/logs/AdvDamp_dampened_diag/`
- vanilla: `/home/work/DDAI_revised/verl/logs/AdvDamp_vanilla_diag/`

**파일 형식:** `{session_id}_flush{NNNN}_rank{R}.pt`

**저장 텐서:**

| 텐서 | shape | dtype | 설명 |
|------|-------|-------|------|
| `token_ratios` | (N, seq_len) | float16 | 토큰별 IS ratio π_θ/π_old |
| `seq_ratios` | (N,) | float32 | 시퀀스별 IS ratio (기하평균) |
| `mask` | (N, seq_len) | bool | 유효 토큰 마스크 |
| `clipped` | (N, seq_len) | bool | PPO 클리핑 발생 여부 |

**flush 주기:** 256 micro-batch 호출마다.
micro-batch size = 2, GPU 3대 → 한 PPO epoch = 12/(2*3) = 2 micro-batch.
따라서 약 128 학습 스텝마다 1회 flush → 전체 ~1,869 스텝에서 약 14개 파일.

**분석 용도:**
- dampened vs vanilla: 오답 토큰의 IS ratio(π_θ/π_old) 하락 속도 비교
- dampening이 정책 업데이트 강도를 실제로 완화하는지 확인

**소스:** `verl/trainer/ppo/core_algos.py` → `_PolicyDiagWriter` 클래스,
`compute_policy_loss_vanilla()` 내부에서 lazy init + 호출


### [D] Dampening Diagnostics — advantage 댐핑 진단 (.pt 파일)

**설정:** (dampened 스크립트에만 적용)
```
algorithm.adv_dampen_diag_dir=/.../AdvDamp_dampened_adv_diag
algorithm.adv_dampen_diag_flush_freq=64
```

**저장 위치:**
- `/home/work/DDAI_revised/verl/logs/AdvDamp_dampened_adv_diag/`
- vanilla 스크립트에서는 `grpo_dampened` estimator를 사용하지 않으므로 이 채널 자체가 비활성.

**파일 형식:** `{session_id}_flush{NNNN}_rank{R}.pt`

**저장 텐서:**

| 텐서 | shape | dtype | 설명 |
|------|-------|-------|------|
| `adv_raw` | (N,) | float32 | dampening 전 원본 advantage (정규화 후) |
| `correction` | (N,) | float32 | softplus 보정량 (= Â_eff - Â_raw) |
| `adv_eff` | (N,) | float32 | dampening 후 최종 advantage |
| `scores` | (N,) | float32 | 시퀀스별 원시 보상 (0 또는 1) |
| `group_idx` | (N,) | int32 | 그룹 소속 인덱스 (같은 프롬프트 = 같은 그룹) |

**flush 주기:** 64 record 호출마다.
학습 스텝당 1회 호출 → 64 스텝마다 1회 flush → 전체 ~1,869 스텝에서 약 29개 파일.
파일당 크기: N=192 samples × 5 tensors × 4 bytes ≈ 4 KB (극소)

**소스:** `verl/trainer/ppo/core_algos.py` → `_DampenDiagWriter` 클래스,
`compute_grpo_dampened_outcome_advantage()` 내부에서 lazy init + 호출

**비활성 시 오버헤드:** `adv_dampen_diag_dir=None`이면 `record()` 첫 줄에서 즉시 반환.
텐서 복사도, 버퍼 할당도 발생하지 않으므로 성능 영향 없음.


## 3. 사후 분석 레시피

### 3.1 기본 통계 추출

```python
import torch, glob

files = sorted(glob.glob("logs/AdvDamp_dampened_adv_diag/*.pt"))
for f in files:
    d = torch.load(f, weights_only=True)

    # 댐핑 비율: correction이 유의미한(>0.01) 샘플의 비율
    frac = (d["correction"] > 0.01).float().mean().item()

    # 평균 보정량 (댐핑된 샘플만)
    dampened = d["correction"][d["correction"] > 0.01]
    corr_mean = dampened.mean().item() if len(dampened) > 0 else 0.0

    # advantage 범위 비교
    raw_min = d["adv_raw"].min().item()
    eff_min = d["adv_eff"].min().item()

    print(f"{f}: dampened={frac:.1%}, corr_mean={corr_mean:.3f}, "
          f"raw_min={raw_min:.2f} → eff_min={eff_min:.2f}")
```

### 3.2 그룹별 정확도 분포

```python
import torch

d = torch.load("flush_0001.pt", weights_only=True)
g = d["group_idx"].long()
G = g.max().item() + 1

# 그룹별 정확도
group_acc = torch.zeros(G).scatter_add_(0, g, d["scores"]) / \
            torch.zeros(G).scatter_add_(0, g, torch.ones_like(d["scores"]))

print(f"Mean group acc: {group_acc.mean():.3f}")
print(f"High-acc groups (>0.8): {(group_acc > 0.8).float().mean():.1%}")
print(f"All-correct groups (=1.0): {(group_acc == 1.0).float().mean():.1%}")
```

### 3.3 정확도별 댐핑 효과

```python
# 고정확도 그룹의 오답에서 dampening 효과가 가장 클 것으로 예상
wrong_mask = d["scores"] == 0.0  # 오답 샘플
high_acc_groups = (group_acc > 0.8)
in_high_acc = high_acc_groups[g]  # (N,) 해당 샘플이 고정확도 그룹 소속인지

target = wrong_mask & in_high_acc  # 고정확도 그룹의 오답
print(f"고정확도 그룹 오답 수: {target.sum().item()}")
print(f"  raw Â 범위: [{d['adv_raw'][target].min():.2f}, {d['adv_raw'][target].max():.2f}]")
print(f"  eff Â 범위: [{d['adv_eff'][target].min():.2f}, {d['adv_eff'][target].max():.2f}]")
print(f"  평균 보정량: {d['correction'][target].mean():.3f}")
```

### 3.4 IS ratio와 advantage 교차 분석

[C] Policy Diag와 [D] Dampening Diag를 결합하면,
극단적 negative advantage가 IS ratio 변동에 미치는 영향을 직접 관찰할 수 있다.
(단, 두 채널의 flush 시점이 다르므로 시간축 정렬이 필요하다.)


## 4. 로깅 디렉토리 요약

### dampened 실험

| 채널 | 경로 |
|------|------|
| [A] wandb | `AdvDampen.Pilot / grpo_dampened_tau2` |
| [B] Rollout | `logs/AdvDamp_dampened_rollout/` |
| [B] Validation | `logs/AdvDamp_dampened_validation/` |
| [C] Policy Diag | `logs/AdvDamp_dampened_diag/` |
| [D] Dampening Diag | `logs/AdvDamp_dampened_adv_diag/` |

### vanilla 실험

| 채널 | 경로 |
|------|------|
| [A] wandb | `AdvDampen.Pilot / grpo_vanilla` |
| [B] Rollout | `logs/AdvDamp_vanilla_rollout/` |
| [B] Validation | `logs/AdvDamp_vanilla_validation/` |
| [C] Policy Diag | `logs/AdvDamp_vanilla_diag/` |
| [D] Dampening Diag | — (비활성: grpo estimator 사용) |


## 5. 핵심 비교 관점

| 관점 | 채널 | dampened에서 기대하는 차이 |
|------|------|--------------------------|
| 엔트로피 붕괴 | [A] `actor/entropy` | dampened가 더 느리게 하락 |
| advantage 극값 | [A] `critic/advantages/min` | dampened의 min이 -τ 근처에서 안정 |
| 오답 IS ratio | [C] `token_ratios` | dampened에서 π_θ→0 속도가 느림 |
| 댐핑 활성 비율 | [D] `correction` | 학습 후반에 증가 (정확도 상승) |
| 그룹 정확도 | [D] `scores` + `group_idx` | 양쪽 동일해야 공정 비교 |
| 정답률 추이 | [A] `critic/score/mean` | dampened ≥ vanilla (목표) |
