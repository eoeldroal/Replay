# Revising_8: rev7(Non-Base) 전환 + M2/ZVP 경로 최적화 정리

## 1) 목적

- Qwen3 Base 계열에서 M2+ZVP 우위가 약한 구간이 관측되어, `Qwen/Qwen3-1.7B`(Non-Base)로 전환한 `rev7` 실험 축을 추가.
- 동시에 M2 후보 평가(`new_log_probs` 재계산) 구간의 병목을 줄이기 위해, 기존 로직은 유지한 채 계산 경량화/병렬 효율 개선을 반영.

---

## 2) 실행 스크립트 변경 (rev7)

신규 실행 파일:

- `/home/work/DDAI_revised/verl/RL_side_3/grpo-qwen3-1.7b-s8-m2-replay-rev7.sh`

핵심 설정:

- 모델: `Qwen/Qwen3-1.7B` (기존 Base에서 변경)
- 실험명: `grpo-qwen3-1.7b-s8-m2-replay-rev7-tau001-b1024`
- Replay:
  - `+algorithm.m2_replay.enable=true`
  - `+algorithm.m2_replay.tau=0.001`
  - `+algorithm.m2_replay.buffer.max_query_groups=1024`
  - `+algorithm.m2_replay.schedule.replay_target_groups=128`
  - `+algorithm.m2_replay.schedule.one_turnover_gate=true`
  - `+algorithm.m2_replay.selection.mode=zvp_recency`
  - `+algorithm.m2_replay.selection.zvp_use_recency=false`
  - `+algorithm.m2_replay.selection.zvp_ema_alpha=1.0`
  - `+algorithm.m2_replay.selection.logprob_groups_per_chunk=64`
  - `+algorithm.m2_replay.selection.log_prob_micro_batch_size_per_gpu=64`
- 생성/검증:
  - `data.max_response_length=3072`
  - `rollout.max_model_len=4096`
  - `val_kwargs.n=4`, `top_k=-1`, `top_p=1.0`, `temperature=1.0`
  - `trainer.total_training_steps=250`

---

## 3) 코드 레벨 변경 요약

### 3.1 M2 logprob 재계산 경량화

적용 파일:

- `/home/work/DDAI_revised/verl/verl/trainer/ppo/ray_trainer.py`

핵심:

- M2 후보 평가용 logprob 계산 호출에서 `calculate_entropy=False`를 명시해 불필요 entropy 계산 제거.
- M2 전용 micro-batch 오버라이드 경로 추가:
  - `selection.log_prob_micro_batch_size_per_gpu` 파라미터를 읽어 M2 평가 경로에만 적용 가능.
- 관련 메트릭 로깅 추가:
  - `m2_replay/selection/log_prob_micro_batch_size_per_gpu`

### 3.2 Worker 단 오버라이드 지원

적용 파일:

- `/home/work/DDAI_revised/verl/verl/workers/fsdp_workers.py`
- `/home/work/DDAI_revised/verl/verl/workers/megatron_workers.py`

핵심:

- `data.meta_info["log_prob_micro_batch_size_override"]`를 읽어 logprob 계산 마이크로배치 크기 오버라이드.
- 유효성 검사(정수, `>0`) 추가.
- `calculate_entropy` 플래그를 meta_info 기반으로 반영.

### 3.3 One-turnover gate (초기 과스캔 완화)

적용 파일:

- `/home/work/DDAI_revised/verl/verl/trainer/ppo/ray_trainer.py`

핵심:

- `schedule.one_turnover_gate` 옵션 추가.
- 버퍼가 충분히 한 바퀴(turnover) 누적되기 전에는 replay 선택을 지연해 초기 full-scan/저효율 구간을 줄이는 목적.
- 메트릭:
  - `m2_replay/gating/one_turnover_enabled`

### 3.4 기존 안정화 로직 유지

- `fixed_total_floor_groups`, `replay_trimmed_for_floor` 관련 floor 안정화 로직은 유지.
- ZVP 관련 선택 로깅(`used_zvp_*`)도 유지.

---

## 4) 로직 불변성 (중요)

- PPO/GRPO의 핵심 업데이트 순서(생성→보상→old_log_prob→adv→replay선택→actor update)는 변경하지 않음.
- M2 게이팅 기준(`m2 <= tau`) 자체는 변경하지 않음.
- 변경은 **M2 후보 평가의 실행 효율/제어 옵션** 중심이며, 알고리즘 본체를 바꾸지 않음.

---

## 5) 검증/테스트

이전 검증 기록:

- `conda activate verl` (환경 변경 없이 사용)

기록 내용:

- 컴파일 체크:
  - `ray_trainer.py`, `fsdp_workers.py`, `megatron_workers.py` compileall 통과
- 단위 테스트:
  - `tests/trainer/ppo/test_m2_replay_core_on_cpu.py`
  - `tests/trainer/ppo/test_m2_replay_chunk_batch_on_cpu.py`
  - `tests/trainer/ppo/test_m2_replay_replay_filter_on_cpu.py`
  - `tests/trainer/ppo/test_m2_replay_adapter_on_cpu.py`
  - 결과: `13 passed`

주의:

- 위 테스트 결과는 문서 작성 시점 기록이다.
- 현재 워크트리 기준으로는 관련 구현 파일이 추가 수정 중이므로 재실행 재확인이 필요하다.

---

## 6) 기대 효과 / 리스크

기대 효과:

- M2 후보 logprob 재계산 비용 감소(특히 entropy 제거 효과).
- M2 평가 경로의 GPU 처리 단위 제어가 가능해져 튜닝 여지 확대.
- 초기 버퍼 단계에서 불필요한 과스캔 완화(one-turnover gate).

리스크:

- `log_prob_micro_batch_size_per_gpu`를 과도하게 키우면 OOM 위험.
- one-turnover gate 활성 시 초반 replay 개입이 늦어져 초기 학습 곡선이 달라질 수 있음.
- Non-Base 전환 자체가 정책 분포를 바꾸므로, Base 대비 직접 수치 비교는 동일 조건 재검증이 필요.
- ZVP 가설 검증에 필요한 source-separated actor 메트릭은 아직 직접 로깅되지 않음.

---

## 7) 현재 권장 관측 지표

- 시간:
  - `timing_s/m2_select_logprob_eval`
  - `timing_s/update_actor`
  - `timing_s/step`
- replay 품질/활성:
  - `m2_replay/pass1/acceptance_rate`
  - `m2_replay/selection/replay_used`
  - `m2_replay/selection/used_zvp_mean`
  - `m2_replay/selection/log_prob_micro_batch_size_per_gpu`
- 안정성:
  - `actor/grad_norm`
  - `actor/entropy`

---

## 8) 결론

- Rev8 단계의 핵심은 “알고리즘 변경”이 아니라, **M2/ZVP 경로의 실효성 확보를 위한 실행 효율화 + 실험 축 전환(Non-Base)** 이다.
- 즉, 동일한 M2/ZVP 철학을 유지하면서, 병목을 줄이고(측정 가능), 비교 가능성이 높은 실험 셋업(rev7)을 확보한 단계로 정리한다.
