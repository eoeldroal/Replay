# RL_side_3 Agents Knowledge Base

이 문서는 `RL_side_3`에서 우리가 실제로 확인한 사실, 현재 코드베이스의 구현 상태, 실험 운영 시 주의점, 다음 에이전트가 바로 이어서 작업할 수 있게 하는 실전 지식을 정리한 문서다.

목표:
- 긴 대화 로그를 다시 읽지 않아도 현재 상황을 빠르게 파악할 것
- “현재 코드 기준으로 맞는 사실”만 남길 것
- 실행/분석/최적화 시 흔히 헷갈리는 포인트를 미리 해소할 것

---

## 1. 프로젝트 목적

- `verl`의 GRPO 학습에 **M2-gated replay**를 얹어서 off-policy replay를 안전하게 재사용한다.
- 핵심 질문:
  - 왜 `Qwen2.5-Math-1.5B`에서는 replay가 잘 먹히는가
  - 왜 `Qwen3-1.7B` 계열에서는 replay가 baseline보다 열세를 보였는가
  - replay 경로를 baseline `verl` 수준으로 더 효율적이고 안정적으로 만들 수 있는가

---

## 2. 현재 중요한 실행 스크립트

### Qwen2.5-Math

- baseline
  - `/home/work/DDAI_revised/verl/RL_side_3/grpo-qwen25-math-1.5b-s8.sh`
  - `/home/work/DDAI_revised/verl/RL_side_3/main-grpo-qwen25-math-1.5b-s8-baseline.sh`
  - `/home/work/DDAI_revised/verl/RL_side_3/main-grpo-qwen25-math-1.5b-s8-baseline-b384.sh`
- replay
  - `/home/work/DDAI_revised/verl/RL_side_3/grpo-qwen25-math-1.5b-s8-m2-replay-rev2.sh`
  - `/home/work/DDAI_revised/verl/RL_side_3/grpo-qwen25-math-1.5b-s8-m2-replay-rev5.sh`
  - `/home/work/DDAI_revised/verl/RL_side_3/main-grpo-qwen25-math-1.5b-s8-m2-replay.sh`

### Qwen3-1.7B

- baseline
  - `/home/work/DDAI_revised/verl/RL_side_3/grpo-qwen3-1.7b-s8.sh`
- replay
  - `/home/work/DDAI_revised/verl/RL_side_3/grpo-qwen3-1.7b-s8-m2-replay-rev7.sh`
  - `/home/work/DDAI_revised/verl/RL_side_3/grpo-qwen3-1.7b-s8-m2-replay-rev8.sh`
  - `/home/work/DDAI_revised/verl/RL_side_3/grpo-qwen3-1.7b-s8-m2-replay-rev9.sh`

### Qwen3-4B-Instruct-2507

- baseline
  - `/home/work/DDAI_revised/verl/RL_side_3/main-grpo-qwen3-4b-instruct-2507-s8-baseline-b384.sh`
  - `/home/work/DDAI_revised/verl/RL_side_3/main-grpo-qwen3-4b-instruct-2507-s8-baseline-b256.sh`
- 모델 경로
  - `/home/work/DDAI_revised/verl/data/models/Qwen3-4B-Instruct-2507`

---

## 3. 모델/데이터 경로

- Qwen2.5-Math-1.5B
  - `/home/work/DDAI_revised/verl/data/models/Qwen2.5-Math-1.5B`
- Qwen3-4B-Instruct-2507
  - `/home/work/DDAI_revised/verl/data/models/Qwen3-4B-Instruct-2507`
- train
  - `/home/work/DDAI_revised/verl/data/deepscaler_preview/train.parquet`
- val/test
  - `/home/work/DDAI_revised/verl/data/minerva/test.parquet`
  - `/home/work/DDAI_revised/verl/data/math500/test.parquet`
  - `/home/work/DDAI_revised/verl/data/aime2024x4/test.parquet`
  - `/home/work/DDAI_revised/verl/data/aime2025x4/test.parquet`
  - `/home/work/DDAI_revised/verl/data/amc23/test.parquet`
  - `/home/work/DDAI_revised/verl/data/olympiadbench/test.parquet`

---

## 4. replay 관련 현재 코드 기준 핵심 사실

### replay는 actor update에만 들어간다

- replay는 critic update에 직접 들어가지 않는다.
- replay는 최종 `actor_batch`에만 붙는다.
- replay 해석의 핵심은 critic metric보다 **actor split metric**이다.

관련 코드:
- `/home/work/DDAI_revised/verl/verl/trainer/ppo/m2_replay_adapter.py`
- `/home/work/DDAI_revised/verl/verl/trainer/ppo/core_algos.py`

### replay/on-policy identity는 순서가 아니라 flag로 보존된다

- 최종 actor batch는 replay-first ordering이지만, 더 중요한 건 `m2_replay_source` tensor가 붙는다는 점이다.
- 따라서 최종 batch를 reorder해도 replay/on-policy 구분은 유지 가능하다.

### `age` 의미

- `age = current_step - last_training_step`
- 단순히 버퍼에 들어온 뒤 몇 step이 지났는지가 아니다.
- replay에 실제로 사용되면 `last_training_step`가 refresh된다.
- replay가 안 돌면 age가 누적된다.

### FIFO와 age는 모순되지 않는다

- 버퍼 eviction은 FIFO다.
- 하지만 age는 `last_training_step` 기준이라,
  - step당 ingress group 수가 적거나
  - replay refresh가 적으면
평균 age는 커질 수 있다.

---

## 5. 현재 지원되는 replay 설정과 주의점

### ingress filter mode

- key: `algorithm.m2_replay.selection.ingress_filter_mode`
- 현재 지원 값
  - `none`
  - `rlvr_halfband`
  - `rlvr_non_degenerate`

의미:
- `rlvr_halfband`: `1 <= success_count <= floor(n / 2)`
- `rlvr_non_degenerate`: `1 <= success_count <= n - 1`

### start mode

- key: `algorithm.m2_replay.schedule.start_mode`
- 현재 지원 값
  - `immediate`
  - `quarter`
  - `half`
  - `buffer_full`
  - `two_turnovers`

관련 코드:
- `/home/work/DDAI_revised/verl/verl/trainer/ppo/ray_trainer.py`

### ZVP mode

- key: `algorithm.m2_replay.selection.zvp_mode`
- 현재 코드 기준 정식 지원 값
  - `sign_only`
  - `adv_magnitude`

중요:
- 대화/문서/스크립트에 `plain` 표현이 남아 있을 수 있다.
- 현재 코드에서는 `plain`이 정식 mode가 아니며, 사실상 `sign_only` 쪽으로 fallback된다고 보는 게 안전하다.

### actor dynamic batch

- 현재 `m2_replay` 구현은 `actor.use_dynamic_bsz=true`를 지원하지 않는다.
- trainer에서 막고 있다.

관련 코드:
- `/home/work/DDAI_revised/verl/verl/trainer/ppo/ray_trainer.py`

---

## 6. replay 최적화 구현 상태

### 이미 구현된 것

#### 6.1 Package B 계열

- final `actor_batch` 기준 metric 추가
- selection 경로 일부 최적화
- `m2_prepare_onpolicy_groups` fast path
  - actor view를 먼저 만든 뒤 accepted group에 대해 직접 slice
- `m2_actor_batch_concat` one-shot concat 계열 최적화

관련 테스트:
- `/home/work/DDAI_revised/verl/tests/trainer/ppo/test_m2_replay_fastpaths_on_cpu.py`
- `/home/work/DDAI_revised/verl/tests/trainer/ppo/test_m2_replay_package_b_on_cpu.py`

#### 6.2 GPU-side M2 fast path

- 기본값
  - `selection.gpu_m2_fastpath=true`
  - `selection.compute_runtime_zvp=false`

의미:
- worker/GPU에서 grouped M2까지 계산하고 driver에는 scalar M2만 가져오도록 함
- `m2_select_m2_eval`와 CPU-side `new_log_probs` 재구성 비용을 줄이는 것이 목표

관련 코드:
- `/home/work/DDAI_revised/verl/verl/workers/fsdp_workers.py`
- `/home/work/DDAI_revised/verl/verl/trainer/ppo/ray_trainer.py`
- `/home/work/DDAI_revised/verl/verl/trainer/ppo/m2_replay.py`

관련 테스트:
- `/home/work/DDAI_revised/verl/tests/trainer/ppo/test_m2_replay_gpu_m2_fastpath_on_cpu.py`

#### 6.3 selection phase 전용 dynamic log-prob override

- 구현됨
- `_compute_old_log_prob` backbone은 유지
- replay selection fresh log-prob에만
  - `log_prob_use_dynamic_bsz`
  - `log_prob_max_token_len_per_gpu`
  override를 줄 수 있게 함

관련 테스트:
- `/home/work/DDAI_revised/verl/tests/trainer/ppo/test_m2_replay_selection_logprob_overrides_on_cpu.py`

중요:
- 이건 replay selection 전용이다.
- 일반 baseline old_log_prob 경로를 자동으로 바꾸는 게 아니다.

---

## 7. timing_s 해석 규칙

### 가장 중요한 원칙

`m2_*` timing은 서로 독립적인 막대가 아니라 **중첩 타이머**다.  
즉 더하면 안 된다.

대략 구조:

- `m2_build_actor_batch`
  - `m2_prepare_onpolicy_groups`
  - `m2_select_replay_total`
    - `m2_select_scan_loop`
      - `m2_select_logprob_eval`
      - `m2_select_m2_eval`
  - `m2_actor_batch_concat`
  - 기타 untimed bookkeeping / rebalance / metric 계산

### 각 타이머 의미

- `m2_prepare_onpolicy_groups`
  - 현재 on-policy batch를 replay ingress 가능한 `QueryGroup`들로 전처리하는 시간
- `m2_select_replay_total`
  - replay selection 전체 시간
- `m2_select_scan_loop`
  - candidate를 실제로 훑으면서 accept/reject 하는 핵심 루프 시간
- `m2_select_logprob_eval`
  - fresh `new_log_prob` 재평가 시간
  - GPU-side M2 fast path가 켜져 있으면 grouped M2 worker 호출 비용도 이 안으로 묶여 보일 수 있음
- `m2_select_m2_eval`
  - driver 쪽에서 `old/new/mask`를 stack하고 M2를 계산하는 시간
  - GPU M2 fast path가 정상적으로 켜지면 크게 줄어야 한다
- `m2_actor_batch_concat`
  - replay + on-policy로 최종 actor batch를 합치는 시간
- `m2_build_actor_batch`
  - 위 전체 + 그 밖의 부가 작업을 포함한 최상위 시간

### metric 해석 시 중요한 사실

- `response_length/mean`은 원래 on-policy batch 기준이다
- 최종 actor 입력 길이/토큰은 `actor_batch/*` metric을 봐야 한다

즉:
- 겉보기 `response_length/mean`은 baseline과 비슷한데
- `actor_update_policy_compute`만 더 큰 경우

이건 replay가 붙은 최종 actor batch가 더 길거나 더 불균형하다는 뜻이다.

---

## 8. 현재까지 확인된 주요 병목

### 1순위

- `m2_select_logprob_eval`

이유:
- exact eval 후보 수가 많을수록 거의 선형적으로 커진다
- 특히 Qwen3 mid-stage는 `scanned_groups=640, 864, 896` 같은 step이 존재했다

### 2순위

- `m2_select_m2_eval`

중요한 사실:
- 수식 자체는 단순하다
- 하지만 현재 구현에서는 과거에 CPU에서
  - `old_log_probs`
  - `new_log_probs`
  - `response_mask`
를 다시 stack하고 큰 tensor를 훑는 비용이 컸다
- GPU-side M2 fast path가 바로 이 병목을 줄이기 위해 들어갔다

### 3순위

- `m2_prepare_onpolicy_groups`

핵심 해석:
- 이건 GPU kernel 병목보다 **Python/DataProto slicing/materialization 병목**에 가깝다
- Qwen2.5에서 이 값이 큰 이유는, 길이보다 ingress group 수가 많아서다

### 4순위

- `m2_actor_batch_concat`

핵심 해석:
- 단순 concat이 아니라, 큰 `DataProto.concat`과 source flag 부착이 반복되면 비용이 커진다

---

## 9. Qwen2.5와 Qwen3에서 replay가 다르게 보인 이유

### Qwen2.5-Math-1.5B

- replay가 잘 먹히던 시기에는 selection acceptance가 높고, fresh hard-negative 쪽이 충분히 공급되었다
- healthy mid-stage에서는
  - `target=128`
  - `scanned≈192`
  - `accepted≈128`
  - `acceptance≈0.667`
같은 패턴이 자주 나왔다

### Qwen3-1.7B

- replay가 stale hard negative를 충분히 잡지 못했고
- later revision에서는 tau를 완화하면 replay는 들어오지만, easy-positive/self-imitation 성향이 강해져 baseline보다 열세를 보였다
- 특히 Qwen3는 길이, clip pressure, response verbosity가 더 커서 replay mixed batch penalty가 심했다

핵심 요약:
- Qwen2.5 replay는 더 자주 **fresh hard-negative suppressor**
- Qwen3 replay는 더 자주 **fresh easy-positive self-imitation**
으로 작동했다

---

## 10. 현재 최적화 아이디어 중 우선순위 높은 것

### A. selection-only dynamic log-prob batching

- 추천도 높음
- 이유:
  - 의미 변화가 거의 없음
  - 현재 병목인 `m2_select_logprob_eval`를 직접 건드림
  - 일반 old_log_prob 경로를 안 건드리고 replay selection fresh log-prob에만 적용 가능

### B. GPU-side M2 fast path 유지

- 이미 구현되어 있고 기본값도 켜져 있음
- runtime ZVP를 training default에서 꺼 둔 상태가 현재 가장 합리적이다

### C. `m2_prepare_onpolicy_groups`와 `m2_actor_batch_concat` 구조 최적화

- 이미 1차 정리됨
- 여기서 더 나아가려면 지연 materialization 같은 구조 변경이 필요할 수 있음

### D. budgeted progressive widening

- 아이디어 자체는 유효
- 하지만 “그냥 B1 보고, 부족하면 B2 보고…”만으로는 현재 sequential scan과 본질적으로 크게 다르지 않다
- 진짜 효율을 만들려면 **tail exact eval을 실제로 안 보거나 덜 보는 budget**이 들어가야 한다
- 즉 semantic change가 selection-only dynamic batching보다 크다

---

## 11. replay 개수가 128보다 적게 나오는 문제

중요한 사실:
- `replay_used=16, 20, 32` 같은 상황은 dynamic bsz 때문이 아니라 **원래 있던 문제**다

원인:
- acceptance가 낮아 실제 selection에서 적게 통과
- `legacy_bonus` + `floor_to_micro_multiple=true`에 의해 micro-group multiple로 floor
- `legacy_bonus`에서는 replay가 적으면 final total groups도 같이 줄어듦

즉 문제의 본질은:
- 시스템 크래시가 아니라
- replay fraction이 크게 흔들리고
- 실험 해석이 baseline-like 쪽으로 흐려진다는 점이다

---

## 12. Qwen3-4B-Instruct-2507 baseline 16k 관련 최신 운영 메모

### 현재 baseline 스크립트

- `/home/work/DDAI_revised/verl/RL_side_3/main-grpo-qwen3-4b-instruct-2507-s8-baseline-b256.sh`

현재 핵심 값:
- `train_batch_size=256`
- `max_prompt_length=1536`
- `max_response_length=16384`
- `rollout.n=8`
- `rollout.max_model_len=17920`
- `rollout.max_num_batched_tokens=17920`
- `rollout.log_prob_micro_batch_size_per_gpu=4`
- `ref.log_prob_micro_batch_size_per_gpu=4`

### 중요한 운영 판단

- 과거 OOM은 generation이 아니라 **`compute_log_prob` 경로**에서 났다
- 즉 16k 설정에서
  - `rollout.log_prob_micro_batch_size_per_gpu=32`
  - `ref.log_prob_micro_batch_size_per_gpu=32`
같은 static 설정은 위험하다
- `4`로 낮춘 현재 값은 그 직접 원인을 완화한 상태다

### 아직 남은 리스크

- `actor.ppo_micro_batch_size_per_gpu=8`은 여전히 공격적일 수 있다
- 즉 old/ref log-prob OOM은 줄였지만, 다음 병목은 actor PPO update 쪽일 수 있다

### `max_num_batched_tokens` 관련 판단

- `16384` response, `17920` model len 실험에서 `max_num_batched_tokens`를 아예 생략해 기본값 `8192`로 가는 건 좋지 않다
- 의도적으로 rollout 비중을 눌러 보고 싶더라도, **작은 명시값으로 고정하는 게 맞다**

---

## 13. 현재 실험 운영 팁

### nohup log 위치

- `/home/work/DDAI_revised/verl/RL_side_3/nohup_logs`

### data-log 위치

- `/home/work/DDAI_revised/verl/data-log/m2-replay-exp/<experiment_name>`

### early sanity check에서 먼저 볼 것

- `timing_s/old_log_prob`
- `timing_s/ref`
- `timing_s/update_actor`
- `timing_s/step`
- replay run이면
  - `m2_replay/selection/gpu_m2_fastpath_enabled`
  - `m2_replay/selection/runtime_zvp_enabled`
  - `m2_replay/selection/replay_used`

---

## 14. 테스트 관련 최신 사실

기존 테스트는 수정하지 않고, 새 테스트를 계속 추가하는 원칙으로 작업했다.

현재 중요한 추가 테스트 파일:
- `/home/work/DDAI_revised/verl/tests/trainer/ppo/test_m2_replay_fastpaths_on_cpu.py`
- `/home/work/DDAI_revised/verl/tests/trainer/ppo/test_m2_replay_package_b_on_cpu.py`
- `/home/work/DDAI_revised/verl/tests/trainer/ppo/test_m2_replay_gpu_m2_fastpath_on_cpu.py`
- `/home/work/DDAI_revised/verl/tests/trainer/ppo/test_m2_replay_selection_logprob_overrides_on_cpu.py`

이 테스트들은 대체로 다음을 검증한다.
- fast path와 legacy path의 결과 동치성
- replay selection override 전달
- GPU M2 fast path 동작
- actor batch build/concat equivalence

---

## 15. 다음 에이전트가 가장 먼저 기억해야 할 것

1. 현재 병목은 generation보다 replay selection 쪽인 경우가 많다.
2. `m2_select_logprob_eval`와 `m2_select_m2_eval`는 이름보다 훨씬 비싼 경로다.
3. selection-only dynamic log-prob batching은 현재 가장 안전한 고효율 최적화 후보 중 하나다.
4. `replay_used`가 작게 나오는 문제는 dynamic bsz 문제가 아니라 원래 selection/legacy_bonus 설계 문제다.
5. Qwen3-4B 16k baseline에서 지난 직접 OOM 원인은 `compute_log_prob`였다. rollout/ref log-prob micro batch를 보수적으로 가져가야 한다.
6. Agents 문서보다 실제 실행 스크립트와 `ray_trainer.py`, `m2_replay.py`, `fsdp_workers.py`가 최신 사실의 기준이다.

