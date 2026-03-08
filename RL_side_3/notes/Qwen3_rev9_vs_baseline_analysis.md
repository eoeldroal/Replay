# Qwen3 rev9 vs baseline 분석

## 1. 분석 목적

이 문서는 다음 질문에 답하기 위해 작성한다.

- 왜 `grpo-qwen3-1.7b-s8-m2-replay-rev9-tau0025-b1024-ndeg-plain-quarter`는 여러 차례의 수정 이후에도 `grpo-qwen3-1.7b-s8-baseline-fresh`를 넘지 못했는가?
- 이 실패를 `stale replay`, `late start`, `easy-positive bias` 같은 기존 가설로 충분히 설명할 수 있는가?
- 아니면 `rev9`는 이미 다른 실패 모드로 넘어간 것인가?

핵심은 단순히 "점수가 낮다"를 반복하는 것이 아니라, 아래를 분리해서 보는 것이다.

1. replay가 실제로 어떤 샘플을 선택하고 있는가
2. 그 replay가 on-policy 분포를 어떤 방향으로 밀고 있는가
3. 그 비용이 얼마이며, 그 비용을 성능으로 회수하는가
4. baseline과 비교했을 때 어디서부터, 어떤 방식으로 학습 곡선이 갈라지는가

---

## 2. 분석에 사용한 로그와 데이터

### 2.1 기준 실험

- baseline
  - `RL_side_3/grpo-qwen3-1.7b-s8.sh`
  - experiment: `grpo-qwen3-1.7b-s8-baseline-fresh`
- replay
  - `RL_side_3/grpo-qwen3-1.7b-s8-m2-replay-rev9.sh`
  - experiment: `grpo-qwen3-1.7b-s8-m2-replay-rev9-tau0025-b1024-ndeg-plain-quarter`

### 2.2 사용한 원시 로그

- baseline local log
  - `data-log/m2-replay-exp/grpo-qwen3-1.7b-s8-baseline-fresh/grpo-qwen3-1.7b-s8-baseline-fresh.log`
- baseline wandb raw output
  - `wandb/run-20260305_174239-2lvrspug/files/output.log`
- rev9 local log
  - `data-log/m2-replay-exp/grpo-qwen3-1.7b-s8-m2-replay-rev9-tau0025-b1024-ndeg-plain-quarter/grpo-qwen3-1.7b-s8-m2-replay-rev9-tau0025-b1024-ndeg-plain-quarter.log`
- rev9 wandb raw output
  - `wandb/run-20260307_043331-ugv5cc4i/files/output.log`

### 2.3 부가 분석에 사용한 dump

- baseline rollout dump
  - `data-log/m2-replay-exp/grpo-qwen3-1.7b-s8-baseline-fresh/rollout_jsonl/*.jsonl`
- rev9 rollout dump
  - `data-log/m2-replay-exp/grpo-qwen3-1.7b-s8-m2-replay-rev9-tau0025-b1024-ndeg-plain-quarter/rollout_jsonl/*.jsonl`
- baseline validation dump
  - `data-log/m2-replay-exp/grpo-qwen3-1.7b-s8-baseline-fresh/validation_jsonl/{0,50,100}.jsonl`
- rev9 validation dump
  - `data-log/m2-replay-exp/grpo-qwen3-1.7b-s8-m2-replay-rev9-tau0025-b1024-ndeg-plain-quarter/validation_jsonl/{0,50}.jsonl`

---

## 3. 먼저 확인한 사실: local log와 wandb raw output은 같은 실험을 기록한다

이 분석은 local log를 주로 사용한다. 다만 local log 해석이 틀리지 않았는지 확인하기 위해 wandb의 `files/output.log`와 step별 숫자를 대조했다.

확인 결과:

- baseline은 step `0..112` 구간에서 local과 wandb raw output이 사실상 동일하다.
- rev9도 핵심 metric은 local과 wandb raw output이 동일하다.
  - 예: step `5`, `24`, `32`, `68`, `80`에서
    - `critic/score/mean`
    - `m2_replay/selection/replay_used`
    - `m2_replay/pass1/scanned_groups`
    가 local과 wandb raw output에서 일치한다.

따라서 아래 결론은 "wandb UI를 잘못 읽었다"가 아니라, 원시 로그 레벨에서 재현되는 사실에 기반한다.

---

## 4. 기존 가설 중 무엇이 이미 기각되었는가

### 4.1 `rev9`는 더 이상 stale replay 실패가 아니다

기존 `rev7`의 핵심 실패는 다음이었다.

- `accepted_groups = 0`
- `rejected_by_tau = 1024`
- full buffer를 다 봐도 tau-safe replay가 없었다

그러나 `rev9`는 완전히 다르다.

- step `5`부터 `replay_used = 128`
- step `5`에서 이미 `accepted_groups = 128`
- 이후 거의 모든 구간에서 `replay_used = 128`
- 선택된 replay의 `age`는 거의 항상 `1`

즉 `rev9`는 stale 때문에 replay가 안 들어오는 실험이 아니다.

### 4.2 `rev9`는 늦은 start 실패도 아니다

`quarter` gating은 실제로 의도대로 작동했다.

- step `5`
  - `buffer_size_pre_select = 285`
  - `quarter_ready = 1`
  - `selection_ready = 1`
  - `replay_used = 128`

즉 `buffer_full`이 너무 늦어서 hard-negative-rich phase를 놓친다는 `rev8`의 문제는 `rev9`에서 해결되었다.

### 4.3 `rev9`는 easy-positive replay 실패도 아니다

`rev8`의 핵심은 `adv_magnitude`로 인해 replay polarity가 `easy-positive` 쪽으로 뒤집힌 것이었다.

반면 `rev9` 중기 `24..32`에서는:

- `used_zvp_neg_frac_mean = 0.803`
- `used_zvp_pos_frac_mean = 0.197`

즉 `rev9`는 replay 부호를 correct-sign 쪽으로 되돌리는 데에는 성공했다.

---

## 5. `rev9`의 진짜 문제 1: replay가 거의 완전히 `1/16`에 붕괴한다

`used_replay_scores(first_10)`를 step별로 직접 파싱하면 다음이 나온다.

### 5.1 step별 선택 replay의 success bucket

- step `5..10`
  - `n = 60`
  - `success_prob_mean = 0.0625`
  - 전부 `1/16`
- step `15..22`
  - `n = 80`
  - 전부 `1/16`
- step `24..32`
  - `n = 90`
  - 전부 `1/16`
- step `68..76`
  - `n = 90`
  - 전부 `1/16`

즉 `rev9`가 뽑는 replay는 실질적으로 다음과 같다.

- fresh
- tau-safe
- hard-negative
- 그리고 거의 항상 `success_count = 1`

이건 단순히 "hard-negative를 선호한다" 수준이 아니다. 더 강하다.

**`rev9`는 replay curriculum이 `1/16` 단일 계층으로 붕괴했다고 보는 것이 더 정확하다.**

### 5.2 왜 이게 중요한가

이 현상은 `rev8`의 반대편 극단이다.

- `rev8`: easy-positive self-imitation
- `rev9`: one-class hard-negative suppressor

즉 `rev9`는 polarity는 고쳤지만, intermediate band를 거의 사용하지 못한다.

여기서 intermediate band란 다음을 뜻한다.

- `2/16`, `3/16`, `4/16`, `5/16` 등
- 이미 완전히 실패한 샘플은 아니지만
- 여전히 correction signal이 강한 샘플

이 구간이 replay에서 사라지면,

- 틀린 trajectory 억제는 할 수 있어도
- 정답 trajectory를 안정적으로 강화하는 구조는 약해진다.

---

## 6. `rev9`의 진짜 문제 2: on-policy 분포를 baseline보다 건강하게 만들지 못한다

단순 scalar 평균 대신, `rollout_jsonl`을 prompt별 16개 샘플 group으로 다시 묶어서 `성공 개수` 분포를 봤다.

### 6.1 baseline group 분포

#### step `24`

- `all_wrong (0/16) = 107`
- `all_right (16/16) = 49`
- `nondeg (1..15) = 100`

#### step `50`

- `all_wrong = 87`
- `all_right = 51`
- `nondeg = 118`

#### step `76`

- `all_wrong = 82`
- `all_right = 58`
- `nondeg = 115`

### 6.2 rev9 group 분포

#### step `24`

- `all_wrong = 118`
- `all_right = 38`
- `nondeg = 100`

#### step `50`

- `all_wrong = 102`
- `all_right = 48`
- `nondeg = 106`

#### step `76`

- `all_wrong = 88`
- `all_right = 48`
- `nondeg = 119`

### 6.3 해석

이 비교는 아주 중요하다.

`rev9`는 hard-negative replay를 강하게 쓰고 있는데도:

- `0/16` group을 baseline보다 더 많이 남긴다
- `16/16` group을 baseline보다 덜 만든다

즉 replay가 학습을 돕는 방향으로 curriculum을 재구성하지 못했다.

더 직설적으로 말하면:

- `rev9`는 "틀린 답을 벌주는 방향"으로는 움직이지만
- 그 결과가 `현재 fresh rollout의 성공 mass 증가`로 이어지지 않는다

이건 replay가 "오류 수정기"가 아니라 "억제기"에 더 가까워졌음을 시사한다.

---

## 7. `rev9`의 진짜 문제 3: 길이와 종료 품질이 baseline보다 나빠진다

이전 분석에서 이미 다음이 보였다.

- `rev9`는 baseline보다 길다
- `clip_ratio`가 높다
- entropy가 더 높게 남는다

이번에는 `rollout_jsonl`과 `validation_jsonl`에서 실제 종료 품질을 직접 봤다.

### 7.1 on-policy rollout에서의 `\boxed{}`와 `</think>` 비율

#### baseline rollout

- step `24`
  - `boxed_ratio = 0.507`
  - `close_think_ratio = 0.473`
- step `50`
  - `boxed_ratio = 0.611`
  - `close_think_ratio = 0.572`
- step `76`
  - `boxed_ratio = 0.641`
  - `close_think_ratio = 0.599`

#### rev9 rollout

- step `24`
  - `boxed_ratio = 0.402`
  - `close_think_ratio = 0.350`
- step `50`
  - `boxed_ratio = 0.512`
  - `close_think_ratio = 0.408`
- step `76`
  - `boxed_ratio = 0.573`
  - `close_think_ratio = 0.437`

### 7.2 held-out validation에서도 같은 현상

#### baseline validation

- step `0`
  - `mean_acc = 0.24028`
  - `mean_chars = 8474.7`
  - `boxed_ratio = 0.320`
  - `close_think_ratio = 0.300`
- step `50`
  - `mean_acc = 0.39588`
  - `mean_chars = 6818.7`
  - `boxed_ratio = 0.612`
  - `close_think_ratio = 0.590`
- step `100`
  - `mean_acc = 0.41154`
  - `mean_chars = 6367.7`
  - `boxed_ratio = 0.685`
  - `close_think_ratio = 0.651`

#### rev9 validation

- step `0`
  - `mean_acc = 0.24072`
  - `mean_chars = 8488.2`
  - `boxed_ratio = 0.324`
  - `close_think_ratio = 0.301`
- step `50`
  - `mean_acc = 0.38530`
  - `mean_chars = 7509.7`
  - `boxed_ratio = 0.555`
  - `close_think_ratio = 0.490`

### 7.3 해석

`rev9`는 train rollout에서만 그런 것이 아니다.

held-out validation에서도:

- 더 길고
- `\boxed{}`를 덜 내고
- `</think>`로 덜 닫고
- accuracy가 낮다

즉 현재 replay는 formatting/closure quality까지 baseline보다 더 나쁘게 만든다.

이건 단순히 "hard-negative를 써서 잠깐 흔들린다"가 아니다.

**정답 경로의 끝맺음과 안정적인 finalization 자체가 baseline보다 늦다.**

---

## 8. 다만 이것을 "다양성 collapse"로 읽으면 안 된다

한 가지는 분명히 해야 한다.

prompt별 16개 샘플의 literal output uniqueness를 보면:

- baseline: 거의 항상 `16`
- rev9: 거의 항상 `16`

즉 `rev9`의 실패는 "모든 샘플이 똑같은 답을 반복한다"는 의미의 collapse가 아니다.

오히려 현재 상황은 반대에 가깝다.

- 다양한 긴 reasoning trajectory는 여전히 많이 남아 있다
- 그런데 그 trajectory들이 정답으로 닫히는 비율이 baseline보다 낮다

즉 문제는 표현 다양성 부족이 아니라,

- 너무 많은 긴 탐색
- closure 약화
- correctness concentration 부족

이다.

---

## 9. prompt-level 질적 비교: 같은 문제에서 baseline은 닫고, rev9는 길게 흔들린다

### 9.1 4-digit even digits divisible by 4 문제, step `24`

- baseline: `14/16` 성공
- rev9: `3/16` 성공

부가 지표:

- baseline `boxed = 14`, `close_think = 14`, 평균 `8347 chars`
- rev9 `boxed = 3`, `close_think = 1`, 평균 `8732 chars`

여기서 rev9는 더 길게 쓰는데도 정답 경로에 잘 닫히지 않는다.

### 9.2 balanced integer 문제, step `50`

- baseline: `12/16`
- rev9: `3/16`

즉 중간 난이도 counting/combinatorics류에서도 baseline 대비 성공 mass가 현저히 낮다.

### 9.3 Alice/Bob, number 36 문제, step `76`

- baseline: `9/16`
- rev9: `1/16`
- 둘 다 `\boxed{}`는 거의 항상 냄

이 예시는 중요하다.

여기서는 단순 truncation이나 boxed formatting만의 문제가 아니다.

- rev9도 boxed는 낸다
- 하지만 reasoning path가 baseline보다 훨씬 자주 틀린 결론으로 간다

즉 현재 문제는

- 일부 prompt에서 답을 못 닫는다
- 다른 일부 prompt에서는 답을 닫아도 reasoning path가 잘못 간다

라는 두 층으로 나타난다.

---

## 10. 정량 스칼라 비교: `rev9`는 초반부터 step 기준, wall-clock 기준 모두 불리하다

### 10.1 step-window 평균 비교

#### early `1..10`

- score
  - baseline `0.2307`
  - rev9 `0.2262`
- entropy
  - baseline `0.2223`
  - rev9 `0.2243`
- response length
  - baseline `2875.3`
  - rev9 `2886.1`
- clip ratio
  - baseline `0.8066`
  - rev9 `0.8133`
- `timing_s/step`
  - baseline `327.3s`
  - rev9 `399.3s`

#### activation `15..22`

- score
  - baseline `0.3425`
  - rev9 `0.2905`
- entropy
  - baseline `0.1825`
  - rev9 `0.2010`
- response length
  - baseline `2700.0`
  - rev9 `2810.8`
- clip ratio
  - baseline `0.6847`
  - rev9 `0.7559`
- `timing_s/step`
  - baseline `298.6s`
  - rev9 `478.3s`

#### mid `24..32`

- score
  - baseline `0.3848`
  - rev9 `0.3208`
- entropy
  - baseline `0.1577`
  - rev9 `0.1839`
- response length
  - baseline `2632.8`
  - rev9 `2790.8`
- clip ratio
  - baseline `0.6254`
  - rev9 `0.7351`
- `timing_s/step`
  - baseline `284.0s`
  - rev9 `469.7s`

#### late `68..76`

- score
  - baseline `0.4282`
  - rev9 `0.3934`
- entropy
  - baseline `0.0824`
  - rev9 `0.1195`
- response length
  - baseline `2489.9`
  - rev9 `2715.5`
- clip ratio
  - baseline `0.5425`
  - rev9 `0.6826`
- `timing_s/step`
  - baseline `263.5s`
  - rev9 `482.8s`

### 10.2 score threshold 도달 시점

#### baseline

- `score >= 0.30`: step `14`, `1.26h`
- `score >= 0.35`: step `17`, `1.51h`
- `score >= 0.40`: step `21`, `1.84h`
- `score >= 0.42`: step `39`, `3.26h`

#### rev9

- `score >= 0.30`: step `17`, `2.02h`
- `score >= 0.35`: step `21`, `2.55h`
- `score >= 0.40`: step `51`, `6.49h`
- `score >= 0.42`: step `55`, `7.00h`

### 10.3 해석

`rev9`는

- step 기준으로도 baseline보다 늦고
- wall-clock 기준으로는 훨씬 더 늦다

따라서 현재 상태를 "느리지만 더 좋아질 가능성이 있는 실험"으로 보기 어렵다.

현재까지는 **초반부터 계속 baseline보다 비효율적인 trajectory**를 밟는 실험에 가깝다.

---

## 11. 비용 분석: replay는 비싸고, 그 비용을 회수하지 못한다

### 11.1 selection/build 비용

중기 `24..32`:

- baseline `timing_s/m2_build_actor_batch ≈ 0.05s`
- rev9 `≈ 106.87s`

후기 `68..76`:

- baseline `≈ 0.05s`
- rev9 `≈ 131.83s`

### 11.2 actor update 비용

중기 `24..32`:

- baseline `update_actor ≈ 69.53s`
- rev9 `≈ 122.02s`

후기 `68..76`:

- baseline `≈ 66.40s`
- rev9 `≈ 121.94s`

### 11.3 scan cost와 acceptance 악화

- step `5`
  - `scanned_groups = 256`
  - `acceptance_rate = 0.5`
- activation `15..22`
  - 평균 `616`
  - 평균 `0.208`
- mid `24..32`
  - 평균 `568.9`
  - 평균 `0.2265`
- late `68..76`
  - 평균 `725.3`
  - 평균 `0.1795`

동시에 `accepted_m2_mean`은 다음처럼 올라간다.

- early `1..10`: `0.00117`
- activation `15..22`: `0.00199`
- mid `24..32`: `0.00202`
- late `68..76`: `0.00204`

### 11.4 해석

즉 시간이 갈수록

- tau-safe 후보를 찾기 더 어려워지고
- 더 많은 그룹을 스캔해야 하고
- 선택되는 replay는 tau 경계에 더 가까워지고
- selection/build/update 비용은 더 커진다

그런데 그 비용이 score improvement로 회수되지 않는다.

이건 단순한 implementation overhead가 아니다.

**현재 replay distribution이 training이 진행될수록 더 trust-region-unfriendly해진다**는 뜻이다.

---

## 12. actor-side 신호도 비효율적이다

현재 replay는 correct-sign이지만, PPO 관점에서는 여전히 비싸다.

### 12.1 전체 actor drift

중기 `24..32`:

- baseline `actor/ppo_kl = 1.07e-05`
- rev9 `1.01e-04`

후기 `68..76`:

- baseline `3.52e-05`
- rev9 `1.50e-04`

### 12.2 replay vs on-policy 내부 비교

중기 `24..32`:

- `actor/replay_ppo_kl = 9.22e-05`
- `actor/onpolicy_ppo_kl = 8.60e-06`

후기 `68..76`:

- `actor/replay_ppo_kl = 1.16e-04`
- `actor/onpolicy_ppo_kl = 3.39e-05`

clip도 비슷하다.

중기 `24..32`:

- `replay_pg_clipfrac = 0.00226`
- `onpolicy_pg_clipfrac = 0.00111`

### 12.3 해석

즉 replay는 여전히 on-policy보다 훨씬 더 큰 KL/clip 비용을 만든다.

문제는 이 비용이

- entropy collapse를 더 빠르게 만들지도 못하고
- response length를 더 짧게 만들지도 못하고
- score를 더 올리지도 못한다

는 점이다.

즉 현재 replay는 actor를 baseline보다 더 비싸게 흔들지만, 더 좋은 곳으로 데려가지는 못한다.

---

## 13. `rev8 -> rev9` 비교: 방향은 고쳤지만 더 좋아지지는 않았다

중기 `24..32`에서 `rev8`과 `rev9`를 비교하면:

- score
  - rev8 `0.3407`
  - rev9 `0.3208`
- entropy
  - rev8 `0.1834`
  - rev9 `0.1839`
- response length
  - rev8 `2734.2`
  - rev9 `2790.8`
- clip ratio
  - rev8 `0.6941`
  - rev9 `0.7351`
- `timing_s/step`
  - rev8 `435.0s`
  - rev9 `469.7s`
- `acceptance_rate`
  - rev8 `0.308`
  - rev9 `0.227`
- `used_zvp_neg_frac_mean`
  - rev8 `0.354`
  - rev9 `0.803`

해석은 명확하다.

- `rev9`는 easy-positive 문제는 확실히 해결했다.
- 그러나 score는 오히려 더 떨어졌고, selection은 더 비싸졌다.

즉 `plain + quarter`는 `rev8`의 잘못된 polarity를 고친 진단 런이었지만,
현재로서는 더 나은 replay curriculum을 만들지는 못했다.

---

## 14. 최종 결론

현재 `rev9`의 실패는 다음 세 문장으로 요약할 수 있다.

1. `rev9`는 더 이상 stale 실패도, late-start 실패도, easy-positive 실패도 아니다.
2. 대신 replay가 거의 완전히 `fresh 1/16 hard-negative`에 붕괴했고, intermediate band를 사실상 잃었다.
3. 그 결과 replay는 높은 selection/build/update 비용을 내면서도, on-policy 분포를 baseline보다 건강하게 만들지 못했다.

이를 조금 더 구조적으로 정리하면 아래와 같다.

### 14.1 무엇이 해결되었는가

- replay가 들어오지 않는 문제
- buffer_full로 너무 늦게 켜지는 문제
- easy-positive self-imitation 문제

### 14.2 무엇이 새롭게 남았는가

- replay class가 `1/16`에 과도하게 집중
- 정답 경로 강화보다 억제 신호가 우세
- formatting/closure quality 열세
- cost 증가 대비 성능 회수 실패

### 14.3 따라서 현재 가장 정확한 진단

`rev9`는 `right-sign replay`이지만 `useful replay curriculum`은 아니다.

혹은 더 직접적으로 말하면,

**현재 Qwen3에서 `rev9`는 "틀린 샘플을 잘 다시 본다"가 아니라, "너무 좁은 종류의 틀린 샘플을 너무 비싸게 다시 본다"에 가깝다.**

---

## 15. 이 분석으로부터 바로 따라오는 함의

현재 결과로부터는 아래 세 가지가 자연스럽다.

1. 다음 실험은 hard-negative를 더 강하게 밀기보다, `1/16`과 `2..k/16` intermediate band를 함께 쓰는 방향이어야 한다.
2. replay가 `정답 경로를 닫는 능력`을 해치고 있으므로, 단순 negative dominance만 보지 말고 `closure/boxing/finalization`과 연결된 positive structure를 함께 봐야 한다.
3. selection cost는 이미 충분히 본질적 병목이므로, 다음 설계는 curriculum과 함께 cost까지 동시에 줄이는 방향이어야 한다.

즉 다음 단계의 질문은 더 이상

- "replay를 더 넣을까?"
- "tau를 더 열까?"

가 아니다.

이제는 오히려

- "어떤 replay를 섞어야 현재 fresh boundary를 덜 망가뜨리는가?"
- "어떤 replay가 closure와 correctness concentration을 같이 올리는가?"

를 물어야 한다.

---

## 16. Qwen2.5-Math와의 비교: 왜 비슷한 replay 계열이 한쪽에서는 먹히고, 다른 쪽에서는 무너지는가

위 분석만 보면 `rev9`의 실패를 다음처럼 오해할 수 있다.

- hard-negative replay 자체가 잘못됐다
- `plain ZVP`가 Qwen3에서만 이상하게 작동했다
- replay를 일찍 켜는 방식이 본질적으로 해롭다

그러나 이 해석은 충분히 엄격하지 않다. 왜냐하면 실제로는 매우 비슷한 계열의 설정에서 `Qwen2.5-Math-1.5B`는 replay가 baseline보다 분명한 이득을 만들었기 때문이다.

이번 절에서는 아래 두 실험을 같이 놓고 `Qwen3 rev9`의 실패를 다시 읽는다.

- `grpo-qwen25-math-1.5b-s8-baseline`
- `grpo-qwen25-math-1.5b-s8-m2-replay-rev5-8gpu-fresh`

즉 질문은 다음과 같다.

> 왜 `Qwen2.5-Math`에서는 hard-negative replay가 실제로 성능을 올리는데, `Qwen3 rev9`에서는 비슷한 방향의 replay가 오히려 baseline을 못 이기는가?

핵심은 \"replay 부호\" 하나가 아니다. 더 정확히는

- rollout geometry
- trust-region compatibility
- replay class의 폭
- replay selection cost

가 함께 갈린다.

### 16.1 비교 대상

사용한 로그:

- Qwen2.5 baseline
  - `data-log/m2-replay-exp/grpo-qwen25-math-1.5b-s8-baseline/grpo-qwen25-math-1.5b-s8-baseline.log`
- Qwen2.5 replay
  - `data-log/m2-replay-exp/grpo-qwen25-math-1.5b-s8-m2-replay-rev5-8gpu-fresh/grpo-qwen25-math-1.5b-s8-m2-replay-rev5-8gpu-fresh.log`
- Qwen3 baseline
  - `data-log/m2-replay-exp/grpo-qwen3-1.7b-s8-baseline-fresh/grpo-qwen3-1.7b-s8-baseline-fresh.log`
- Qwen3 replay
  - `data-log/m2-replay-exp/grpo-qwen3-1.7b-s8-m2-replay-rev9-tau0025-b1024-ndeg-plain-quarter/grpo-qwen3-1.7b-s8-m2-replay-rev9-tau0025-b1024-ndeg-plain-quarter.log`

### 16.2 먼저, Qwen2.5 replay는 실제로 baseline을 이긴다

Qwen2.5 baseline과 replay `rev5-8gpu-fresh`를 같은 구간 평균으로 비교하면 효과 방향이 분명하다.

#### early `1..10`

- score
  - baseline `0.2436`
  - replay `0.2466`
  - `+0.0030`
- entropy
  - baseline `0.3703`
  - replay `0.3565`
  - `-0.0138`
- response length
  - baseline `1087.7`
  - replay `1066.8`
  - `-20.9`
- `timing_s/step`
  - baseline `155.3s`
  - replay `161.5s`
  - `+6.2s`

#### mid `24..32`

- score
  - baseline `0.3101`
  - replay `0.3220`
  - `+0.0119`
- entropy
  - baseline `0.2431`
  - replay `0.2365`
  - `-0.0067`
- response length
  - baseline `949.9`
  - replay `896.2`
  - `-53.7`
- `timing_s/step`
  - baseline `127.8s`
  - replay `136.4s`
  - `+8.6s`

#### late `68..76`

- score
  - baseline `0.3554`
  - replay `0.3712`
  - `+0.0158`
- entropy
  - baseline `0.1911`
  - replay `0.1724`
  - `-0.0187`
- response length
  - baseline `892.7`
  - replay `794.8`
  - `-97.9`
- `timing_s/step`
  - baseline `122.6s`
  - replay `133.1s`
  - `+10.5s`

즉 Qwen2.5 replay는

- 더 빠르게 entropy를 깎고
- 더 짧게 답하고
- score를 더 올리고
- step cost는 약간만 더 든다

는 패턴을 보인다.

score threshold 기준으로도 비슷하다.

- `score >= 0.30`
  - baseline: step `17`, `0.689h`
  - replay: step `14`, `0.616h`
- `score >= 0.35`
  - baseline: step `34`, `1.316h`
  - replay: step `30`, `1.249h`
- `score >= 0.40`
  - baseline: step `77`, `2.799h`
  - replay: step `66`, `2.594h`
- `score >= 0.42`
  - baseline: step `121`, `4.307h`
  - replay: step `77`, `2.999h`

즉 Qwen2.5에서는 replay가 단순히 \"점수가 조금 높다\"가 아니라,
실제로 학습 곡선 자체를 앞당긴다.

### 16.3 놀랍게도, Qwen2.5 replay도 hard-negative 중심이다

표면적으로 보면 `Qwen2.5 rev5`와 `Qwen3 rev9`는 둘 다 hard-negative replay다.

Qwen2.5 replay `used_replay_scores(first_10)` pooled 결과:

- early `2..10`
  - `success_prob_mean = 0.0653`
  - 주로 `1/16`, 일부 `2/16`
- mid `24..32`
  - `success_prob_mean = 0.0743`
  - `1/16` 다수, `2/16` 일부, `3/16` 소수
- late `68..76`
  - `success_prob_mean = 0.0667`
  - 다시 `1/16` 중심

Qwen3 rev9는:

- early `2..10`
  - `success_prob_mean = 0.0625`
  - 전부 `1/16`
- mid `24..32`
  - 전부 `1/16`
- late `68..76`
  - 전부 `1/16`

즉 차이는 단순히 \"한쪽은 hard-negative, 다른 쪽은 positive\"가 아니다.

더 정확히는:

- Qwen2.5 replay는 hard-negative 중심이지만 약간의 `2/16`, `3/16`이 살아 있다
- Qwen3 replay는 거의 완전히 `1/16` 단일 class로 붕괴한다

이 차이는 작아 보이지만, 실제로는 매우 중요하다.

Qwen2.5에서는 replay가

- 강한 억제 신호를 주면서도
- 약한 intermediate correction을 조금은 남겨 둔다

반면 Qwen3에서는 replay가

- 지나치게 좁은 실패 클래스에만 몰리면서
- 현재 on-policy boundary를 넓게 보지 못한다

### 16.4 그러나 진짜 차이는 replay 부호보다 rollout geometry다

가장 큰 차이는 사실 이쪽이다.

#### Qwen2.5

- response length
  - early `1087.7`
  - mid `949.9`
  - late `892.7`
- `clip_ratio = 0.0` 전 구간
- replay accepted `m2_mean`
  - early `0.000499`
  - mid `0.000513`
  - late `0.000533`
- replay acceptance rate
  - early `0.756`
  - mid `0.869`
  - late `0.800`
- replay scanned groups
  - early `139.2`
  - mid `147.6`
  - late `160.0`

#### Qwen3

- response length
  - early `2886.1`
  - mid `2790.8`
  - late `2715.5`
- `clip_ratio`
  - early `0.813`
  - mid `0.735`
  - late `0.683`
- replay accepted `m2_mean`
  - early `0.001169`
  - mid `0.002025`
  - late `0.002042`
- replay acceptance rate
  - early `0.280`
  - mid `0.227`
  - late `0.180`
- replay scanned groups
  - early `166.4`
  - mid `568.9`
  - late `725.3`

이 비교가 가장 중요하다.

같은 hard-negative replay라도

- Qwen2.5에서는 trajectory가 짧고 clip이 없어서
  - M2가 아주 낮고
  - tau-safe 후보가 많이 살고
  - 상위 hard-negative를 싸게 확보할 수 있다
- Qwen3에서는 trajectory가 길고 clip이 높아서
  - M2가 훨씬 크고
  - tau-safe 후보 풀이 매우 얇아지고
  - 같은 128개를 얻기 위해 훨씬 더 많이 스캔해야 한다

즉 두 모델의 차이는 replay sign보다
**replay가 trust region 안에 얼마나 싸게 남을 수 있는가** 에 더 가깝다.

### 16.5 비용 구조도 완전히 다르다

Qwen2.5 replay의 추가 비용:

- `timing_s/step` 증가
  - early `+6.2s`
  - mid `+8.6s`
  - late `+10.5s`

즉 replay를 넣어도 baseline보다 조금 느릴 뿐이다.

반면 Qwen3 replay의 추가 비용:

- early `+72.0s`
- mid `+185.7s`
- late `+219.3s`

그리고 Qwen3에서는 그 비용의 대부분이

- `m2_build_actor_batch`
- 더 큰 `update_actor`

에서 발생한다.

예를 들어 mid `24..32`에서:

- Qwen2.5 replay 추가 cost는 한 step당 대략 `8.6s`
- Qwen3 replay 추가 cost는 한 step당 `185.7s`

즉 둘 다 replay target은 `128`이지만, 실제 계산 환경은 전혀 다르다.

### 16.6 start timing도 질적으로 다르다

Qwen2.5 replay `rev5`는 사실상 초반부터 바로 붙는다.

- step `1`
  - `new_groups = 136`
  - replay는 아직 `0`
- step `2`
  - `replay_used = 128`
  - `scanned_groups = 128`
  - `acceptance_rate = 1.0`

즉 step `2`부터 full replay가 붙는다.

Qwen3 rev9도 일찍 붙기는 한다.

- step `5`
  - `replay_used = 128`
  - `scanned_groups = 256`
  - `acceptance_rate = 0.5`

하지만 차이는 분명하다.

- Qwen2.5는 small fresh pool에서도 tau-safe hard-negative가 넘친다
- Qwen3는 quarter gating을 써도 이미 step `5`부터 hard-negative replay를 채우는 비용이 더 크다

즉 `quarter`가 잘못된 것이 아니라,
**Qwen3의 fresh candidate pool 자체가 Qwen2.5만큼 replay-friendly하지 않다** 는 뜻이다.

### 16.7 그래서 무엇이 공통이고, 무엇이 진짜 차이인가

공통점:

- 둘 다 replay는 hard-negative 쪽으로 기운다
- 둘 다 replay는 fresh age `~1`을 주로 쓴다
- 둘 다 target `128` replay를 실제로 채운다

진짜 차이:

1. Qwen2.5는 replay가 `1/16` 중심이지만 완전히 단일 bucket로 붕괴하지는 않는다
2. Qwen2.5는 response가 짧고 clip이 없어 tau-safe hard-negative가 많이 산다
3. Qwen2.5는 accepted M2가 매우 낮아 replay가 actor를 trust-region 밖으로 세게 밀지 않는다
4. 따라서 replay cost가 작고, entropy/length/score가 baseline보다 모두 좋은 방향으로 움직인다
5. Qwen3는 같은 sign의 replay라도 long / clipped geometry 때문에 replay selection이 비싸고 trust-region 경계에 붙으며, 결과적으로 baseline의 fresh curriculum을 희석한다

### 16.8 이 비교를 반영한 더 엄격한 결론

기존 결론을 더 냉정하게 고치면 이렇다.

`rev9`가 실패한 이유는 단순히

- hard-negative replay라서도 아니고
- plain ZVP라서도 아니고
- replay를 early-start해서도 아니다

오히려 다음이 더 정확하다.

1. `Qwen2.5-Math`에서는 hard-negative replay가 짧고 tau-safe한 상태로 많이 남아, replay가 실제로 효율적인 correction signal이 된다.
2. `Qwen3`에서는 비슷한 hard-negative replay가 long / clipped geometry 때문에 tau 경계에 붙고 selection cost가 급증한다.
3. 그 결과 Qwen3의 replay는 sign은 맞아도, 실질적으로는 너무 좁고 너무 비싼 suppressive replay가 된다.

즉 `Qwen3 rev9`의 실패는
\"hard-negative replay의 일반적 실패\"
가 아니라,
**Qwen3 rollout geometry 하에서 같은 replay family가 더 이상 efficient correction signal이 아니게 되는 failure** 로 보는 것이 가장 정확하다.

## 17. `tau` 완화 이력까지 합친 응답 양상 비교: replay를 살리자 무엇이 무너졌는가

앞 절까지는 `baseline`과 `rev9`를 중심으로 비교했다.
하지만 이 해석을 더 엄격하게 하려면, `Qwen3`에서 실제로 어떤 순서로 현상이 바뀌었는지를 같이 봐야 한다.

기존 `Revising_9.md`, `Revising_10.md`에 남아 있는 흐름을 다시 정리하면 다음과 같다.

1. `rev7` (`tau=0.001`, halfband 계열)
   - hard-negative replay를 원했지만, late stage에는 `rejected_by_tau = 1024`, `replay_used = 0`
   - 즉 replay는 사실상 죽어 있었다
2. `rev8 tau=0.0015`
   - tau를 1차 완화했지만 replay는 부분적으로만 활성화되었다
   - `adv_magnitude + non_degenerate + buffer_full` 조합이 들어가면서 selection polarity가 easy-positive 쪽으로 기울기 시작했다
3. `rev8 tau=0.0025`
   - replay는 충분히 들어오기 시작했지만, 이것이 곧바로 성능 향상으로 이어지지 않았다
   - 실제로는 `fresh easy-positive self-imitation` 쪽으로 더 강하게 이동했다
4. `rev9 tau=0.0025`
   - `plain + quarter`로 polarity를 바로잡았지만, 이번에는 `fresh 1/16 hard-negative only`로 과교정되었다

즉 `tau` 완화의 효과는 단순하지 않았다.

- `tau`를 여는 것 자체는 replay activation을 복구했다
- 하지만 그 뒤에 **어떤 replay가 활성화되느냐**가 더 중요했고
- 그 결과는 `rev8`과 `rev9`에서 전혀 다르게 나타났다

### 17.1 응답 양상 관점에서 보면, `rev7`은 아직 baseline과 같은 세계에 있었다

`rollout_jsonl` 기준으로 `step 24 / 50 / 76 / 100`의 on-policy 응답 패턴을 보면:

#### baseline

- step `24`: `acc = 0.3960`, `chars = 7731.5`, `boxed = 0.507`, `close_think = 0.473`
- step `50`: `acc = 0.4243`, `chars = 7329.5`, `boxed = 0.611`, `close_think = 0.572`
- step `76`: `acc = 0.4514`, `chars = 7217.0`, `boxed = 0.642`, `close_think = 0.599`
- step `100`: `acc = 0.4851`, `chars = 6778.1`, `boxed = 0.724`, `close_think = 0.680`

#### rev7 (`tau=0.001`, replay 사실상 비활성)

- step `24`: `acc = 0.4023`, `chars = 7725.6`, `boxed = 0.515`, `close_think = 0.477`
- step `50`: `acc = 0.4336`, `chars = 7237.7`, `boxed = 0.626`, `close_think = 0.585`
- step `76`: `acc = 0.4485`, `chars = 7098.5`, `boxed = 0.667`, `close_think = 0.629`
- step `100`: `acc = 0.4680`, `chars = 6684.5`, `boxed = 0.736`, `close_think = 0.695`

여기서 중요한 점은, `rev7`은 replay 측면에서는 실패했지만 **응답 양상 자체는 baseline과 거의 같은 세계**에 있었다는 점이다.

- 길이는 baseline과 거의 같거나 약간 짧다
- `\\boxed{}`와 `</think>` 비율도 baseline과 비슷하거나 조금 높다
- held-out validation도 step `50` 기준 `rev7 = 0.4078`, `baseline = 0.3959`로 큰 열세가 아니다

즉 `rev7`의 핵심 병목은
**replay가 없었다는 것**
이지,
**정책 자체가 baseline보다 더 길고 더 미완성된 방향으로 망가졌다는 것**
은 아니었다.

이건 중요하다.
나중에 `rev8/9`에서 보이는 응답 품질 악화는 단순히 "Qwen3는 원래 어렵다"가 아니라, replay가 실제로 학습에 들어온 뒤에 생긴 현상으로 읽어야 하기 때문이다.

### 17.2 `tau`를 열고 replay를 살리자, 먼저 easy-positive self-imitation이 나타났다

`rev8 tau=0.0015`와 `rev8 tau=0.0025`의 `step 24` rollout을 보면:

#### rev8 (`tau=0.0015`)

- `acc = 0.3782`
- `chars = 7830.0`
- `boxed = 0.482`
- `close_think = 0.445`

#### rev8 (`tau=0.0025`)

- `acc = 0.3518`
- `chars = 8048.3`
- `boxed = 0.445`
- `close_think = 0.405`

baseline step `24`와 비교하면:

- baseline: `acc = 0.3960`, `chars = 7731.5`, `boxed = 0.507`, `close_think = 0.473`

즉 `tau`를 더 열수록:

1. accuracy가 내려가고
2. 출력이 길어지며
3. `\\boxed{}` 비율이 떨어지고
4. `</think>`로 reasoning을 닫는 비율이 떨어진다

이건 `Revising_10.md`에서 정리했던 `fresh easy-positive self-imitation` 해석과 맞는다.

왜냐하면 `adv_magnitude + tau 완화` 조합은 replay를 살리는 동시에,
이미 길게 풀어서 맞힌 trajectory family도 많이 통과시켰기 때문이다.

그 결과 policy는

- 더 빨리 답을 닫는 쪽으로 정리되지 않고
- 긴 reasoning family를 더 오래 유지하며
- formatting closure도 baseline보다 약해졌다

즉 `tau` 완화 자체가 나쁜 것이 아니라,
**그렇게 열린 통로를 통해 어떤 종류의 replay가 들어왔는가**가 핵심이었다.

### 17.3 같은 `tau=0.0025`에서도 `rev8`과 `rev9`는 전혀 다른 실패를 보인다

이제 `rev8 tau=0.0025`와 `rev9 tau=0.0025`를 같은 `step 24` 기준으로 비교하면:

#### rev8 (`tau=0.0025`, advmag, buffer_full)

- `acc = 0.3518`
- `chars = 8048.3`
- `boxed = 0.445`
- `close_think = 0.405`

#### rev9 (`tau=0.0025`, plain, quarter)

- `acc = 0.3232`
- `chars = 8237.4`
- `boxed = 0.402`
- `close_think = 0.350`

즉 `rev9`는

- replay polarity는 correct-sign으로 고쳤는데
- 응답 품질은 `rev8`보다도 더 나빠졌다

이게 뜻하는 바는 분명하다.

`tau=0.0025`에서 이미 replay는 충분히 들어온다.
문제는 이제 "들어오느냐/안 들어오느냐"가 아니라,

- `rev8`: easy-positive replay가 policy를 길고 느슨하게 만든다
- `rev9`: one-class hard-negative replay가 policy를 짧고 sharp하게 만들지도 못하면서 종료 품질까지 악화시킨다

즉 두 경우 모두
**replay activation 자체는 복구되었지만, useful curriculum은 형성되지 않았다.**

### 17.4 응답 양상 기준으로 보면 `rev9`는 `rev7`보다 오히려 더 나쁜 세계에 있다

이 비교가 특히 중요하다.

`rev7`과 `rev9`를 나란히 보면:

#### step `24`

- `rev7`: `acc = 0.4023`, `chars = 7725.6`, `boxed = 0.515`, `close_think = 0.477`
- `rev9`: `acc = 0.3232`, `chars = 8237.4`, `boxed = 0.402`, `close_think = 0.350`

#### step `50`

- `rev7` validation: `acc = 0.4078`, `chars = 6691.9`, `boxed = 0.636`, `close_think = 0.611`
- `rev9` validation: `acc = 0.3853`, `chars = 7509.7`, `boxed = 0.556`, `close_think = 0.490`

#### step `100` vs late rev9

- `rev7` rollout step `100`: `acc = 0.4680`, `chars = 6684.5`, `boxed = 0.736`, `close_think = 0.695`
- `rev9` rollout step `76`: `acc = 0.4177`, `chars = 8009.2`, `boxed = 0.573`, `close_think = 0.437`

즉 replay가 거의 죽어 있던 `rev7`이,
replay를 적극적으로 쓰는 `rev9`보다

- 더 정확하고
- 더 짧고
- 더 잘 닫히는

응답을 만들고 있다.

이건 불편하지만 중요하다.

즉 `Qwen3`에서는 현재까지의 replay 계열 수정이
baseline을 넘지 못하는 수준을 넘어서,
**아예 "replay를 적극적으로 살린 뒤의 정책"이 "replay가 사실상 죽은 정책"보다 응답 품질이 더 나빠질 수 있다**는 뜻이다.

### 17.5 따라서 `tau`의 역할은 이렇게 다시 읽어야 한다

기존 `Revising_9.md`에서는 `tau=0.001`이 너무 보수적이라 `replay_used ≈ 0`을 만든다고 정리했다.
이건 맞다.

하지만 이번 비교까지 합치면, `tau`에 대한 더 정확한 해석은 다음이다.

1. `tau`가 너무 작으면 (`rev7`)
   - replay activation 자체가 거의 죽는다
   - 문제는 stale/tau deadlock이다

2. `tau`를 열면 (`rev8`, `rev9`)
   - replay activation은 복구된다
   - 하지만 그 뒤에는 **selection rule과 model geometry가 replay의 의미를 결정한다**

3. Qwen3에서는 `tau` 완화만으로는 충분하지 않다
   - `rev8`: 열린 통로로 easy-positive가 들어와 self-imitation loop가 된다
   - `rev9`: 같은 `tau`에서 plain으로 돌리면 one-class hard suppressor가 된다

즉 `tau`는 필요한 조건이지만 충분한 조건이 아니다.

더 냉정하게 말하면:

- `tau=0.001`은 너무 작아서 replay가 없다
- `tau=0.0025`는 replay는 살리지만, 그 replay가 useful하다는 보장은 전혀 없다

그리고 현재 `Qwen3` 로그는,
후자 쪽 문제가 더 크다는 것을 보여 준다.

### 17.6 응답 양상까지 합쳐서 보면, 현재 Qwen3 failure는 세 단계로 분해된다

이제 `rev7 -> rev8 -> rev9`를 응답 패턴까지 포함해 다시 쓰면:

1. `rev7`
   - replay가 거의 없다
   - 하지만 응답 양상은 baseline과 비슷하다
   - 즉 실패는 주로 activation failure

2. `rev8`
   - replay가 살아난다
   - 대신 응답이 더 길어지고, closure가 약해지고, easy-positive replay가 길게 맞힌 trajectory family를 강화한다
   - 즉 실패는 polarity failure

3. `rev9`
   - replay polarity는 바로잡힌다
   - 그런데 응답은 더 길고, `boxed`/`</think>`는 더 약하고, `1/16` hard-negative 한 계층에만 몰린다
   - 즉 실패는 curriculum collapse + cost explosion

이 세 단계는 각각 다른 문제다.

그래서 현재 분석을 한 문장으로 줄이면:

> Qwen3에서는 `tau`를 여는 것 자체가 문제가 아니라, `tau`를 연 뒤 replay가 실제로 어떤 응답 family를 강화/억제하느냐가 핵심이며, 현재까지의 경로는 `no replay -> wrong replay -> too-narrow replay` 순서로 이동했다.

### 17.7 이 보강이 주는 실제 함의

이 보강 이후에는 다음 두 문장을 더 강하게 말할 수 있다.

1. `Qwen3`에서의 실패는 "replay가 안 돼서"가 아니라, **replay가 policy의 응답 종료 품질을 개선하는 curriculum으로 작동하지 못해서**다.
2. `Qwen2.5-Math`에서의 성공은 단순히 score gain만이 아니라, replay가 실제로 응답을 더 짧고 더 잘 닫히는 방향으로 정리하기 때문이다.

즉 앞으로의 개선 방향도

- 더 많은 hard-negative
- 더 많은 replay
- 더 큰 tau

가 아니라,

- intermediate band를 포함한 replay curriculum
- `\\boxed{}` / `</think>` closure를 무너뜨리지 않는 replay
- long clipped geometry에서 selection cost와 trust-region cost를 같이 제어하는 방식

으로 가야 한다.

## 18. 실제 rollout 텍스트 비교: Qwen3와 Qwen2.5는 무엇을 다르게 "쓴다"는 것인가

앞 절까지는 score, length, clip ratio, replay composition을 중심으로 비교했다.
하지만 여기까지 오면 한 가지 질문이 남는다.

> 실제 raw generation을 보면, Qwen3와 Qwen2.5는 무엇이 그렇게 다른가?

이 절은 그 질문에 답하기 위한 질적 보강이다.

다만 먼저 제한을 분명히 둔다.

### 18.1 제한: 성공한 Qwen2.5 rev5 run에는 raw rollout dump가 남아 있지 않다

이번 저장소에서 직접 확인 가능한 raw rollout dump는:

- `Qwen3 baseline-fresh`
- `Qwen3 rev7`
- `Qwen3 rev8`
- `Qwen3 rev9`
- 새로 띄운 `main-grpo-qwen25-math-1.5b-s8-m2-replay-tau0025-b1024-ndeg-plain-quarter`

이다.

반면, 실제로 "잘 됐다"는 결론에 썼던 과거 `Qwen2.5 baseline`, `Qwen2.5 rev5-8gpu-fresh`에는
현재 raw rollout JSONL이 남아 있지 않다.

따라서 아래 질적 비교는 정확히는:

1. `Qwen3` 계열의 raw rollout dump
2. 현재 남아 있는 `Qwen2.5-Math` family의 raw rollout dump
3. 과거 성공한 `Qwen2.5 rev5`의 scalar trajectory

를 합쳐 해석한 것이다.

즉 이 절은 "성공한 rev5와 같은 step의 exact text 비교"는 아니다.
하지만 모델 family 차이와 rollout style 차이를 읽는 데에는 충분한 근거를 준다.

### 18.2 aggregate raw rollout만 봐도 모델 family의 글쓰기 방식이 다르다

`step 24` 기준 raw rollout 통계를 비교하면:

#### Qwen3 baseline

- `acc = 0.3960`
- `chars = 7731.5`
- `boxed = 0.507`
- `close_think = 0.473`
- group histogram: `g0 = 12`, `gND = 235`, `g16 = 9`

#### Qwen3 rev9

- `acc = 0.3232`
- `chars = 8237.4`
- `boxed = 0.402`
- `close_think = 0.350`
- group histogram: `g0 = 24`, `gND = 225`, `g16 = 7`

#### Qwen2.5-Math family (현재 `main-qwen25` run, step 24)

- `acc = 0.3428`
- `chars = 2523.9`
- `boxed = 0.977`
- `close_think = 0.000`
- group histogram: `g0 = 13`, `gND = 243`, `g16 = 0`

이 차이는 매우 중요하다.

Qwen2.5는 현재 raw rollout 기준으로도:

- 응답이 훨씬 짧고
- 거의 항상 `\\boxed{}`를 내고
- `<think> ... </think>` 구조를 길게 끌지 않는다

반면 Qwen3는 baseline 단계에서도:

- 이미 매우 긴 reasoning trace를 깔고
- `\\boxed{}`와 `</think>`는 절반 수준만 안정적으로 낸다

즉 replay를 넣기 전부터 모델 family의 rollout geometry가 다르다.

### 18.3 아주 쉬운 산술 문제에서도 두 모델의 반응 방식이 다르다

raw rollout에서 쉽게 읽히는 예로,
Qwen3 baseline step `24`에는 다음 prompt가 있다.

- `If x + sqrt(81) = 25, what is the value of x?`

이 prompt에서 Qwen3 baseline의 성공 샘플은:

- 길이 `1404 chars`
- 또는 `2241 chars`
- 둘 다 `<think>`로 시작해
  - `sqrt(81)=9`를 다시 상기하고
  - 양변에서 `9`를 빼고
  - 계산을 다시 verbalize한 뒤
  - 마지막에 `\\boxed{16}`으로 닫는다

즉 맞는 샘플조차 "짧게 답한다"가 아니라
**긴 chain-of-thought를 끝까지 수행한 뒤 정답을 낸다.**

같은 step `24`에서 Qwen3 rev9의 성공 샘플은 더 길다.

- 한 샘플은 `2192 chars`
- 또 다른 샘플은 `2998 chars`

특히 rev9 성공 샘플은 단순한 산술 문제에서도

- "square root has positive/negative interpretation?" 같은 부수 설명을 더 붙이고
- principal root를 장황하게 확인하며
- baseline보다 더 많은 incidental reasoning token을 쓴다

즉 rev9는 성공 샘플조차
**정답 경로를 더 압축해서 배우지 못하고, 오히려 더 장황한 성공 궤적을 만든다.**

반면 Qwen2.5-Math family의 현재 raw rollout 같은 유형의 쉬운 문제에서는:

- 성공 샘플 길이가 `118 chars`, `372 chars`
- 매우 빨리 `\\boxed{-9}`로 닫는다
- 일부 샘플은 짧은 Python 코드 블록을 붙이지만, 그래도 전체 길이는 Qwen3의 성공 샘플보다 훨씬 짧다

즉 같은 "정답을 맞히는" 사례에서도

- Qwen3: 긴 `<think>` trajectory를 사용한 뒤 닫는다
- Qwen2.5: 빠르게 answer-oriented structure로 닫는다

이 차이는 replay 관점에서 매우 크다.
같은 hard-negative replay라도, Qwen3에서는 replay 한 번이 훨씬 더 많은 incidental token을 다시 학습하게 만들기 때문이다.

### 18.4 실패 샘플도 질적으로 다르다

Qwen3 baseline / rev9의 실패 샘플을 보면:

#### Qwen3 baseline 실패 예시

- 구슬 문제 실패 샘플 길이: `2920 chars`, `3519 chars`
- 둘 다 `\\boxed{}`도 내고 `</think>`도 닫는다
- 그런데 reasoning이 틀려서 오답이다

즉 Qwen3에서는
**형식은 지켰고 길게도 썼지만, reasoning path 자체가 틀린** 실패가 많다.

#### Qwen3 rev9 실패 예시

- 같은 구슬 문제 실패 샘플 길이: `3571 chars`
- 다른 조합 문제 실패 샘플 길이: `5158 chars`
- 역시 `\\boxed{}`와 `</think>`를 모두 내는 샘플도 많다

즉 rev9의 실패는

- boxed를 못 내서도 아니고
- close_think가 전혀 없어서도 아니고
- 단순 truncation 때문만도 아니다

오히려
**긴 reasoning을 수행하고도, 잘못된 경로를 더 오래 끌고 가는 실패**
가 많다.

반면 Qwen2.5-Math family의 현재 raw rollout 실패 샘플은 대체로 더 짧다.

- 한 실패 샘플은 `503 chars`
- 다른 실패 샘플은 `853 chars`

실패 양상도 다르다.

- vector sum 문제에서는 코드 블록을 만들다가 final print / formatting이 어색하게 끝난다
- counting 문제에서도 긴 combinatorial search보다 비교적 직접적인 procedural reasoning을 시도한다

즉 Qwen2.5 쪽 실패는
**짧고 직접적인 절차의 local mistake**
에 더 가깝고,
Qwen3 쪽 실패는
**아주 긴 trajectory 전체가 잘못된 방향으로 흘러가는 mistake**
에 더 가깝다.

이 차이가 바로 replay에서 중요한 차이다.

### 18.5 왜 이런 질적 차이가 replay의 효율 차이로 이어지는가

이제 앞 절의 정량 분석과 결합하면 인과가 더 명확해진다.

Qwen3의 raw rollout은:

- 맞아도 길고
- 틀려도 길고
- `<think>` 구조에 많은 incidental reasoning token이 들어 있다

따라서 hard-negative replay를 다시 넣으면,
우리는 단순히 "틀린 정답 토큰"만 다시 보는 것이 아니라,
**그 긴 잘못된 reasoning trajectory 전체를 다시 off-policy로 재평가하고 학습**하게 된다.

이건 곧:

- 높은 `m2`
- 낮은 acceptance rate
- 큰 `scanned_groups`
- 큰 `replay_ppo_kl`
- 큰 `replay_pg_clipfrac`

로 연결된다.

반면 Qwen2.5-Math family는 raw rollout 기준으로도:

- 훨씬 짧고
- answer-oriented closure가 빠르며
- 실패도 상대적으로 local procedural mistake에 가깝다

그래서 hard-negative replay 한 번이
Qwen3에서처럼 거대한 trajectory 전체를 다시 흔드는 것이 아니라,
**비교적 짧은 correction signal**로 작동할 가능성이 높다.

이 해석은 과거 성공 run인 `Qwen2.5 rev5` scalar와도 맞는다.

- early `response_length ~ 1088`
- mid `~ 896`
- late `~ 795`
- `clip_ratio = 0`
- acceptance rate `~0.76 ~ 0.87`

즉 Qwen2.5에서는 실제로 replay가
"짧고 trust-region-friendly한 correction"
으로 남아 있었다.

### 18.6 그래서 raw rollout 관점에서 보면, Qwen3의 진짜 문제는 "hard-negative"가 아니다

이 절의 질적 비교를 반영하면,
Qwen3 failure를 더 정확히 이렇게 쓸 수 있다.

- `rev9`가 실패한 이유는 hard-negative를 써서가 아니다
- 문제는 **Qwen3의 hard-negative trajectory 자체가 너무 길고, too much incidental reasoning을 포함한다는 점**이다

즉 Qwen3에서 replay가 비싸고 비효율적인 이유는
단순히 selection score나 tau gate 때문만이 아니라,
**모델이 원래 만들어 내는 rollout text의 구조 자체**
에도 있다.

한 줄로 줄이면:

> Qwen2.5는 replay가 다시 보기 좋은 짧은 오류를 만든다. Qwen3는 replay가 다시 보기 너무 비싼 긴 오류를 만든다.

이 차이가 결국

- `Qwen2.5`: hard-negative replay가 correction signal이 됨
- `Qwen3`: hard-negative replay가 high-cost suppressive signal이 됨

으로 이어진다.
