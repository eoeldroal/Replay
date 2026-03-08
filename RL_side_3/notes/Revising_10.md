# Revising_10: Qwen3 rev8 열세의 실제 원인과 rev9 설계 근거

## 1) 목적

이 문서는 앞선 `Revising_9`의 분석을 한 단계 더 좁혀서, 아래 질문에 답하기 위해 작성한다.

- 왜 `Qwen2.5-Math rev5`에서는 M2+ZVP replay가 실제 이득을 만들었는가
- 왜 `Qwen3 rev8`에서는 replay가 충분히 들어오는데도 baseline보다 열세를 보이는가
- 그 차이를 바탕으로, 다음 실험(`rev9`)은 어떤 방향으로 가야 하는가

중요한 점은, 이번 문서는 "가능해 보이는 설명"을 늘어놓는 대신, **실제 로그가 직접 지지하는 원인만 남기는 것**을 목표로 한다.

---

## 2) 먼저 정정할 점: rev8의 주원인은 더 이상 stale이 아니다

`rev7` 단계에서는 stale + tau mismatch가 핵심 병목이었다.

하지만 `rev8`(`tau=0.0025`, `buffer_full`, `rlvr_non_degenerate`, `adv_magnitude`)에 들어오면 상태가 바뀐다.

`Qwen3 rev8` step `24..32` 평균:

- `m2_replay/selection/replay_used = 128`
- `m2_replay/pass1/acceptance_rate ~= 0.308`
- `m2_replay/selection/used_zvp_pos_frac_mean ~= 0.646`
- `m2_replay/selection/used_zvp_neg_frac_mean ~= 0.354`

그리고 `used_replay_scores(first_10)`를 step `24..32`에서 모아 보면:

- pooled sample 수: `90`
- 평균 `success_prob ~= 0.8389`
- 평균 `age = 1.0`
- 평균 `m2 ~= 0.00192`
- `success_count` 분포:
  - `15/16`: `49`
  - `14/16`: `12`
  - `13/16`: `9`
  - `12/16`: `5`
  - 나머지: 소수

즉 `rev8`은

- replay가 안 들어오는 상태가 아니고
- stale sample 때문에 못 고르는 상태도 아니며
- 오히려 **매우 fresh한 sample이 잘 들어오는데, 그 sample의 polarity가 잘못되어 있는 상태**다.

정확히는:

- `age=1` 수준의 fresh replay가 대량으로 통과하고
- 그 replay가 `1/16`, `2/16` hard-negative보다 `14/16`, `15/16` easy-positive group 쪽으로 기울어 있다

따라서 `rev8`의 직접 병목은 더 이상 freshness failure가 아니라, **selection polarity failure**다.

---

## 3) baseline과 rev8의 차이는 단순 성능 차이가 아니라, 업데이트 방향 차이다

`Qwen3 baseline-fresh`와 `Qwen3 rev8`을 같은 step `24..32`에서 비교하면:

### baseline

- `critic/score/mean ~= 0.3848`
- `response_length/mean ~= 2632.8`
- `response_length/clip_ratio ~= 0.6254`
- `actor/entropy ~= 0.1577`
- `actor/pg_clipfrac ~= 0.00157`
- `timing_s/step ~= 284.0`

### rev8

- `critic/score/mean ~= 0.3407`
- `response_length/mean ~= 2734.2`
- `response_length/clip_ratio ~= 0.6941`
- `actor/entropy ~= 0.1834`
- `actor/pg_clipfrac ~= 0.00344`
- `timing_s/step ~= 435.0`

즉 `rev8`은 baseline 대비

- 점수는 낮고
- 응답은 더 길고
- length cap 근처에 더 자주 닿고
- entropy는 더 높게 남고
- PPO clip도 더 많이 발생하며
- step time도 훨씬 길다

이 차이는 "느려서 덜 학습했다" 수준이 아니다.

핵심은 `rev8`의 replay가 actor update를 **baseline보다 더 나쁜 방향으로 밀고 있다**는 점이다.

---

## 4) 왜 rev8의 replay는 나쁜 방향으로 작동하는가

### 4.1 replay가 이미 batch의 1/3을 차지한다

`rev8` step `24..32` 평균:

- `actor/replay_seq_frac = 0.3333`
- `actor/replay_ppo_kl ~= 1.13e-4`
- `actor/replay_pg_clipfrac ~= 0.00231`
- `actor/onpolicy_pg_clipfrac ~= 0.00113`

즉 replay는 "보조적인 약한 신호"가 아니라, 이미 actor update의 큰 부분을 차지한다.

그런데 그 replay가 hard correction이 아니라 positive-heavy self-imitation이면, 그 왜곡도 그대로 커진다.

### 4.2 GRPO는 그룹 점수를 response token 전체에 broadcast한다

구현상 GRPO outcome score는 response token 전체에 broadcast된다.

- `verl/trainer/ppo/core_algos.py:549`
- `verl/trainer/ppo/core_algos.py:574`

또 replay는 그룹 전체가 다시 actor batch에 들어간다.

- `verl/trainer/ppo/m2_replay_adapter.py:348`

즉 replay가 `15/16` 그룹이면, 실제 actor가 받는 신호는:

- winner 15개 trajectory 전체에 `+A`
- loser 1개 trajectory 전체에 `-A`

가 걸리는 구조다.

여기서 중요한 것은, positive winner 15개가 "같은 짧은 정답 경로" 하나가 아니라, **서로 약간씩 다른 긴 성공 trajectory family**라는 점이다.

따라서 replay가 `15/16` 위주로 기울면, 정책은:

- 정답 토큰만 강화하는 것이 아니라
- 그 앞의 장황한 reasoning
- 검산 문장
- 불필요한 중간 전개
- 늦게 나오는 최종 답 직전 토큰

까지 함께 강화하게 된다.

이게 baseline보다 응답을 더 길게 만들고, entropy collapse를 늦추는 직접 메커니즘이다.

### 4.3 clip/KL cost가 커지는 이유도 동일하다

`rev8`에서 replay sample은:

- off-policy이고
- age는 1이어도 이미 이전 step 정책에서 생성된 sample이며
- Qwen3 특성상 trajectory가 길다

이 조합 때문에 current policy와 stored old policy 사이 ratio drift가 on-policy보다 더 커지기 쉽다.

그리고 replay의 부호가 positive-heavy이면, loss는 이미 성공했던 토큰들의 확률을 "한 번 더 올리려" 한다.

그 결과:

- upper clip이 더 잘 걸리고
- `actor/replay_pg_clipfrac`가 on-policy보다 높아지며
- KL budget을 replay가 더 많이 소비한다

즉 `rev8`은 현재 실패 경계를 고치는 대신, **이미 잘 된 긴 답변 family를 off-policy로 한 번 더 밀다가 clip 비용을 크게 내는 구조**가 된다.

---

## 5) Qwen2.5 rev5는 왜 반대로 잘 되었는가

비교 대상은 아래 두 실험이다.

- baseline: `grpo-qwen25-math-1.5b-s8-baseline`
- replay: `grpo-qwen25-math-1.5b-s8-m2-replay-rev5-8gpu-fresh`

### 5.1 Qwen2.5 rev5는 replay가 매우 일찍 켜진다

`Qwen2.5 rev5`:

- `first_nonzero replay_used = step 2`
- `first_full replay_used = step 2`

반면 `Qwen3 rev8`:

- `first_nonzero replay_used = step 15`
- `first_full replay_used = step 15`

이 차이는 결정적이다.

`Qwen2.5`에서는 replay가 policy가 아직 충분히 약한 early phase에 개입한다.
반면 `Qwen3 rev8`은 replay가 켜질 때 이미 baseline이 `score 0.30 ~ 0.35` 구간을 넘어가고 있다.

즉:

- `Qwen2.5`: replay가 초기 오류 수정 장치로 작동
- `Qwen3 rev8`: replay가 이미 어느 정도 성공한 정책 위에 뒤늦게 붙는다

### 5.2 Qwen2.5 rev5는 선택된 replay의 polarity가 정확히 반대다

`Qwen2.5 rev5-8gpu-fresh` step `2..10`에서 `used_replay_scores(first_10)`를 모아 보면:

- pooled sample 수: `90`
- 평균 `success_prob ~= 0.0653`
- 평균 `age = 1.0`
- 평균 `m2 ~= 0.000399`
- 평균 `zvp_runtime_neg_frac ~= 0.959`
- 평균 `zvp_runtime_pos_frac ~= 0.041`
- `success_count` 분포:
  - `1/16`: `86`
  - `2/16`: `4`

즉 Qwen2.5 rev5는 거의 pure hard-negative replay다.

반면 `Qwen3 rev8`은 같은 pooled 기준으로:

- 평균 `success_prob ~= 0.8389`
- 평균 `zvp_runtime_pos_frac ~= 0.775`
- `15/16`가 절대다수

즉 두 실험은 같은 "replay"라는 이름을 쓰지만, actor가 실제로 받는 학습 신호의 방향은 정반대다.

### 5.3 Qwen2.5는 길이와 tau margin도 훨씬 유리하다

`Qwen2.5 baseline` early `1..10` 평균:

- `response_length/mean ~= 1087.7`
- `response_length/clip_ratio = 0`

`Qwen2.5 rev5-8gpu-fresh` early `1..10` 평균:

- `response_length/mean ~= 1066.8`
- `response_length/clip_ratio = 0`
- `m2_replay/pass1/acceptance_rate ~= 0.756`
- `m2_replay/selection/replay_used ~= 115.2`

반면 `Qwen3 rev8` mid `24..32`는:

- `response_length/mean ~= 2734.2`
- `response_length/clip_ratio ~= 0.694`
- `acceptance_rate ~= 0.308`

즉 Qwen2.5는:

- trajectory가 짧고
- truncation pressure가 없고
- hard-negative가 tau-safe하며
- 그래서 early replay가 자연스럽게 "실수 억제기"로 작동한다

반면 Qwen3는:

- trajectory가 훨씬 길고
- truncation pressure가 매우 높고
- tau-safe 풀 안에서 easy-positive가 더 많이 살아남고
- 그래서 replay가 "긴 성공 경로 재강화기"로 바뀐다

---

## 6) 그래서 Qwen3 rev8의 열세는 정확히 어떤 인과사슬로 발생하는가

가장 압축하면 다음 순서다.

1. `buffer_full` 때문에 replay 시작이 늦다.
2. replay가 켜질 시점에는 policy가 이미 꽤 많이 올라와 있다.
3. `rlvr_non_degenerate + adv_magnitude` 조합이 `14/16`, `15/16` easy-positive group을 상위로 올린다.
4. 그 easy-positive group 전체가 replay batch로 actor update에 다시 들어간다.
5. 긴 성공 trajectory family 전체가 강화된다.
6. 그 결과:
   - 응답이 더 길어지고
   - entropy collapse가 늦어지고
   - clip/KL cost가 커지고
   - 현재 실패 경계 수정이 희석된다.
7. 그래서 baseline보다 낮은 `critic/score/mean`이 지속된다.

즉 `rev8`의 문제는 "replay가 약하다"가 아니라, **replay가 잘못된 것을 강하게 하고 있다**는 것이다.

---

## 7) 기존 분석에서 보강해야 할 부분

앞선 논의 중 유지해야 할 주장과 줄여야 할 주장을 정리하면:

### 유지해야 할 주장

- `rev7`에서는 stale + tau deadlock이 핵심이었다.
- `Qwen2.5 rev5`와 `Qwen3 rev8`의 차이는 모델명 차이 하나로 설명되지 않는다.
- Qwen3에서는 길고 clip-heavy한 rollout regime이 replay를 훨씬 어렵게 만든다.
- `adv_magnitude`는 Qwen3 rev8에서 easy-positive 편향을 키운 유력 원인이다.

### 더 조심해서 써야 할 주장

- "`Qwen3는 ingress가 너무 적어서 안 된다`"는 단정
  - `rev8`에서는 `rlvr_non_degenerate` 덕분에 replay supply 자체는 꽤 충분하다.
  - 현재 병목은 supply보다 polarity다.
- "`tau를 더 풀면 해결된다`"는 단정
  - `rev8`은 이미 `128/128`을 채운다.
  - 지금은 양이 아니라 어떤 sample이 들어오느냐가 더 중요하다.
- "`stale이 여전히 주범이다`"는 단정
  - `rev8`에서는 selected replay age가 이미 `1.0`이다.
  - stale은 `rev7`의 핵심 병목이었고, `rev8`의 핵심 병목은 아니다.

---

## 8) rev9 설계는 왜 `quarter + plain + non_degenerate + tau0025`인가

앞선 분석을 그대로 따르면, 다음 실험의 목적은 명확하다.

- supply는 유지하고
- replay 시작은 더 앞당기고
- selection polarity만 hard/intermediate 쪽으로 되돌려야 한다

그래서 `rev9`는 아래 조합으로 가는 것이 자연스럽다.

- `start_mode=quarter`
- `ingress_filter_mode=rlvr_non_degenerate`
- `zvp_mode=plain`
- `tau=0.0025`

### 8.1 왜 `quarter`인가

`Qwen3 rev8` early ingress 누적은 실제 로그에서 대략 아래와 같다.

- step 1: `67`
- step 2: 누적 `131`
- step 3: 누적 `210`
- step 4: 누적 `270`
- step 5: 누적 `349`
- step 8: 누적 `558`

버퍼가 `1024`일 때 pre-select 기준 임계값은:

- `quarter = 256`
- `half = 512`
- `buffer_full = 1024`

실제 ready step은:

- `quarter`: step `5`
- `half`: step `9`
- `buffer_full`: step `15`

즉 `quarter`는 Qwen3에서 replay를 다시 early phase로 끌어당기는 가장 직접적인 방법이다.

### 8.2 왜 `non_degenerate`는 유지하는가

이번 라운드의 핵심 질문은:

- 공급이 부족해서 안 되는가
- 아니면 ranking polarity가 틀어져서 안 되는가

를 분리하는 것이다.

따라서 `halfband`로 ingress를 줄이면 해석이 섞인다.

`non_degenerate`를 유지하면:

- replay supply는 충분히 유지되고
- selection만 `adv_magnitude -> plain`으로 바꿔
- polarity 변화만 상대적으로 깨끗하게 볼 수 있다

### 8.3 왜 `plain`인가

현재 `rev8`의 가장 강한 의심점은 `adv_magnitude`다.

성공한 control은 `Qwen2.5 rev5`의 plain ZVP 계열이고,
실패한 `Qwen3 rev8`은 `adv_magnitude`가 easy-positive 편향을 키우는 쪽으로 보인다.

따라서 이번에는:

- `14/16`, `15/16` positive-heavy group에 붙는 bonus를 제거하고
- `1/16`, `2/16`, `3/16` hard/intermediate 쪽이 다시 올라오는지

를 먼저 보는 것이 맞다.

### 8.4 왜 `tau`는 그대로 두는가

이번 라운드에서 `tau`까지 바꾸면 해석이 다시 섞인다.

지금은 먼저:

- 시작 시점을 앞당기고
- ranking polarity만 plain으로 되돌린 뒤
- replay가 실제로 hard/intermediate 쪽으로 이동하는지

를 보는 편이 더 깔끔하다.

즉 `rev9`는 성능 최적화 런이기보다, **원인 판별력이 높은 진단 런**에 가깝다.

---

## 9) 현재 기준에서 가장 설득력 있는 결론

현재까지의 로그와 코드 기준으로 가장 설득력 있는 결론은 아래다.

- `Qwen2.5 rev5`가 잘된 이유는 replay 자체가 좋아서가 아니라,
  - replay가 매우 일찍 켜졌고
  - 선택된 replay가 fresh hard-negative였으며
  - trajectory가 짧고 tau-safe해서
  - replay가 실제로 verbosity suppression + error correction으로 작동했기 때문이다.

- `Qwen3 rev8`이 진 이유는 replay가 안 들어와서가 아니라,
  - replay가 늦게 켜졌고
  - 선택된 replay가 fresh easy-positive로 기울었으며
  - 긴 trajectory 전체를 다시 강화하는 방향으로 작동해
  - baseline보다 더 긴 응답, 더 높은 entropy, 더 큰 clip/KL cost를 만들었기 때문이다.

가장 압축하면:

- `Qwen2.5 rev5`: `fresh hard-negative suppressor`
- `Qwen3 rev8`: `fresh easy-positive self-imitation`

열세의 본질은 여기에 있다.

---

## 10) 현재 반영된 구현/실험 상태

이번 분석을 바탕으로, 코드와 실행 스크립트는 다음 상태까지 반영되었다.

### 코드

- `start_mode`에 `quarter`, `half` 옵션 추가
- 관련 gating metric 추가
- CPU 테스트 추가 및 통과

### 실행 스크립트

- 신규 실행 파일:
  - `RL_side_3/grpo-qwen3-1.7b-s8-m2-replay-rev9.sh`
- 핵심 설정:
  - `start_mode=quarter`
  - `ingress_filter_mode=rlvr_non_degenerate`
  - `zvp_mode=plain`
  - `tau=0.0025`

즉 `rev9`는 현재 분석에서 도출된 "가장 적절한 다음 한 수"를 그대로 구현한 실험 축이다.
