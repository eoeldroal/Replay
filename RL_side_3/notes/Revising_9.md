# Revising_9: 왜 Qwen2.5-Math에서는 Replay가 되고, Qwen3에서는 막히는가

## 1) 문제 정의

이번 정리의 출발점은 아래 두 실험이다.

- `RL_side_3/grpo-qwen3-1.7b-s8.sh`
- `RL_side_3/grpo-qwen3-1.7b-s8-m2-replay-rev7.sh`

두 실험은 W&B와 로컬 로그가 모두 쌓이고 있고, `rev7`은 replay가 들어가야 하는 설정이다.

하지만 실제 관측은 기대와 다르다.

- `Qwen/Qwen3-1.7B` (`rev7`)에서는 replay가 baseline 대비 뚜렷한 이득을 만들지 못한다.
- 반면 `Qwen2.5-Math-1.5B`에서는 `rev2`, `rev5` 계열 replay가 분명한 이득을 보인다.

이 문서는 현재까지의 로그와 코드 확인을 바탕으로, 그 차이를 설명하는 가장 설득력 있는 해석을 정리한다.

---

## 2) 핵심 결론

현재 `Qwen3 rev7`의 실패 모드는

- "replay가 약하게나마 돌지만 효과가 작다"

가 아니다.

실제 상태는 다음에 가깝다.

- replay 후보는 만들어진다.
- 하지만 상위 후보 대부분이 stale하다.
- 그 stale 후보들이 `tau=0.001` M2 gate에서 전부 걸린다.
- 그래서 replay가 actor batch에 거의 들어가지 못한다.

반대로 `Qwen2.5-Math`에서는 replay priority가 높은 후보가 아직 충분히 fresh해서 M2 gate를 통과한다.

가장 짧게 요약하면:

- `Qwen2.5-Math`: `fresh, short, tau-safe hard negative`
- `Qwen3 rev7`: `stale, long, clipped, tau-unsafe hard negative`

즉 현재 `Qwen3 rev7`의 핵심 병목은 **ZVP 자체의 실패**가 아니라 **freshness failure**다.

---

## 3) 로그에서 바로 확인되는 사실

### 3.1 Qwen3 rev7: replay가 사실상 죽어 있다

`grpo-qwen3-1.7b-s8-m2-replay-rev7-tau001-b1024` tail 로그:

- `m2_replay/pass1/accepted_groups = 0`
- `m2_replay/selection/replay_used = 0`
- `m2_replay/pass1/rejected_by_tau = 1024`

즉 replay 경로는 "약하게 기여하는 수준"이 아니라, M2 gate에 의해 거의 완전히 차단되어 있다.

validation도 baseline 대비 지속적인 우위를 보여주지 않는다.

- baseline step 100 avg acc: 약 `0.4115`
- rev7 step 100 avg acc: 약 `0.4069`

### 3.2 Qwen2.5-Math replay는 실제로 작동한다

`grpo-qwen25-math-1.5b-s8-m2-replay-rev5-8gpu-fresh` 기준:

- `replay_used` 평균: 약 `127.16`
- tail `replay_used`: `128`
- replay가 0이 아닌 step 비율: 약 `0.9935`

즉 replay는 단순히 설정만 켜진 것이 아니라, 거의 항상 target을 채우고 있다.

### 3.3 Qwen2.5 rev5는 초기 동일 구간 비교에서도 baseline보다 낫다

late collapse 착시를 피하기 위해 baseline과 rev5를 같은 첫 `152` step까지만 비교했을 때:

- baseline `critic/score/mean` tail: 약 `0.386`
- rev5 `critic/score/mean` tail: 약 `0.410`

즉 rev5의 이득은 단순한 장기 붕괴 회피 효과만은 아니다.

---

## 4) 결정적 차이: high-ZVP 후보가 fresh하냐 stale하냐

late-stage 요약:

### Qwen2.5 rev5

- 전체 candidate pool:
  - `success_prob ~= 0.230`
  - `age ~= 5.29`
- top-50 candidate:
  - `success_prob ~= 0.0726`
  - `age ~= 1.61`
- 실제 selected replay group:
  - `success_prob ~= 0.0886`
  - `age ~= 1.10`
  - `m2 ~= 0.00055`

### Qwen3 rev7

- 전체 candidate pool:
  - `success_prob ~= 0.242`
  - `age ~= 8.34`
- top-50 candidate:
  - `success_prob ~= 0.0625`
  - `age ~= 8.03`
- 실제 selected replay group:
  - 없음

이 차이가 핵심이다.

두 모델 모두 top-ZVP는 거의 `1/16` 수준의 hard negative처럼 보인다.

하지만:

- `Qwen2.5 rev5`에서는 그 hard negative가 fresh하다 (`age ~ 1-2`)
- `Qwen3 rev7`에서는 그 hard negative가 이미 stale하다 (`age ~ 8-10`)

이 freshness 차이가 M2 acceptance를 완전히 갈라놓는다.

---

## 5) `age`가 정확히 무엇을 뜻하는가

`age`는 단순히 "FIFO에 들어온 지 몇 step"이 아니다.

코드 기준 의미는:

- `age = current_step - last_training_step`
- `last_training_step`는 처음에는 insertion step으로 들어간다
- replay group이 실제로 사용되면 현재 step으로 다시 갱신된다

관련 파일:

- `verl/trainer/ppo/m2_replay.py`
- `verl/trainer/ppo/ray_trainer.py`

즉:

- replay가 잘 돌면 useful candidate는 반복 사용되면서 fresh하게 유지될 수 있고
- replay가 전혀 안 돌면 `last_training_step`이 갱신되지 않아 age가 계속 증가한다

이게 다음과 같은 자기강화 루프를 만든다.

- `Qwen2.5`: replay 사용 -> age refresh -> top candidate가 계속 fresh
- `Qwen3`: replay 미사용 -> no refresh -> top candidate가 더 stale -> M2 rejection 악화

---

## 6) FIFO인데 왜 버퍼가 "늙어 보이는가"

FIFO는 단지 "최근 `max_query_groups`개의 ingress group만 남긴다"는 뜻이다.

버퍼 평균 age가 작아야 한다는 뜻은 아니다.

버퍼 크기가 `1024`, step당 새로 들어오는 ingress group 수가 `m`이면:

- 버퍼는 대략 최근 `1024 / m` step의 history를 담고
- 평균 age는 그 절반 수준이 된다

late-tail ingress:

- `Qwen2.5 rev5`: `new_groups ~= 88.45 / step`
- `Qwen3 rev7`: `new_groups ~= 65.5 / step`

따라서 history window는 대략:

- `Qwen2.5 rev5`: `1024 / 88.45 ~= 11.6 step`
- `Qwen3 rev7`: `1024 / 65.5 ~= 15.6 step`

관측된 age mean도 거의 그대로 따라간다:

- `Qwen2.5 rev5` full pool age mean: 약 `5.29`
- `Qwen3 rev7` full pool age mean: 약 `8.34`

즉 FIFO가 깨진 것이 아니다.

`Qwen3`가 더 "늙어 보이는" 이유는:

- step당 들어오는 halfband group이 적고
- 그래서 같은 1024칸이 더 긴 step history를 담고
- replay refresh도 전혀 일어나지 않기 때문이다

---

## 7) 왜 Qwen3는 fresh ingress가 더 적은가

현재 replay ingress는 `rlvr_halfband`다. 의미는:

- `1 <= success_count <= floor(G / 2)`
- `rollout.n = 16`이면 `1..8`

late-tail on-policy group 분포:

### Qwen2.5 rev5

- `0/16`: `28.0%`
- `1..8/16`: `34.2%`
- `9..16/16`: `37.8%`

### Qwen3 rev7

- `0/16`: `30.0%`
- `1..8/16`: `25.6%`
- `9..16/16`: `44.4%`

해석:

- `Qwen3`는 halfband group이 더 적다
- `Qwen3`는 high-success group이 더 많다
- 하지만 단순히 "전체적으로 더 좋아졌다"로 끝나지 않는다
- 분포 자체가 더 **bimodal**해졌다고 보는 편이 맞다

즉:

- 잘 푸는 문제는 더 잘 풀고
- 아예 못 푸는 문제도 여전히 존재하며
- 중간대가 더 얇아진 상태다

현재 replay는 바로 그 중간대에 의존하고 있다.

---

## 8) 왜 hard-negative group이 높은 ZVP를 받는가

현재 ZVP 구현은 sign-only다.

- `A > 0`이면 `1 - p`
- `A < 0`이면 `p`
- advantage magnitude는 의도적으로 무시

관련 파일:

- `verl/trainer/ppo/m2_replay.py`

GRPO outcome advantage는 group 내부 상대비교다.

예를 들어 16개 rollout 중 1개만 맞으면:

- 맞은 1개 rollout은 positive advantage
- 나머지 15개 rollout은 negative advantage

현재 ZVP는 advantage 크기를 무시하고 token 평균을 내기 때문에, `1/16` 그룹은 구조적으로 강한 점수를 받는다.

- positive rollout은 매우 적고
- negative rollout은 매우 많으며
- 그 negative rollout에 높은 probability를 주고 있으면 score가 크게 오른다

그래서 현재 sign-only ZVP에서는 `1/16`형 group이 top-ZVP를 독점하기 쉽다.

중요한 예외:

- `0/16` group은 자동으로 top-ZVP가 되지 않는다
- 모두 같은 정도로 실패하면 GRPO advantage 자체가 거의 0으로 수렴하기 때문이다

즉 현재 ZVP는 "순수한 난이도 점수"보다는

- "희귀한 승자 1개 + 자신 있게 틀리는 다수의 패자"

를 높은 점수로 보는 성격이 더 강하다.

---

## 9) 왜 pure ZVP는 Qwen2.5 rev5에서는 되고, Qwen3 rev7에서는 안 되는가

중요한 수정점:

- "pure ZVP 자체가 문제다"는 진단은 충분히 정확하지 않다

반례:

- `Qwen2.5-Math rev5`도
  - `selection.mode = zvp_recency`
  - `zvp_use_recency = false`

즉 pure ZVP 계열인데 잘 작동한다.

따라서 더 정확한 진단은:

- pure ZVP 자체는 작동할 수 있다
- 문제는 **어떤 group이 top-ZVP가 되느냐**다

관측:

- `Qwen2.5 rev5`: top-ZVP group이 hard하지만 fresh
- `Qwen3 rev7`: top-ZVP group이 hard하지만 stale

따라서 병목은 "ZVP family failure"가 아니라 **Qwen3 동역학에서의 freshness failure**다.

---

## 10) 왜 Qwen3에서는 M2가 이렇게 공격적으로 막히는가

Replay gate:

- accept iff `m2 <= tau`
- 현재 `tau = 0.001`

`m2`는 response token 위에서 old/new log-prob 차이의 평균 제곱값이다.

같은 age에서도 Qwen3가 더 불리한 이유는 두 가지다.

### 10.1 훨씬 길고 자주 잘리는 trajectory

late tail 관측:

- `Qwen2.5 rev5`: `response_length/mean ~= 797`, `clip_ratio = 0`
- `Qwen3 rev7`: `response_length/mean ~= 2338`, `clip_ratio ~= 0.44`

즉 `Qwen3` trajectory는 단순히 오래된 것만이 아니라, 더 길고 더 자주 unfinished 상태다.

이러면 old/new log-prob mismatch가 더 커지므로 M2가 더 빨리 커진다.

### 10.2 replay refresh loop 자체가 없다

replay가 전혀 안 돌기 때문에:

- selected group이 없다
- `last_training_step`가 갱신되지 않는다
- stale group이 stale한 채 남는다
- top-ranked group이 tau-safe 영역에서 더 멀어진다

이게 현재 deadlock이다.

---

## 11) magnitude-aware ZVP는 도움이 될까

가능성은 높다. 하지만 그것만으로 충분하진 않다.

도움이 되는 이유:

- sign-only ZVP는 `1/16`을 과하게 선호한다
- magnitude-aware ZVP는 이 구조적 bias를 줄인다
- `n=16` GRPO에서 theoretical average `|adv|`는 `1/16`에서 가장 낮고 중간대에서 가장 높다

계산된 average absolute GRPO advantage:

- `1/16`: 약 `0.4688`
- `4/16`: 약 `0.8385`
- `8/16`: 약 `0.9682`

관측된 bucket-wise mean ZVP에 theoretical `avg|adv|`를 곱한 surrogate를 보면, ranking peak는 `1/16`에서 멀어지고 대략 아래로 이동할 가능성이 높다.

- `3/16 .. 6/16`
- 특히 `4/16 .. 5/16`

이건 현재의 extreme hard-negative bias를 줄이는 방향이다.

하지만 `Qwen3 rev7`에서는 거의 모든 success bucket이 이미 stale하다.

- `1/16` age mean: 약 `8.40`
- `2/16` age mean: 약 `8.48`
- `3/16` age mean: 약 `8.04`
- `4/16` age mean: 약 `8.41`
- `7/16` age mean: 약 `8.21`
- `8/16` age mean: 약 `8.23`

즉 magnitude-aware ZVP만 넣으면

- `1/16 stale` 대신
- `4/16 stale`

를 더 많이 고를 수는 있어도, stale 문제 자체는 그대로 남을 수 있다.

결론:

- magnitude-aware ZVP는 유망하다
- 하지만 freshness 개선 없이는 충분조건이 아니다

---

## 12) ingress를 `1..15`로 넓히면 도움이 될까

가능성은 있다. 다만 의미를 정확히 이해해야 한다.

`n=16`에서 ingress를 `1..8`에서 `1..15`로 넓히면, 현재 `Qwen3` tail 분포 기준:

- 현재 ingress mass: 약 `65.5 groups / step`
- 가정상 `1..15` ingress mass: 약 `128.4 groups / step`

즉 effective buffer history span이 거의 절반으로 줄어든다.

- 현재 mean-age proxy: 약 `7.8`
- `1..15` mean-age proxy: 약 `4.0`

따라서 `1..15`는 freshness를 크게 개선할 가능성이 있다.

하지만 비용도 있다.

- 추가로 들어오는 mass의 거의 절반이 `9..15/16` high-success group이다

즉 `1..15`는 단순한 freshness tweak가 아니다.

- "middle-band replay"
에서
- "거의 모든 non-degenerate group replay"

로 replay의 의미가 바뀐다.

그래도 논문용 정당화는 `1..12`, `2..12` 같은 임의의 cutoff보다 훨씬 쉽다.

- `0/16`, `16/16`은 degenerate group
- within-group reward variance가 거의 0
- GRPO replay signal도 거의 없다

즉 극단 두 개만 제외하는 방식은 방어가 가능하다.

---

## 13) 설정별로 어떤 정답률 근처가 많이 샘플링될까

### 현재: `1..8` + sign-only ZVP

예상 dominant bucket:

- 거의 `1/16` 근처

이건 이미 로그에서 확인된다.

### `1..15` + sign-only ZVP

예상 dominant bucket:

- 여전히 low-success 쪽, 아마 `1/16` 근처

freshness는 좋아지겠지만 sign-only bias는 남는다.

### `1..15` + magnitude-aware ZVP

예상 dominant replay-used neighborhood:

- 중간 정답률 쪽으로 이동
- 대략 `3/16 .. 6/16`
- 가장 강한 mass는 `4/16 .. 5/16`

중요한 단서:

- 완전히 정중앙일 필요는 없다
- 현재 ZVP 정의와 on-policy 분포 비대칭 때문에 low-success 쪽으로 약간 기울 가능성은 여전히 있다

---

## 14) 현재 가장 정확한 해석

강한 버전:

- "`Qwen3`가 더 똑똑하니 replay가 안 맞는다"

는 너무 거칠다.

더 정확한 해석은:

1. `Qwen3`는 step당 halfband group을 덜 만든다
2. 그래서 FIFO여도 buffer가 더 긴 history를 담는다
3. replay가 한 번도 안 도니 age refresh가 일어나지 않는다
4. pure sign-only ZVP는 stale hard negative를 계속 top으로 올린다
5. long clipped trajectory가 그 stale candidate를 더 쉽게 tau 밖으로 민다

즉 실제 문제는 "지능" 하나가 아니라:

- thinner halfband supply
- stronger bimodality
- stale ranking dynamics
- harsher trajectory geometry

의 결합이다.

---

## 15) 현재 가장 타당한 수정 방향

지금 기준으로 가장 설득력 있는 다음 단계는:

1. ingress를 `1..8`에서 `1..15`로 확장
2. sign-only ZVP를 magnitude-aware ZVP로 변경

이 조합이 현재 가장 방어 가능하고도 실제 병목을 잘 찌른다.

이유:

- `1..15`는 `1..12`, `2..12` 같은 임의 cutoff보다 논문에서 설명하기 쉽다
- magnitude-aware ZVP는 관측된 `1/16` 과선호를 직접 줄인다
- 둘을 같이 쓰면
  - freshness
  - extreme hard-negative ranking bias

를 동시에 건드릴 수 있다

기대 효과:

- replay activation 회복
- stale `1/16` dominance 완화
- replay usage의 middle-success 이동
- top-ranked group의 `m2 <= tau` 가능성 증가

보장되지 않는 것:

- 이것만으로 최종 validation uplift가 반드시 생긴다는 보장은 없다

따라서 다음 실험의 1차 성공 기준은 아래여야 한다.

- `replay_used > 0`가 안정적으로 유지되는가
- top-ranked candidate age가 내려가는가
- selected success bucket이 `1/16` 독점에서 벗어나는가
- accepted M2가 tau 아래에서 안정적으로 유지되는가

최종 성능 평가는 그 다음이다.

---

## 16) 최종 요약

`Qwen2.5-Math`에서 replay가 되는 이유는 작은 모델이라서가 아니라, replay 가치가 높은 group이 fresh하고 짧고 tau-safe하게 남아 있기 때문이다. `Qwen3 rev7`이 실패하는 이유는 stale hard negative가 ranking 상단을 점유하고, 긴 clipped trajectory가 그 stale group을 더 M2-incompatible하게 만들기 때문이다. FIFO는 정상이다. 다만 halfband ingress가 얇아 같은 1024 buffer가 더 긴 step history를 담고, replay refresh도 전혀 일어나지 않는다. 여기에 sign-only ZVP가 `1/16`형 group을 과도하게 우선시한다. 현재까지 가장 근거가 강한 다음 수정 방향은, ingress를 모든 non-degenerate group(`1..15/16`)으로 넓히고 sign-only ZVP를 advantage-magnitude-aware ZVP로 바꾸는 것이다. 단, 첫 목표는 최종 점수 향상보다 replay activation 복구여야 한다.
