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

## 2) 먼저 정의해 둘 용어

이 문서에서 자주 쓰는 지표는 아래 의미로 사용한다.

- `success_prob`
  - 한 query group 안에서 성공 rollout 비율
  - `rollout.n = 16`이면 `1/16 = 0.0625`, `8/16 = 0.5`
- `age`
  - `current_step - last_training_step`
  - 단순히 버퍼에 들어온 지 몇 step이 아니라, 마지막으로 학습에 반영된 이후 몇 step이 지났는가를 뜻한다
- `m2`
  - response token 위에서 `new_log_prob - old_log_prob`의 평균 제곱 drift
  - accept iff `m2 <= tau`
- `ingress`
  - 현재 step on-policy group 중 replay buffer에 새로 들어가는 group
  - `rev7` 기준 ingress filter는 `rlvr_halfband`, 즉 `1 <= success_count <= 8`
- `hard negative`
  - `1/16`, `2/16`처럼 소수만 성공하고 대부분 실패한 group
  - 현재 policy가 자신 있게 틀리는 token이 많이 포함된 group을 뜻하는 경우가 많다
- `response_length/clip_ratio`
  - PPO clipping이 아니라, 응답이 길이 제한에 닿은 비율에 가까운 proxy다
  - `actor/pg_clipfrac`, `critic/vf_clipfrac`와는 의미가 다르다

중요한 구현 사실도 하나 미리 적어 둔다.

- replay selection은 `select -> append` 순서다
- 즉 현재 step에서 방금 생성한 on-policy sample은 같은 step의 replay 후보가 아니다
- 현재 step에서 볼 수 있는 freshest replay candidate는 보통 `age = 1`인 이전 step sample이다

---

## 3) 핵심 결론

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

여기서 `hard negative`는 대략 `1/16`, `2/16`처럼 대부분 실패한 group을 뜻한다. 현재 sign-only ZVP는 이런 group 안의 negative-advantage token에서 `p`를 더하고 advantage magnitude는 무시하므로, "희귀한 성공 1개 + 자신 있게 틀리는 다수의 패자" 구조를 자주 상위로 올린다.

즉 현재 `Qwen3 rev7`의 핵심 병목은 **ZVP family 자체의 실패**라기보다, 더 정확히는 **freshness failure 위에 쌓인 M2 gate incompatibility**다.

---

## 4) 로그에서 바로 확인되는 사실

### 4.1 Qwen3 rev7: replay가 사실상 죽어 있다

`grpo-qwen3-1.7b-s8-m2-replay-rev7-tau001-b1024` tail 로그:

- `m2_replay/pass1/accepted_groups = 0`
- `m2_replay/selection/replay_used = 0`
- `m2_replay/pass1/rejected_by_tau = 1024`

즉 replay 경로는 "약하게 기여하는 수준"이 아니라, M2 gate에 의해 거의 완전히 차단되어 있다.

validation도 baseline 대비 지속적인 우위를 보여주지 않는다.

- baseline step 100 avg acc: 약 `0.4115`
- rev7 step 100 avg acc: 약 `0.4069`

### 4.2 Qwen2.5-Math replay는 실제로 작동한다

`grpo-qwen25-math-1.5b-s8-m2-replay-rev5-8gpu-fresh` 기준:

- `replay_used` 평균: 약 `127.16`
- tail `replay_used`: `128`
- replay가 0이 아닌 step 비율: 약 `0.9935`

즉 replay는 단순히 설정만 켜진 것이 아니라, 거의 항상 target을 채우고 있다.

### 4.3 Qwen2.5 rev5는 초기 동일 구간 비교에서도 baseline보다 낫다

late collapse 착시를 피하기 위해 baseline과 rev5를 같은 첫 `152` step까지만 비교했을 때:

- baseline `critic/score/mean` tail: 약 `0.386`
- rev5 `critic/score/mean` tail: 약 `0.410`

즉 rev5의 이득은 단순한 후반 붕괴 회피가 아니라, 초기 구간에서도 일관된 성능 상승을 동반한다.

---

## 5) 결정적 차이: high-ZVP 후보가 fresh하냐 stale하냐

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

## 6) `age`는 refresh되지만, FIFO eviction은 사라지지 않는다

`age`는 단순 FIFO age가 아니라 `last_training_step` 기준 age다.

코드 기준 의미는:

- `age = current_step - last_training_step`
- `last_training_step`는 처음에는 insertion step으로 들어간다
- replay group이 실제로 사용되면 현재 step으로 다시 갱신된다

즉:

- replay가 잘 돌면 useful candidate는 반복 사용되는 동안 fresh하게 유지될 수 있고
- replay가 전혀 안 돌면 `last_training_step`이 갱신되지 않아 age가 계속 증가한다

하지만 이것이 "선택된 sample이 영구히 버퍼에 남는다"는 뜻은 아니다.

- 버퍼 eviction은 여전히 FIFO다
- 버퍼가 가득 차면 가장 먼저 들어온 group이 `popleft()`로 제거된다
- 즉 selection은 priority freshness를 바꿀 뿐이고, residence time 자체를 무한히 늘리지는 못한다

정확히는:

- `selection/use`는 `last_training_step`를 갱신한다
- `FIFO eviction`은 `insertion_order`를 기준으로 계속 작동한다

따라서 `Qwen2.5`에서 보이는 것은 "유용한 group이 영구 보존된다"가 아니라, **버퍼에 살아 있는 동안 계속 fresh하게 유지될 수 있다**는 쪽에 가깝다.

---

## 7) FIFO인데 왜 버퍼가 "늙어 보이는가"

FIFO는 단지 "최근 `max_query_groups`개의 ingress group만 남긴다"는 뜻이다.

버퍼 평균 age가 작아야 한다는 뜻은 아니다.

버퍼 크기가 `1024`, step당 새로 들어오는 ingress group 수가 `m`이면:

- 버퍼는 대략 최근 `1024 / m` step의 history를 담고
- 평균 age는 그 절반 수준이 된다

여기서 `m`은 현재 설정 기준 `rlvr_halfband`, 즉 `1..8/16` success group만 세어서 계산한 값이다.

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

## 8) 왜 Qwen3는 fresh ingress가 더 적은가

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

## 9) 왜 hard-negative group이 높은 ZVP를 받는가

현재 ZVP 구현은 sign-only다.

- `A > 0`이면 `1 - p`
- `A < 0`이면 `p`
- advantage magnitude는 의도적으로 무시

GRPO outcome advantage는 group 내부 상대비교다.

예를 들어 16개 rollout 중 1개만 맞으면:

- 맞은 1개 rollout은 positive advantage
- 나머지 15개 rollout은 negative advantage

현재 ZVP는 advantage 크기를 무시하고 token 평균을 내기 때문에, `1/16` 그룹은 구조적으로 강한 점수를 받는다.

- positive rollout은 매우 적고
- negative rollout은 매우 많으며
- 그 negative rollout에 높은 probability를 주고 있으면 score가 크게 오른다

이때 "on-policy라서 무조건 score가 커진다"고 일반화하는 것은 과하지만, current 또는 near-current policy가 실제로 생성한 실패 trace 안에는 high-probability negative-advantage token이 많이 남을 수 있으므로, sign-only ZVP의 confident-mistake bias가 더 강하게 드러날 수는 있다.

그래서 현재 sign-only ZVP에서는 `1/16`형 group이 top-ZVP를 독점하기 쉽다.

중요한 예외:

- `0/16` group은 자동으로 top-ZVP가 되지 않는다
- 모두 같은 정도로 실패하면 GRPO advantage 자체가 거의 0으로 수렴하기 때문이다

즉 현재 ZVP는 "순수한 난이도 점수"보다는

- "희귀한 승자 1개 + 자신 있게 틀리는 다수의 패자"

를 높은 점수로 보는 성격이 더 강하다.

---

## 10) 왜 pure ZVP는 Qwen2.5 rev5에서는 되고, Qwen3 rev7에서는 안 되는가

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

## 11) 왜 Qwen3에서는 M2가 이렇게 공격적으로 막히는가

Replay gate:

- accept iff `m2 <= tau`
- 현재 `tau = 0.001`

`m2`는 response token 위에서 old/new log-prob 차이의 평균 제곱값이다.

같은 age에서도 Qwen3가 더 불리한 이유는 두 층으로 봐야 한다.

### 11.1 훨씬 길고 자주 잘리는 trajectory

late tail 관측:

- `Qwen2.5 rev5`: `response_length/mean ~= 797`, `response_length/clip_ratio = 0`
- `Qwen3 rev7`: `response_length/mean ~= 2338`, `response_length/clip_ratio ~= 0.44`

여기서의 `clip_ratio`는 PPO clipping이 아니라, 응답이 length limit에 닿은 비율에 가까운 proxy다.

중요한 점은, 더 길고 unfinished한 suffix가 많이 포함된 trajectory일수록, current policy가 그 branch 전체를 조금 덜 선호하게 되었을 때 suffix 다수 토큰에서 작은 drift가 넓게 누적된다는 데 있다.

`m2`는 response token 전체 평균 제곱 drift이므로,

- 짧은 응답에서는 작은 drift가 국소적으로 끝날 수 있지만
- 길고 unfinished한 응답에서는 many-small-drifts가 평균값 자체를 tau 밖으로 밀기 쉽다

간단한 예시:

- 길이 `2400` 응답에서 뒤 `1200` token만 평균 `0.045` 정도의 log-prob drift를 보이면
- 평균 제곱 drift는 대략 `0.001` 수준에 닿을 수 있다
- 반면 길이 `800` 응답에서 같은 정도 drift가 뒤 `200` token에만 나타나면 평균은 훨씬 작게 남는다

즉 `Qwen3`에서는 one-step drift가 작아도, **길고 불안정한 suffix가 많아서** `m2 <= 0.001`를 만족하기 어렵다.

### 11.2 지금의 직접 병목은 ranking보다 distribution에 더 가깝다

질문: "버퍼를 끝까지 보면 ZVP가 낮더라도 결국 m2-safe sample 하나쯤 고르지 않나?"

late-stage `Qwen3 rev7`에서는 로그가 이미 답을 준다.

- `scanned_groups = 1024`
- `rejected_by_tau = 1024`
- `accepted_groups = 0`

즉 이건:

- top-ranked 몇 개만 보고 멈춘 것이 아니라
- 버퍼 전체를 끝까지 다 봤는데도
- `tau-safe` sample이 없었다

는 뜻이다.

이건 아주 중요하다.

즉 `Qwen3 rev7`의 문제는 두 겹이다.

1. sign-only ZVP가 stale hard negative를 위로 올리는 ranking 문제
2. late-stage에는 full buffer 자체가 tau-incompatible해지는 distribution 문제

그래서 이 시점의 직접 병목은 "정렬만 바꾸면 해결된다"가 아니라, **버퍼 안 sample 집합 자체가 현재 gate와 맞지 않는다**는 데 있다.

### 11.3 replay refresh 부재는 직접 원인이라기보다 증폭 메커니즘이다

replay가 전혀 안 돌기 때문에:

- selected group이 없다
- `last_training_step`가 갱신되지 않는다
- stale group이 stale한 채 남는다
- top-ranked group이 tau-safe 영역에서 더 멀어진다

이건 직접 초기 원인이라기보다 **deadlock을 키우는 amplifier**에 가깝다.

예시:

- step `t`에서 후보 A가 `age = 7`, `m2 = 0.0013`이라 reject
- replay가 안 되었으므로 `last_training_step`는 그대로
- step `t+1`에서 A는 `age = 8`
- trajectory가 길고 clipped되어 있으면 current policy와의 drift가 같거나 더 커질 수 있음
- 다시 reject

즉:

- ranking이 stale sample을 위로 올리고
- gate는 그 stale sample을 버리고
- replay는 0이 되고
- refresh도 0이 되어 stale 상태가 더 굳어진다

이게 현재 deadlock이다.

---

## 12) magnitude-aware ZVP는 도움이 될까

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

## 13) ingress를 `1..15`로 넓히면 도움이 될까

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

그래도 논문용 정당화는 `1..12`, `2..12` 같은 임의 cutoff보다 훨씬 쉽다.

- `0/16`, `16/16`은 degenerate group
- within-group reward variance가 거의 0
- GRPO replay signal도 거의 없다

즉 극단 두 개만 제외하는 방식은 방어가 가능하다.

---

## 14) 설정별로 어떤 정답률 근처가 많이 샘플링될까

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

여기에 low-success 쪽 잔여 편향이 완전히 사라지진 않을 수 있다.

- magnitude-aware는 `1/16`의 count bias를 줄인다
- 하지만 token term 자체는 여전히 negative-advantage token에서 `p`를 사용한다
- current 또는 near-current policy가 생성한 trace 안에는 high-probability negative-advantage token이 많이 남을 수 있다

따라서 더 안전한 예상은:

- peak는 middle band로 이동하되
- score 비대칭 때문에 low-success 쪽으로 약간의 잔여 편향이 남을 수 있다

---

## 15) 현재 가장 정확한 해석

강한 버전:

- "`Qwen3`가 더 똑똑하니 replay가 안 맞는다"

는 너무 거칠다.

더 정확한 해석은:

1. `Qwen3`는 step당 halfband group을 덜 만든다
2. 그래서 FIFO여도 buffer가 더 긴 history를 담는다
3. replay refresh 부재는 직접 원인이라기보다 stale 누적을 키우는 증폭 메커니즘이다
4. pure sign-only ZVP는 stale hard negative를 계속 top으로 올린다
5. long clipped trajectory가 그 stale candidate를 더 쉽게 tau 밖으로 민다
6. late-stage에는 top-ranked 몇 개만의 문제가 아니라 full buffer 자체가 tau-incompatible해진다

즉 실제 문제는 "지능" 하나가 아니라:

- thinner halfband supply
- stronger bimodality
- stale ranking dynamics
- harsher trajectory geometry
- full-buffer gate incompatibility

의 결합이다.

---

## 16) 현재 가장 타당한 수정 방향과 그 한계

지금 기준으로 가장 설득력 있는 다음 단계는:

1. ingress를 `1..8`에서 `1..15`로 확장
2. sign-only ZVP를 magnitude-aware ZVP로 변경

이 조합이 현재 가장 방어 가능하고도 실제 병목을 어느 정도 찌른다.

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

하지만 냉정하게 보면, 이 두 가지는 **근본 해결책이라기보다 강한 diagnostic patch**에 가깝다.

이유:

- 이 둘은 ranking과 ingress distribution을 바꾸지만, `m2 <= tau`라는 absolute gate 구조 자체는 바꾸지 않는다
- 길고 truncated된 trajectory가 one-step drift만으로도 tau 밖으로 밀리는 문제를 직접 해결하지 않는다
- same-step sample은 원래 replay 후보가 아니므로, freshest candidate mass를 늘리는 데도 한계가 있다

즉 `rev8`의 성공 기준은 최종 성능 향상보다 먼저 아래여야 한다.

- `replay_used > 0`가 안정적으로 유지되는가
- top-ranked candidate age가 내려가는가
- selected success bucket이 `1/16` 독점에서 벗어나는가
- accepted M2가 tau 아래에서 안정적으로 유지되는가

최종 성능 평가는 그 다음이다.

---

## 17) tau 완화: 어디까지 여는 것이 타당한가

지금까지의 분석을 종합하면, `Qwen3 non-base`에서 `tau=0.001`은 지나치게 보수적일 가능성이 높다.

다만 이걸 곧바로 "`0.002`로 올리면 된다"로 가면 위험하다.

핵심은 아래 두 가지를 동시에 봐야 한다.

1. 기존 successful run에서 실제로 받아들여진 `accepted_m2_mean` 구간이 어디였는가
2. 현재 `Qwen3 non-base`에서 `tau`를 열면 어떤 종류의 stale sample까지 같이 열릴 위험이 있는가

### 17.1 기준점: 기존 successful run의 accepted M2

`Qwen2.5 rev2`, `Qwen3-Base rev6`, `Qwen3-Base rev5(tau=0.002)`를 기준점으로 보면:

- `Qwen2.5 rev2 (tau=0.001)`
  - `accepted_m2_mean` 평균 `~ 0.000584`
  - `p90 ~ 0.000682`
  - `max ~ 0.000958`
- `Qwen3-Base rev6 (tau=0.001)`
  - `accepted_m2_mean` 평균 `~ 0.000752`
  - `p90 ~ 0.000780`
  - `max ~ 0.000805`
- `Qwen3-Base rev5 (tau=0.002)`
  - `accepted_m2_mean` 평균 `~ 0.001307`
  - `p90 ~ 0.001543`
  - `max ~ 0.001750`

이 숫자가 의미하는 바는 명확하다.

- `tau=0.001`은 원래 상당히 conservative한 gate다
- `tau=0.002`는 단순한 소폭 완화가 아니라, replay acceptance operating point 자체를 꽤 크게 이동시키는 값이다

### 17.2 현재 Qwen3 non-base에서 보이는 상황

`Qwen3 rev7`에서는 nonzero accept가 사실상 한 번뿐이었다.

- `accepted_m2_mean ~ 0.000967`
- nonzero accept step은 단 1회
- late-stage에는 `scanned_groups = 1024`, `rejected_by_tau = 1024`, `accepted_groups = 0`

즉 현재 non-base는:

- `0.001` 바로 아래에 걸쳐 있는 sample은 아주 드물게 존재한다
- 하지만 buffer 다수는 `0.001`을 분명히 넘는다

이 구조라면 첫 probe로 가장 타당한 값은 `0.0015`다.

### 17.3 왜 0.0015가 1차 후보인가

`tau`는 squared-drift threshold이므로, 실제 허용 RMS drift는 아래처럼 변한다.

- `sqrt(0.001)  ~ 0.0316`
- `sqrt(0.0015) ~ 0.0387`
- `sqrt(0.002)  ~ 0.0447`

즉:

- `0.001 -> 0.0015`는 per-token RMS drift budget을 약 `22%` 늘린다
- `0.001 -> 0.002`는 약 `41%` 늘린다

따라서 해석은 이렇다.

- `0.0015`: 보수적 완화
- `0.002`: 공격적 완화

현재 `Qwen3 non-base`는

- response가 길고
- truncation pressure가 높고
- stale / unfinished tail이 많고
- non-degenerate ingress도 얇다

는 점을 감안하면, 첫 카드로 `0.002`를 쓰는 것은 stale bad trace까지 과하게 열 위험이 있다.

### 17.4 현재 권장안

현 시점의 가장 방어 가능한 순서는:

1. `tau = 0.0015`
2. 그래도 `replay_used`가 거의 0이면 `tau = 0.002`

즉:

- 하나만 고르라면 `0.0015`
- `0.002`는 구조 rescue를 노리는 2차 카드

로 보는 게 맞다.

### 17.5 왜 0.002를 상한에 가깝게 보나

`Qwen3-Base rev5 (tau=0.002)`에서는:

- `replay_used = 128`이 안정적으로 채워졌고
- `accepted_m2_mean`도 `~ 0.0013`대로 올라갔다
- scan depth도 크게 줄었다

즉 `0.002`는 "조금 더 여는 값"이 아니라, gate의 실질 operating point를 바꾸는 값이다.

이건 짧고 안정적인 `Qwen3-Base`에서는 잘 작동했지만,
긴 clipped trajectory가 많은 `Qwen3 non-base`에서는
좋은 replay뿐 아니라 나쁜 stale replay도 같이 열 수 있다.

그래서 `0.002`는 쓸 수 있는 값이지만, 기본값으로 바로 점프하기엔 공격적이다.

### 17.6 tau를 올릴 때 반드시 같이 봐야 할 로그

`replay_used`만 보면 안 된다.

최소한 아래 다섯 개를 같이 봐야 한다.

- `m2_replay/selection/replay_used`
- `m2_replay/pass1/accepted_m2_mean`
- `actor/replay_ppo_kl`
- `actor/replay_pg_clipfrac`
- `m2_replay/selection/used_zvp_neg_frac_mean`

해석:

- `replay_used`는 늘었는데
- `replay_ppo_kl`와 `replay_pg_clipfrac`가 같이 급증하면

그건 replay activation 회복이라기보다, stale sample을 과하게 연 것일 수 있다.

### 17.7 현재 문맥에서의 냉정한 결론

`tau` 완화는 지금 당장 시도할 가치가 있다.

하지만:

- 이건 근본 해결책이 아니라 gate 완화다
- 특히 `Qwen3 non-base`의 long / clipped geometry 자체를 고치지 않는다
- 따라서 `tau`를 올린다고 해서 durable quality gain이 바로 보장되지는 않는다

그럼에도 불구하고, 현재 `0.001`은 너무 빡빡해 보이므로,

- **1차 probe로 `0.0015`**
- **2차 rescue로 `0.002`**

가 지금까지의 로그와 비교 실험들을 기준으로 가장 방어 가능한 선택이다.

---

## 18) 최종 요약

`Qwen2.5-Math`에서 replay가 되는 이유는 작은 모델이라서가 아니라, replay 가치가 높은 group이 fresh하고 짧고 tau-safe하게 남아 있기 때문이다. `Qwen3 rev7`이 실패하는 이유는 stale hard negative가 ranking 상단을 점유하고, 긴 clipped trajectory가 그 stale group을 더 M2-incompatible하게 만들기 때문이다. FIFO는 정상이다. 다만 halfband ingress가 얇아 같은 1024 buffer가 더 긴 step history를 담고, replay refresh도 전혀 일어나지 않는다. 여기에 sign-only ZVP가 `1/16`형 group을 과도하게 우선시한다. late-stage에는 top 후보만의 문제가 아니라 full buffer 자체가 tau 밖으로 밀려 있었기 때문에, 단순히 더 아래까지 스캔한다고 해결되지 않았다. 현재까지 가장 근거가 강한 다음 수정 방향은, ingress를 모든 non-degenerate group(`1..15/16`)으로 넓히고 sign-only ZVP를 advantage-magnitude-aware ZVP로 바꾸는 것이다. 다만 이것은 ranking과 공급을 바꾸는 강한 진단 패치이지, 현재 `Qwen3 non-base`의 absolute gate mismatch를 직접 해결하는 근본 패치라고 단정할 수는 없다.

---

## 19) 2026-03-06 코드 반영

위 가설 수준으로 남겨 두지 않고, 실제 실행 코드에 아래 두 옵션을 추가했다.

### 19.1 ingress filter mode 확장

기존:

- `selection.ingress_filter_mode = rlvr_halfband`
- 의미: `1 <= success_count <= floor(n / 2)`

추가:

- `selection.ingress_filter_mode = rlvr_non_degenerate`
- 의미: `1 <= success_count <= n - 1`

의도:

- `0/n`, `n/n`의 degenerate group만 제외
- halfband보다 더 넓은 fresh 공급을 허용
- 기존 script는 기본값이 유지되므로 그대로 실행 가능

구현 위치:

- `verl/trainer/ppo/m2_replay_adapter.py`
- `verl/trainer/ppo/ray_trainer.py`

### 19.2 ZVP mode 확장

기존:

- `selection.zvp_mode = sign_only`
- 기본 동작이며 기존 score를 그대로 유지

추가:

- `selection.zvp_mode = adv_magnitude`
- 같은 bounded token term을 쓰되, `|adv|`로 가중한 self-normalized score 사용

의도:

- sign-only ZVP의 `1/16` 과선호를 완화
- score scale을 크게 흔들지 않으면서 advantage magnitude를 반영
- on-policy 초기 cache와 runtime replay 재평가 양쪽에 동일하게 적용

구현 위치:

- `verl/trainer/ppo/m2_replay.py`
- `verl/trainer/ppo/m2_replay_adapter.py`
- `verl/trainer/ppo/ray_trainer.py`

### 19.3 replay start_mode 추가와 rev8 스크립트 업데이트

이번에는 replay 시작 시점을 bool 하나로 표현하던 기존 방식을 정리하고, 의미가 분명한 schedule mode를 추가했다.

추가:

- `schedule.start_mode = immediate`
- `schedule.start_mode = buffer_full`
- `schedule.start_mode = two_turnovers`

의미:

- `immediate`
  - buffer가 덜 찬 상태에서도 바로 replay selection 시작
  - 새 기본값
- `buffer_full`
  - buffer가 capacity를 한 번 채우는 즉시 replay selection 시작
- `two_turnovers`
  - buffer가 가득 찬 뒤, insertion count가 `2 * capacity`를 넘은 시점부터 selection 시작
  - 기존 `one_turnover_gate=true`의 실제 의미와 동일

하위호환:

- `schedule.start_mode`가 없고 `schedule.one_turnover_gate=true`면 `two_turnovers`로 해석
- `schedule.start_mode`가 없고 `schedule.one_turnover_gate=false`면 `immediate`로 해석

즉 기존 실행 코드는 깨지지 않고, 새 스크립트부터는 `start_mode`를 명시적으로 쓰게 된다.

구현 위치:

- `verl/trainer/ppo/ray_trainer.py`

새 실행 스크립트:

- `RL_side_3/grpo-qwen3-1.7b-s8-m2-replay-rev8.sh`

`rev8`은 이제 `rev7` 대비 아래 네 가지가 다르다.

- `+algorithm.m2_replay.selection.ingress_filter_mode=rlvr_non_degenerate`
- `+algorithm.m2_replay.selection.zvp_mode=adv_magnitude`
- `+algorithm.m2_replay.schedule.start_mode=buffer_full`
- `+algorithm.m2_replay.tau=0.0015`

해석:

- `rlvr_non_degenerate`는 fresh candidate pool을 넓히기 위한 변경
- `adv_magnitude`는 sign-only ZVP의 `1/16` 과선호를 줄이기 위한 변경
- `buffer_full`은 `immediate`보다 보수적이고 `two_turnovers`보다 덜 늦은 중간 모드
- `tau=0.0015`는 `0.001`보다 완화하지만 `0.002`보다는 방어적인 1차 probe 값

또한 기존 `one_turnover_gate=true`를 쓰던 `rev6`, `rev7` 스크립트는 의미를 드러내기 위해
`schedule.start_mode=two_turnovers`로 명시적으로 바꾸었다.

### 19.4 이번 변경의 1차 검증 포인트

`rev8`의 1차 성공 기준은 final accuracy가 아니라 replay activation 복구다.

- `m2_replay/selection/replay_used > 0`가 안정적으로 유지되는가
- top-ranked candidate age가 `rev7`보다 낮아지는가
- selected success bucket이 `1/16` 독점에서 벗어나는가
- `accepted_m2_mean`이 `tau=0.0015` 아래에서 안정적으로 유지되는가

이 네 가지가 먼저 확인되어야 한다.

---

## 20) 2026-03-06 추가 로깅 보강

`rev8` 이후에도 여전히 중요한 질문이 하나 남는다.

- replay가 실제로 actor update에 들어갔을 때
- 그 replay 토큰들이 on-policy 토큰보다 더 강하게 PPO clipping에 걸리는가

기존 W&B 로그만으로는 이 질문에 직접 답할 수 없었다.

- 기존 `actor/pg_clipfrac`는 replay + on-policy가 섞인 전체 actor batch 기준 단일 scalar였다
- 따라서 `replay_used`가 커져도, clipfrac 변화가 replay 때문인지 on-policy 때문인지 분리할 수 없었다
- `critic/vf_clipfrac`는 actor replay와 직접 연결되지 않으므로 이 질문의 핵심 로그가 아니다

이번에 actor batch source mask를 추가하고, source별 metric을 분리 로깅하도록 보강했다.

### 20.1 추가한 actor source mask

- actor batch concat 시 replay sample에는 `m2_replay_source=True`
- on-policy sample에는 `m2_replay_source=False`

이 mask를 actor worker까지 유지한 뒤 policy loss 함수로 넘긴다.

구현 위치:

- `verl/trainer/ppo/m2_replay_adapter.py`
- `verl/workers/actor/dp_actor.py`
- `verl/workers/actor/megatron_actor.py`
- `verl/workers/utils/losses.py`

### 20.2 새로 추가한 W&B 로그

추가 로그는 아래와 같다.

- `actor/replay_pg_clipfrac`
- `actor/onpolicy_pg_clipfrac`
- `actor/replay_pg_clipfrac_lower`
- `actor/onpolicy_pg_clipfrac_lower`
- `actor/replay_ppo_kl`
- `actor/onpolicy_ppo_kl`
- `actor/replay_seq_frac`
- `actor/replay_token_frac`

해석:

- `replay_pg_clipfrac`가 높고 `onpolicy_pg_clipfrac`가 낮다면, replay 토큰이 현재 policy 기준으로 더 aggressive한 update 영역에 있다는 뜻이다
- 반대로 둘이 비슷하면 replay가 특별히 더 clip-heavy하지 않다는 뜻이다
- `replay_token_frac` 없이 clipfrac만 보면 해석이 틀어질 수 있으므로, 반드시 같이 봐야 한다

### 20.3 이번 로깅 보강의 목적

이 로깅은 알고리즘 자체를 바꾸는 변경이 아니라, `rev8` 이후 해석력을 높이기 위한 instrumentation이다.

이제 `rev8`을 보면 최소한 아래 질문에는 직접 답할 수 있다.

- replay가 실제 actor batch에서 얼마나 큰 비중을 차지했는가
- replay 토큰이 on-policy 토큰보다 더 자주 clipped 되었는가
- replay 쪽 KL이 on-policy보다 구조적으로 더 큰가

즉 다음 단계에서는 단순히 `replay_used`만 보는 것이 아니라,

- `replay_used`
- `replay_token_frac`
- `replay_pg_clipfrac`
- `replay_ppo_kl`

를 묶어서 봐야 한다.