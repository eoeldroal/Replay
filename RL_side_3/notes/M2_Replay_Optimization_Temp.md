# M2 Replay Optimization Temp

이 문서는 현재 `m2_replay` 경로의 병목과, 로직 변경을 최소화하면서 적용 가능한 최적화 방향을 임시로 정리한 문서다.
구현 완료 후에는 삭제하거나 다른 문서로 흡수하는 것을 전제로 한다.

## 1. 목적

현재 관찰된 문제는 두 가지다.

1. `m2_build_actor_batch`가 지나치게 비싸다.
2. `actor_update_policy_compute`도 baseline 대비 과도하게 커진다.

이 둘을 구분해서 봐야 한다.

- `m2_build_actor_batch`는 replay 전용 selection/control-flow 병목이다.
- `actor_update_policy_compute`는 기본 `verl` actor update 엔진 자체의 문제라기보다, replay가 actor에 넣는 배치의 크기와 길이 분포가 baseline보다 불리하기 때문에 커진다.

## 2. 기본 진단 요약

### 2.1 기본 verl 경로는 비교적 단순하다

baseline에서는 trainer가 거의 바로 actor update로 들어간다.

- `verl/trainer/ppo/ray_trainer.py`
  - `update_actor` 직전 별도 replay selection 경로 없음
- `verl/workers/fsdp_workers.py`
  - `self.actor.update_policy(data=data)` 호출
- `verl/workers/actor/dp_actor.py`
  - 실제 PPO update 수행

즉 baseline의 hot path는 대략 다음과 같다.

1. rollout
2. old log prob
3. advantage
4. actor update

### 2.1.1 기본 verl 경로의 중요한 설계 원칙

기본 `verl` 코드베이스는 단순히 빠른 것이 아니라, hot path를 일정한 원칙으로 구성한다.
현재 확인된 핵심 원칙은 다음과 같다.

1. **batch-first, tensor-first**
   - 가능한 한 `DataProto` 전체 배치를 유지한 채 tensor 연산으로 처리한다.
   - hot path에서 query 단위 Python object 루프를 최소화한다.

2. **한 단계당 한 번의 큰 연산**
   - `old_log_prob`, `values`, `actor update`는 batch 전체에 대해 큰 단위 연산으로 수행한다.
   - 작은 호출을 많이 반복하는 방식보다 큰 호출을 적게 하는 방식에 가깝다.

3. **update 직전 길이 균형화**
   - trainer는 rollout 후 `balance_batch`를 통해 DP rank 사이의 valid token 불균형을 줄인다.
   - 이는 loss 의미를 바꾸기보다, 실행 효율과 안정성을 높이기 위한 재배치다.

4. **padding-aware, token-aware 실행**
   - `use_remove_padding=True` 경로를 통해 실제 non-pad token 기준으로 연산량을 줄인다.
   - 따라서 batch의 길이 분포와 순서가 곧 성능에 직접 영향을 준다.

이 네 가지는 단순한 구현 디테일이 아니라, 기본 `verl` 경로가 효율적이고 안정적으로 동작하는 이유라고 보는 것이 맞다.

### 2.2 m2 replay 경로는 actor update 전에 큰 파이프라인이 하나 더 붙는다

현재 replay 경로는 `_m2_build_actor_batch()` 안에서 다음을 수행한다.

1. on-policy batch를 query group으로 재구성
2. ingress 필터링
3. buffer 후보 정렬
4. fresh `new_log_probs` 재계산
5. M2 계산 및 tau 필터링
6. replay batch와 on-policy batch concat
7. 그 결과를 actor update에 전달

즉 replay는 actor update 자체를 바꾸기보다, actor update 이전에 매우 무거운 selection 파이프라인을 추가한다.

### 2.3 현재 가장 큰 병목은 selection 쪽이다

실제 late-stage 로그에서는 다음 현상이 반복된다.

- `replay_used = 0`
- `accepted_groups = 0` 또는 `1`
- `scanned_groups = 1024`
- `rejected_by_tau = 1023~1024`

그런데도 시간은 다음처럼 크게 나온다.

- `timing_s/m2_select_logprob_eval`: 수십 초
- `timing_s/m2_select_m2_eval`: 수십 초 미만이지만 여전히 큼
- `timing_s/m2_build_actor_batch`: actor update보다 더 큼

즉 replay를 실제로 거의 못 쓰는 step에서도 selection cost를 full로 지불한다.
이건 현재 구현이 utility와 비용이 분리되어 있음을 의미한다.

### 2.4 actor update가 더 비싸 보이는 이유는 단순히 “400토큰 차이”가 아니다

여기서 중요한 함정이 하나 있다.
현재 WandB의 `response_length/mean`은 실제 actor가 학습한 `actor_batch` 기준이 아니라, 원래 on-policy `batch` 기준이다.

따라서 다음이 가능하다.

- 로그상 `response_length/mean` 차이는 작게 보임
- 하지만 실제 actor는 replay가 섞인 더 길고 더 불균형한 batch를 학습함
- 결과적으로 `actor_update_policy_compute`는 크게 증가함

또한 Qwen3 rev9에서는 baseline과 replay의 actor update 대상 group 수 자체가 다르다.

- baseline: on-policy 256 groups
- replay: on-policy 256 + replay 128 = 최종 384 groups

즉 Qwen3에서 update 시간이 baseline보다 더 큰 이유는,

1. batch group 수가 더 많고
2. replay 길이 분포가 더 불리하고
3. static microbatch split이 token 균형을 맞추지 못하기 때문

이다.

### 2.5 기본 verl은 actor update 직전 batch balancing을 이미 수행한다

이 부분은 현재 replay 경로를 이해하는 데 매우 중요하다.

기본 trainer loop는 rollout 결과를 repeat/union한 뒤, `trainer.balance_batch=True`이면 `_balance_batch()`를 호출한다.
즉 baseline 경로는 actor update 전에 이미 길이 균형화가 적용된 batch를 사용한다.

하지만 replay 경로는 그 균형화된 on-policy batch를 바탕으로 query group을 만들고, 이후 `replay_batch`를 앞에 다시 붙여 새로운 `actor_batch`를 만든다.
즉 최종 actor update 입력은 **재차 balancing되지 않는다.**

따라서 현재 구조는 다음과 같이 해석해야 한다.

1. baseline:
   - rollout batch 생성
   - 길이 균형화
   - actor update

2. replay:
   - rollout batch 생성
   - 길이 균형화된 on-policy batch 획득
   - replay batch를 다시 앞에 붙여 새로운 actor batch 구성
   - **이 최종 actor batch는 균형화가 깨진 상태로 actor update**

즉 replay는 단순히 샘플 수가 늘어나는 것뿐 아니라, 기본 `verl`이 애써 만들어 둔 균형화 상태를 마지막에 한 번 더 무너뜨리는 구조다.
이 차이는 `actor_update_policy_compute` 차이를 설명하는 강한 코드 수준 근거다.

### 2.6 기본 `verl` 문서가 권장하는 방향과도 현재 replay 경로는 어긋난다

`docs/perf/perf_tuning.rst`와 `docs/perf/best_practices.rst`를 다시 보면, `verl`이 성능 최적화의 핵심으로 반복해서 강조하는 것은 아래 세 가지다.

1. `use_remove_padding=True`
2. 큰 batch / 큰 forward-only batch
3. 가능하면 `use_dynamic_bsz=True`

즉 공식 문서도 hot path 최적화의 핵심을 “길이 불균형 완화 + 큰 단위 호출”로 본다.

현재 replay 경로는 이 문서 철학과 비교해도 불리하다.

1. query group 단위 object 생성이 많다
2. selection에서 `compute_log_prob`를 여러 chunk로 반복 호출한다
3. 최종 actor batch는 baseline처럼 다시 균형화되지 않는다

따라서 replay 최적화 방향은 독자적인 감각적 heuristic보다, `verl` 문서가 이미 권장하는 방향을 replay 경로에 재적용하는 쪽이 더 정당하다.

## 3. 현재 병목을 구조적으로 나누면

### 3.1 Selection 병목

핵심 비용은 아래 세 가지다.

1. candidate scan
2. fresh `compute_log_prob` 재평가
3. M2 계산

특히 `compute_log_prob` 재평가가 가장 비싸다.
이 부분은 baseline에는 존재하지 않는 추가 forward pass 집합이다.

### 3.2 Actor update 병목

actor update의 core kernel은 baseline과 같다.
차이는 actor에 들어가는 입력이다.

현재 actor update는

- `use_remove_padding=True`이므로 실제 non-pad token 수에 민감하고
- `use_dynamic_bsz=False`이므로 sample count 기준으로 split되며
- replay와 on-policy를 concat한 뒤 contiguous split을 사용한다.

즉 길이 분포가 다른 replay 샘플이 섞이면 microbatch별 token workload 편차가 커진다.

추가로 중요한 점은, 현재 replay 경로의 actor update 병목은 기본 actor kernel이 비효율적이라서가 아니라는 점이다.

- replay가 실제로 `0`개 사용된 step에서는 baseline과 actor update 시간이 거의 비슷하다.
- replay가 실제로 들어간 step에서만 actor update 비용이 급증한다.

이는 actor core implementation 자체보다, **replay가 actor에게 공급하는 최종 batch의 크기와 배치 순서가 문제**임을 뜻한다.

### 3.3 `compute_log_prob` 경로는 actor update와 별도로 다시 봐야 한다

이 부분은 selection 최적화에서 중요하다.

`FSDP` 경로의 `compute_log_prob()`는 actor update와 별도로 아래 설정을 직접 읽는다.

1. `rollout.log_prob_micro_batch_size_per_gpu`
2. `rollout.log_prob_max_token_len_per_gpu`
3. `rollout.log_prob_use_dynamic_bsz`

즉 현재 환경이 FSDP actor worker 경로라면, 이론적으로는 **actor update의 dynamic batch를 켜지 않더라도, logprob 재평가 쪽만 별도로 더 효율적인 batching을 적용할 여지가 있다.**

이 점은 중요하다.

- 현재 replay selection의 가장 큰 병목은 fresh `compute_log_prob` 재평가다.
- 따라서 actor update semantics를 그대로 두더라도, logprob 재평가 경로만 개선하면 큰 효과를 얻을 수 있다.

다만 이 방향은 backend/worker 경로에 따라 결합 제약이 있을 수 있으므로, 실제 적용 전에는 현재 실험이 FSDP worker 경로를 타는지 다시 확인해야 한다.

## 4. 기본 verl 구현과 현재 replay 구현의 철학 비교

### 4.1 기본 verl 구현 방안

기본 `verl`은 다음 성격을 가진다.

1. 효율적
   - 큰 batch 연산 위주
   - remove-padding 활용
   - trainer 단계에서 길이 균형화 수행

2. 안정적
   - hot path에서 debug/분석용 부가 계산이 제한적
   - core path가 비교적 짧다

3. 강력함
   - 동일한 batch-first 원칙을 rollout, old_log_prob, values, actor update 전반에 일관되게 적용한다

### 4.2 현재 replay 구현 방안

현재 `m2_replay`는 다음 성격을 가진다.

1. Python object 중심
   - query group 단위 슬라이싱과 object 생성이 많다

2. 반복 호출 중심
   - `compute_log_prob`를 선택 후보에 대해 여러 번 호출한다

3. debug/analysis 기능이 training path에 잔존
   - full score dump
   - runtime ZVP 통계 재계산

4. 기본 verl의 balancing 철학을 최종 actor batch까지 유지하지 못함
   - on-policy batch는 balanced
   - replay를 붙인 최종 actor batch는 unbalanced

즉 현재 replay 구현은 기본 `verl`의 “batch-first, balanced, hot-path-minimal” 철학과 충돌하는 부분이 많다.

## 5. 개선 원칙 재정의

개선의 방향은 단순히 “더 줄이자”가 아니라 아래처럼 정리되어야 한다.

1. **기본 `verl`의 hot-path 철학을 replay 경로에 복원한다.**
2. **선택 결과를 바꾸지 않는 선에서, 더 큰 단위 연산과 정적 재배치를 사용한다.**
3. **debug/analysis 계산을 training default 경로에서 분리한다.**
4. **최종 actor batch도 baseline과 동일하게 길이 균형화 원칙을 적용한다.**

### 5.1 이 원칙을 코드 경로에 직접 대응시키면

현재까지 실행 경로를 따라가며 확인한 결과, 아래 대응이 가장 엄밀하다.

1. baseline `verl`의 강점은 단순히 actor kernel이 빠르다는 데 있지 않다.
   - trainer가 rollout 이후 `_balance_batch()`를 적용한 뒤 actor update에 들어간다는 점이 중요하다.
   - 즉 baseline은 **실행 직전 입력 배치를 정리해 둔 뒤** 효율 좋은 actor kernel을 호출한다.

2. replay 경로의 가장 큰 구조적 문제는, 이 정리된 상태를 마지막에 다시 무너뜨린다는 점이다.
   - `on-policy batch`는 이미 balanced 상태다.
   - 하지만 replay 경로는 이후 `replay_batch + onpolicy_batch`를 다시 concat하여 최종 `actor_batch`를 만든다.
   - 따라서 최종 actor 입력은 baseline이 확보한 균형화 성질을 더 이상 보장받지 못한다.

3. selection 경로의 가장 큰 구조적 문제는, baseline이 지향하는 “큰 batch를 적게 호출” 철학과 반대로 흘러간다는 점이다.
   - 현재 replay selection은 batched path를 쓰더라도 chunk 반복 호출이 많다.
   - 즉 baseline이 추구하는 `한 단계당 한 번의 큰 연산`과 달리, replay는 `많은 작은 호출`과 `반복 split/concat`을 수행한다.

정리하면, 현재 replay 최적화의 핵심은 “새 heuristic을 더 얹는 것”이 아니라,
**baseline `verl`이 이미 잘하고 있는 두 가지 원칙을 replay 경로에도 복원하는 것**이다.

1. 실행 직전 batch balancing
2. 큰 호출 위주의 batch-first 실행

## 6. 최적화 원칙

이번 문서의 전제는 다음과 같다.

1. 알고리즘 의미는 최대한 유지한다.
2. `actor.use_dynamic_bsz=true`는 사용할 수 없다.
3. 따라서 selection path의 중복/불필요 계산 제거와, actor batch의 정적 재배치를 중심으로 본다.

## 7. 우선순위별 최적화 방향

### P0. training default에서 debug 성격 계산 제거

#### P0-1. `dump_full_scores` 기본 비활성화

현재 full score dump는 실제 training hot path에 들어와 있다.
이 기능은 분석에는 유용하지만, 기본 training 경로에서는 제거하는 것이 맞다.

기대 효과:
- 작은 상수항 감소
- 불필요한 payload 생성 및 파일 I/O 제거

비고:
- 병목의 주범은 아니지만, 기본 training 경로에서 켜둘 이유가 거의 없다.

#### P0-2. accepted sample의 runtime ZVP 재계산을 debug 모드로 제한

현재 selection은 `m2 <= tau` 판정 후 accepted group에 대해 runtime ZVP 통계를 다시 계산한다.
이는 selection/학습의 본질 조건이 아니라 debug 정보에 가깝다.

기대 효과:
- accepted path에서 추가 GPU/CPU 계산 제거
- late-stage에는 효과가 제한적이지만 early/mid-stage에서는 분명한 절감

### P1. selection path에서 “같은 계산을 늦게 또 하는 구조” 제거

#### P1-1. ingress 시점 ZVP 선계산으로 buffer-wide backfill 제거

현재 일부 start mode에서는 ingress 시점에 ZVP를 충분히 채우지 않고, selection 직전에 buffer 전체를 다시 보며 `only_missing=True` backfill을 수행한다.
이건 동일한 통계를 더 비싼 위치에서 계산하는 구조다.

기대 효과:
- selection 직전 buffer-wide pass 제거
- logic 변화 없이 계산 시점만 앞으로 이동

주의:
- score semantics는 유지하되, “언제 계산하느냐”만 바뀌는 형태여야 한다.

### P2. `compute_log_prob` 호출 구조 최적화

#### P2-1. superchunk 방식으로 `compute_log_prob` 호출 횟수 축소

현재 selection은 chunk 단위로 `DataProto.concat -> compute_log_prob -> split`을 반복한다.
이 반복 호출 횟수가 많다.

개선 방향:

- 정렬된 candidate를 더 큰 superchunk로 묶는다.
- 한 번에 `compute_log_prob`를 호출한다.
- 결과를 group별로 다시 split해서 순차적으로 tau 판정에 사용한다.

기대 효과:
- `compute_log_prob` 호출 횟수 감소
- `DataProto.concat` 횟수 감소
- GPU launch/dispatch overhead 감소

중요:
- candidate 순서와 tau 판정 순서는 유지한다.
- selection 결과를 바꾸지 않도록 구현해야 한다.

보강:
- 현재 `compute_log_prob` 경로 자체가 `micro_batch_size`, `max_token_len`, `use_dynamic_bsz`를 meta_info로 읽어 batch를 나누는 구조이므로,
  이 방향은 “selection semantics를 바꾸지 않고 호출 구조만 baseline-friendly하게 바꾸는” 최적화와 잘 맞는다.

#### P2-1-b. superchunk 내부 길이 재배치 + 결과 원순서 복원

위 superchunking을 더 발전시키면, chunk 내부의 group들을 길이 기준으로 재배치한 뒤 `compute_log_prob`를 호출하고, 결과를 다시 원래 group 순서로 복원할 수 있다.

이 방식의 장점은 다음과 같다.

1. selection semantics 유지
   - tau 판정 순서와 최종 선택 순서는 그대로 유지 가능
2. 기본 `verl`의 balancing 철학 재사용
   - 작은 길이/큰 길이를 섞거나 정렬하여 microbatch workload 편차를 줄일 수 있음
3. `compute_log_prob` hot path 자체를 더 효율적인 입력 조건에서 실행 가능

즉 이 방향은 “selection logic은 유지하되, `compute_log_prob` 입력 배치를 더 baseline 친화적으로 만든다”는 의미에서 매우 유력하다.

추가로, 이 재배치는 새 heuristic을 처음부터 만드는 것보다,
기존 `verl.utils.seqlen_balancing`의 workload 추정(`calculate_workload`)과 partition helper(`get_seqlen_balanced_partitions`)를 재사용하는 쪽이 더 바람직하다.

이유는 다음과 같다.

1. baseline trainer의 `_balance_batch()`가 이미 그 helper를 사용한다
2. `rearrange_micro_batches()`도 같은 helper 계열을 사용해 길이 workload를 정렬/완화한다
3. 즉 replay용 재배치도 기존 `verl`이 검증한 workload proxy를 공유하는 편이 더 일관적이다

#### P2-2. tail 구간에서 adaptive chunk shrink

현재는 남은 slot 수가 거의 없더라도 큰 chunk를 통째로 평가할 수 있다.
이를 아래처럼 바꾼다.

- 초반: 큰 chunk
- 남은 slot이 적어질수록: 작은 chunk

기대 효과:
- 필요 없는 마지막 chunk 대량 평가 감소
- exact logic에 거의 영향을 주지 않는 상수항 최적화

### P3. actor update 입력 batch를 정적으로 재배치

#### P3-1. actor batch length bucketing / reordering

`dynamic_bsz`를 못 쓰는 조건에서 가장 중요한 actor-side 최적화다.

현재는 `replay_batch`와 `onpolicy_batch`를 concat한 뒤 그대로 contiguous split한다.
이 구조는 긴 replay group이 특정 microbatch에 몰리게 만들 수 있다.

개선 방향:

- actor update 직전, 각 sample 또는 group의 길이를 계산한다.
- 길이 기준으로 reorder한다.
- 이후 기존 `split()` 경로를 그대로 사용한다.

가능한 전략:

1. 오름차순 정렬
2. 내림차순 정렬
3. 짧은 것/긴 것을 교차 배치하는 zig-zag reorder

권장:
- 단순 정렬보다 microbatch total token 합이 비슷해지도록 하는 정적 heuristic이 더 좋다.

권장 보강:
- 가능하면 ad-hoc sort 대신, baseline `_balance_batch()`와 동일한 workload proxy 및 partition helper를 재사용한다.
- 즉 “replay 전용 정렬 규칙”을 새로 만드는 것보다, baseline `verl`의 balancing 방식을 최종 actor batch에 다시 적용하는 쪽이 더 정당하다.

기대 효과:
- microbatch별 token workload 편차 감소
- `actor_update_forward_micro`, `actor_update_backward_micro` 감소

중요:
- loss, replay 여부, tau, selection logic은 안 바뀐다.
- 단지 학습 입력의 순서만 바뀐다.

#### P3-2. 기존 `_balance_batch()`를 actor batch에도 다시 적용

이건 현재까지 확인된 코드베이스 비교에서 가장 중요한 개선안 중 하나다.

현재 baseline은 actor update 직전에 `_balance_batch()`가 적용되지만, replay는 최종 actor batch가 그 이후에 새로 만들어지므로 이 이점을 잃는다.

따라서 가장 자연스러운 개선은 다음이다.

1. on-policy batch는 기존대로 balance
2. replay batch를 붙여 최종 actor batch 생성
3. 최종 actor batch에 대해 다시 `_balance_batch()` 또는 동일한 balancing helper를 적용

이 방식의 장점:

1. 완전히 새로운 heuristic을 도입하지 않는다
2. 기본 `verl`이 이미 채택한 안정적 balancing 코드를 재사용한다
3. replay 경로를 baseline 철학과 다시 맞춘다

현 시점에서 actor-side 최적화 중 가장 강한 후보는 `P3-1` 일반론보다, 오히려 이 `P3-2`의 형태로 정식 helper를 재사용하는 것이다.

### P4. metric 보강으로 실제 병목 가시화

현재 `response_length/mean`은 actor가 실제 학습한 batch를 반영하지 못한다.
따라서 최적화 전후 비교를 위해 아래 metric이 필요하다.

1. `actor_batch/response_length_mean`
2. `actor_batch/token_count`
3. `actor_batch/replay_token_count`
4. `actor_batch/onpolicy_token_count`
5. `actor_batch/microbatch_token_max_min_diff`

기대 효과:
- 현재 “400 토큰 정도 차이인데 왜 update가 1.5~1.7배 비싸지?”라는 혼란 해소
- 최적화 효과를 실제 actor batch 기준으로 판단 가능

## 8. 현 시점에서 추천하는 실제 적용 순서

로직 변경을 최소화하면서도 공격적으로 가려면 다음 순서가 적절하다.

1. `dump_full_scores` 기본 off
2. runtime ZVP debug-only화
3. ingress 시 ZVP 선계산 + selection backfill 제거
4. selection superchunking
5. selection superchunk 내부 길이 재배치 + 결과 복원
6. tail adaptive chunk shrink
7. 최종 actor batch에 baseline-style balancing 재적용
8. actor batch 기준 metric 추가

### 8.1 구현 우선순위를 더 엄밀히 정리하면

위 목록은 단순 나열이 아니라, 아래 세 층으로 이해하는 것이 좋다.

#### Layer A. baseline `verl` 철학 복원

이 층은 “현재 replay가 baseline 대비 왜 불리한가?”에 직접 대응한다.

1. 최종 actor batch에 baseline-style balancing 재적용
2. selection superchunk 내부 길이 재배치 + 결과 복원

이 두 개는 각각

- actor update 입력을 baseline스럽게 되돌리는 조치
- selection의 `compute_log_prob` 입력을 baseline스럽게 되돌리는 조치

에 해당한다.

즉 현재 가장 중요한 구조적 개선은 사실상 이 Layer A다.

여기서 더 정확히 말하면, Layer A는 두 개의 서로 다른 hot path를 겨냥한다.

1. actor update hot path
   - 최종 actor batch rebalance
2. selection logprob hot path
   - superchunk 내부 길이 재배치

즉 둘 다 “길이 workload를 정리한다”는 공통 철학을 갖지만, 실제로는 서로 다른 병목에 대응한다.

#### Layer B. 불필요한 계산 제거

이 층은 selection 의미를 바꾸지 않으면서도 hot path에서 낭비를 줄인다.

1. runtime ZVP debug-only화
2. ingress 시 ZVP 선계산 + selection backfill 제거
3. `dump_full_scores` 기본 off

이 Layer B는 “정답을 바꾸지 않는 비용 절감”에 가깝다.

#### Layer C. 호출 구조 최적화

이 층은 asymptotic을 바꾸지는 못하지만, 상수항을 크게 줄인다.

1. selection superchunking
2. tail adaptive chunk shrink

이 Layer C는 특히 `m2_select_logprob_eval`이 큰 구간에서 중요하다.

### 8.2 내가 실제 구현 순서를 다시 잡는다면

실제 구현은 아래 순서가 가장 타당하다.

1. `dump_full_scores` 기본 off
2. runtime ZVP debug-only화
3. ingress ZVP eager compute
4. selection superchunking
5. selection superchunk 내부 길이 재배치 + 결과 복원
6. 최종 actor batch re-balance
7. actor batch metric 보강
8. tail adaptive chunk shrink

이 순서를 추천하는 이유는 다음과 같다.

- 1~3은 리스크가 매우 낮다.
- 4~6이 실제 체감 성능을 크게 바꿀 가능성이 높다.
- 7은 이후 판단의 정확도를 높여 준다.
- 8은 마지막 상수항 정리에 가깝다.

### 8.3 현재 조건에서 추가로 검토할 수 있는 옵션

`actor.use_dynamic_bsz`를 켤 수 없다면, 다음 옵션은 별도로 검토할 가치가 있다.

1. `rollout.log_prob_use_dynamic_bsz`
2. `rollout.log_prob_max_token_len_per_gpu`
3. `ref.log_prob_use_dynamic_bsz`
4. `ref.log_prob_max_token_len_per_gpu`

이 옵션들은 actor update가 아니라 `compute_log_prob` / `compute_ref_log_prob` forward-only 경로에 영향을 준다.
따라서 현재 replay selection이 fresh logprob 재평가에서 크게 병목되는 상황에서는, actor training semantics를 건드리지 않고도 selection 쪽 throughput 개선 수단이 될 수 있다.

다만 이것은 현재 backend 경로와 메모리 제약을 확인한 뒤 제한적으로 검토해야 하며, 이번 문서에서는 우선순위상 “후보 옵션”으로만 둔다.

## 9. 기대 효과에 대한 현실적 판단

### 기대 효과가 큰 부분

- final actor batch balancing 재적용
- selection superchunk 내부 길이 재배치
- selection superchunking
- ingress ZVP 선계산

### 기대 효과가 중간인 부분

- runtime ZVP debug-only화
- tail adaptive chunk shrink

### 기대 효과가 작지만 해야 하는 부분

- full score dump off
- metric 보강

## 10. 남는 한계

아무리 위 최적화를 해도, 현재 selection logic을 정확히 유지하면 late-stage full scan 성질은 남는다.

즉,

- acceptance가 거의 0인 구간
- `replay_used`가 거의 0인 구간

에서는 결국 buffer 많은 부분을 보게 된다.

이건 상수항 최적화로는 줄일 수 있지만, asymptotic 구조를 바꾸지는 못한다.
따라서 아주 큰 폭의 감소를 원하면 결국 아래 중 하나가 필요하다.

1. cheap prefilter 추가
2. scan budget 추가
3. approximate tau guard 추가
4. replay target을 adaptive하게 줄이는 정책

하지만 위 네 가지는 모두 현재 문서의 범위인 “로직 최소 변경”을 넘는 방향이다.

### 10.1 따라서 장기 로드맵은 두 단계로 나뉘어야 한다

#### Stage 1. 의미 변화 최소 최적화

목표:
- baseline `verl`의 강점을 replay 경로에 최대한 이식
- selection semantics 유지
- actor update semantics 유지

성공 기준:
- `m2_build_actor_batch` 유의미한 감소
- `actor_update_policy_compute`의 replay overhead 완화
- 선택 결과 분포가 크게 변하지 않음

#### Stage 2. 그 다음에야 selection policy 자체를 다시 논의

만약 Stage 1 이후에도 late-stage 비용이 너무 크다면, 그때는 비로소 아래를 논의할 수 있다.

1. scan budget
2. cheap prefilter
3. approximate tau guard
4. adaptive replay target

즉 현재 시점에서 해야 할 일은 “selection 로직을 더 공격적으로 바꾸는 것”이 아니라,
**현재 selection 로직이 baseline 수준의 시스템 설계를 최대한 누리도록 만드는 것**이다.

## 11. 최종 정리

현재 m2 replay 구현은 두 가지 이유로 baseline verl 대비 느리다.

1. selection/control-flow가 본질적으로 매우 무겁다.
2. actor는 replay 때문에 더 크고, 더 길고, 더 불균형한 batch를 받는다.

이 중 두 번째는 기본 verl actor kernel의 문제가 아니다.
기본 verl의 update 코드는 여전히 효율적이다.
문제는 replay가 그 엔진에 더 나쁜 입력을 공급한다는 점이다.

따라서 현 시점에서 가장 합리적인 방향은 다음이다.

- selection path의 중복/디버그 계산 제거
- `compute_log_prob` 호출 횟수 감소 및 입력 재배치
- `dynamic_bsz` 없이도 가능한 baseline-style balancing을 최종 actor batch까지 복원

이 세 축이 현재 최적화의 중심이 되어야 한다.

## 12. 변경 정도 대비 효율 지표

여기서는 최적화 후보를 단순 아이디어 목록이 아니라, **변경 리스크와 기대 효율을 같은 축 위에서 비교**하기 위한 지표로 정리한다.

### 12.1 지표 정의

#### A. Logic Perturbation Score (LPS)

로직을 얼마나 건드리는지를 0~5로 둔다.

- 0: 관측/로그만 추가
- 1: debug/분석 경로 제거 또는 기본값 변경
- 2: 계산 시점 이동, 호출 구조 변경, 의미 보존형 batching 변경
- 3: 기존 helper 재사용 기반의 배치 재배치, mini-batch 구성 변화 유발
- 4: scan/selection 정책 변화
- 5: 알고리즘 의미 자체 변화

#### B. Efficiency Gain Score (EGS)

효율 개선 기대치를 0~5로 둔다.

- 0: 거의 없음
- 1: 소규모 상수항 감소
- 2: 눈에 띄는 상수항 감소
- 3: 특정 병목 구간에서 유의미한 감소
- 4: 전체 step time 체감 감소 가능
- 5: 핵심 병목 구조를 실질적으로 완화

#### C. Confidence Score (CS)

현재 코드/문서/로그 근거의 강도를 0~5로 둔다.

- 0: 추정
- 1: 약한 정성 추정
- 2: 일부 로그 근거
- 3: 코드 근거 또는 반복 로그 근거
- 4: 코드 + 로그 모두 강함
- 5: 코드 + 로그 + 문서 철학까지 일치

#### D. Recommendation Index (RI)

최종 추천 지표는 아래처럼 둔다.

`RI = (EGS × CS) / (LPS + 1)`

의미:

- 효율 기대가 크고
- 근거가 강하며
- 로직 변경이 적을수록

우선순위가 높아진다.

이 지표는 절대적 수학이라기보다, 현재 단계의 의사결정 기준으로 사용한다.

### 12.2 현재 후보별 점수

| 후보 | LPS | EGS | CS | RI | 판단 |
|---|---:|---:|---:|---:|---|
| `actor_batch` 기준 metric 추가 | 0 | 2 | 5 | 10.0 | 반드시 해야 함 |
| `dump_full_scores` 기본 off | 1 | 1 | 4 | 2.0 | 바로 적용 |
| runtime ZVP debug-only화 | 1 | 2 | 4 | 4.0 | 바로 적용 |
| ingress ZVP eager compute | 2 | 2 | 4 | 2.67 | 낮은 리스크로 유효 |
| selection superchunking | 2 | 4 | 4 | 5.33 | 매우 유력 |
| selection tail adaptive shrink | 2 | 2 | 3 | 2.0 | 후순위 보조 |
| 최종 `actor_batch` rebalance | 3 | 5 | 5 | 6.25 | 핵심 우선순위 |
| selection superchunk 내부 길이 재배치 + 결과 복원 | 3 | 4 | 4 | 4.0 | 핵심 보강안 |
| `rollout/ref.log_prob_use_dynamic_bsz` 별도 검토 | 3 | 3 | 2 | 1.5 | 환경 검증 후 후보 |
| cheap prefilter / scan budget / approximate tau | 4~5 | 4~5 | 2~3 | 낮음 | Stage 2 이후 |

### 12.3 이 표에서 읽어야 할 핵심

현재 가장 추천도가 높은 것은 두 부류다.

1. **관측 정비**
   - `actor_batch` 기준 metric 추가

2. **baseline 철학 복원**
   - 최종 `actor_batch` rebalance
   - selection superchunking
   - selection 내부 길이 재배치 + 복원

즉 현 시점의 방향은 “selection 정책을 바꾸는 것”이 아니라,
**baseline `verl`이 원래 확보하던 실행 효율을 replay 경로에도 다시 적용하는 것**이어야 한다.

## 13. 현재 코드 기준 리스크 재판정

### 13.1 최종 actor batch 재배치의 실제 리스크는 생각보다 낮다

표면적으로는 `actor.shuffle=true may weaken replay-first ordering semantics` 경고가 있어서 순서 변경이 위험해 보일 수 있다.
하지만 실제 loss/metric 경로를 다시 보면, replay와 on-policy의 구분은 순서가 아니라 `m2_replay_source` flag로 처리된다.

- source flag 주입: `m2_replay_adapter.py`
- loss split metric 사용: `verl/workers/utils/losses.py`, `verl/trainer/ppo/core_algos.py`

즉 replay/on-policy 분리는 **order-invariant** 하다.

현재 순서가 영향을 주는 부분은 실제로는 다음이다.

1. contiguous `split()` 기반 mini/micro-batching
2. replay-first 디버그/해석 관례

따라서 최종 actor batch rebalance는 “알고리즘 의미를 바꾸는 변경”이라기보다,
**mini-batch composition을 baseline스럽게 다시 정렬하는 변경**으로 보는 편이 정확하다.

이 때문에 본 문서에서는 LPS를 3으로 두었지만, 의미 변화 위험도는 4~5 수준이 아니라는 점을 명확히 해 둔다.

### 13.2 selection 내부 길이 재배치도 의미 변화는 제한적이다

selection 내부 길이 재배치는 다음 조건만 지키면 의미 변화가 매우 작다.

1. `compute_log_prob` 입력 순서만 임시로 바꿀 것
2. 반환 결과는 원래 group 순서로 복원할 것
3. tau 판정 순서와 최종 selection 순서는 유지할 것

즉 이것은 `compute_log_prob`를 baseline-friendly한 입력으로 태우기 위한 **실행 최적화**이지,
selection policy를 바꾸는 변경이 아니다.

## 14. 최종 추천안

### 14.1 Stage 1에서 실제로 가져갈 패키지

현재 조건에서 가장 균형이 좋은 패키지는 아래다.

1. `actor_batch` 기준 metric 추가
2. `dump_full_scores` 기본 off
3. runtime ZVP debug-only화
4. ingress ZVP eager compute
5. selection superchunking
6. selection superchunk 내부 길이 재배치 + 결과 복원
7. 최종 actor batch rebalance

이 패키지의 의미는 다음과 같다.

- selection 정책은 유지
- tau 정책은 유지
- replay 비율 정책은 유지
- actor loss 정책은 유지
- 단지 실행 경로를 baseline `verl` 철학에 더 가깝게 되돌림

### 14.2 지금은 보류해야 하는 것

아래는 지금 단계에서 바로 들어가면 안 된다.

1. cheap prefilter
2. scan budget
3. approximate tau guard
4. replay target adaptive control

이 항목들은 분명히 속도를 더 줄일 수 있지만,
현재 단계에서는 “selection semantics가 바뀌어서 빨라진 것인지”와
“시스템 구현이 baseline스럽게 정리돼서 빨라진 것인지”가 섞이게 된다.

따라서 지금은 Stage 1의 정리형 최적화를 먼저 끝낸 뒤, 그 다음 단계에서만 다시 논의하는 것이 맞다.

### 14.3 변경 예산별 추천 패키지

같은 RI 표를 보더라도, 실제 구현에서는 "우리가 허용하는 로직 변경 예산"을 먼저 정하는 편이 낫다.
이를 위해 아래처럼 세 가지 패키지로 다시 나눈다.

#### Package S. 보수형 (Logic Budget 최소)

목표:
- selection/actor 의미를 거의 건드리지 않고, hot path의 불필요 비용만 제거

포함:
1. `actor_batch` 기준 metric 추가
2. `dump_full_scores` 기본 off
3. runtime ZVP debug-only화
4. ingress ZVP eager compute
5. selection tail adaptive shrink

특징:
- selection 결과 분포가 바뀔 가능성이 매우 낮다
- 구현 리스크가 가장 작다
- 다만 가장 큰 체감 병목인 actor-side 불균형과 `compute_log_prob` 호출 구조는 충분히 해결하지 못한다

권장 상황:
- 먼저 안전하게 1차 정리를 하고 싶을 때
- 현재 실험을 계속 돌리면서 코드 리스크를 최소화해야 할 때

#### Package B. 균형형 (현재 가장 추천)

목표:
- selection semantics는 유지하면서, baseline `verl`의 실행 철학을 replay 경로에 복원

포함:
1. `actor_batch` 기준 metric 추가
2. `dump_full_scores` 기본 off
3. runtime ZVP debug-only화
4. ingress ZVP eager compute
5. selection superchunking
6. selection superchunk 내부 길이 재배치 + 결과 원순서 복원
7. 최종 `actor_batch` rebalance

특징:
- selection policy, tau policy, replay 비율 정책은 유지한다
- 하지만 실제 실행 batch는 baseline 친화적으로 재정렬한다
- 지금까지의 코드/문서/로그 근거를 종합했을 때, 가장 정당하고 효과 대비 리스크가 좋은 패키지다

권장 상황:
- 지금처럼 "로직 변경은 최소화하되, 실제 체감 속도도 확실히 낮추고 싶다"는 경우
- baseline `verl`의 장점을 replay 경로에 최대한 이식하고 싶은 경우

#### Package A. 공격형 (의미 보존은 유지하되 구현 변경 폭 큼)

목표:
- Stage 1 의미 보존 최적화를 가능한 한 크게 적용

포함:
1. Package B 전부
2. `rollout/ref.log_prob_use_dynamic_bsz` 별도 검토
3. selection superchunk 크기/마이크로배치 자동 조정
4. actor batch rebalance를 helper 재사용 수준이 아니라 전용 fast path로 정리

특징:
- 여전히 selection 의미를 직접 바꾸지는 않지만, 실행 계층 수정 폭은 커진다
- backend/메모리 조건에 따라 예상 밖 상호작용 가능성이 있다
- 지금 단계에서는 추천 1순위는 아니다

권장 상황:
- Package B 적용 후에도 selection/actor 양쪽이 여전히 크게 병목일 때
- 실험 환경을 더 강하게 통제할 수 있을 때

### 14.4 로직 변경 대비 효율을 기준으로 한 최종 추천

현재 기준에서 내가 실제로 추천하는 것은 아래 두 줄뿐이다.

1. **바로 구현할 1순위:** Package B
2. **정말 조심스럽게 가야 한다면:** Package S

반대로, Package A는 지금 바로 갈 이유가 충분하지 않다.
왜냐하면 현재 가장 먼저 확인해야 할 것은

- baseline `verl`의 balancing 철학을 replay 경로에 복원했을 때
- selection semantics를 안 바꿔도 병목이 얼마나 내려가는지

이기 때문이다.

즉 지금 단계에서의 최적화 방향은
"새 아이디어를 넣는 것"이 아니라
**"현재 아이디어가 baseline `verl` 수준의 시스템 설계를 누리게 만드는 것"**이다.

## 15. 최종 결론

이번 재검토를 통해, 현재 병목에 대한 해석은 아래처럼 정리하는 것이 가장 엄밀하다.

1. 기본 `verl`은 이미
   - 큰 batch 위주,
   - 길이 workload 균형화,
   - remove-padding 활용,
   - hot-path 최소화
   라는 원칙으로 상당히 잘 짜여 있다.

2. 현재 `m2_replay`는 알고리즘이 나빠서만 느린 것이 아니라,
   - selection에서 baseline 철학과 반대로 작은 호출을 많이 만들고,
   - actor update 직전에는 baseline이 확보한 균형화 상태를 다시 깨뜨리기 때문에
   시스템적으로 불리하다.

3. 따라서 현재 가장 정당한 개선 방향은
   - selection 의미를 바꾸는 것보다,
   - baseline `verl`의 실행 철학을 replay 경로에 다시 적용하는 것이다.

4. 이 기준에서 최적의 1차 패키지는 Package B이며,
   이는 "로직 변경 최소"와 "체감 성능 개선"의 비율이 가장 좋다.

정리하면,
현재 가장 좋은 개선안은 **더 새로운 replay 규칙**이 아니라,
**현재 replay 규칙을 baseline `verl`처럼 실행되게 만드는 것**이다.

## 16. Package B와 384 baseline 비교 시 성립해야 하는 사실

이 절은 "256 on-policy + 128 replay"가 "384 on-policy baseline"보다 실제로 빨라질 수 있는지에 대한 판단 기준을 정리한다.

### 16.1 먼저, per-step 비교의 구조는 이렇게 다르다

`384 baseline`은 한 step에서 대략 아래를 모두 384 기준으로 수행한다.

1. rollout / generation
2. reward
3. old log prob
4. advantage
5. actor update (384 groups)

반면 `256 + 128 replay`는 다르다.

1. rollout / generation은 **256 on-policy**만 수행
2. reward도 **256 on-policy**만 수행
3. old log prob도 **256 on-policy**만 수행
4. advantage도 **256 on-policy**만 수행
5. 대신 replay selection/build 비용이 추가됨
6. actor update는 **256 on-policy + 128 replay = 최종 384 groups**로 수행

즉 replay 경로는 actor update 기준으로는 384와 비슷한 계산을 하지만,
rollout/old-log-prob/adv 기준으로는 256만 수행한다.

따라서 이론적으로는 replay가 384 baseline보다 per-step 기준으로 빨라질 여지가 있다.

### 16.2 replay가 baseline-b384보다 빨라지기 위한 불등식

기호를 아래처럼 두자.

- `G384`: baseline-b384의 generation/rollout 비용
- `O384`: baseline-b384의 old_log_prob + adv + reward 비용
- `A384`: baseline-b384의 actor update 비용
- `G256`, `O256`: replay 경로에서 on-policy 256만 처리하는 rollout/old-log-prob 계열 비용
- `S`: replay selection/build 비용 (`m2_build_actor_batch`의 핵심)
- `A_mix`: replay 경로의 actor update 비용 (최종 384 groups, but replay mixed)

그러면 아주 단순화하면,

- baseline-b384: `T_base ≈ G384 + O384 + A384`
- replay: `T_replay ≈ G256 + O256 + S + A_mix`

replay가 baseline-b384보다 빨라지려면,

`S + (A_mix - A384) < (G384 - G256) + (O384 - O256)`

가 되어야 한다.

즉,

- replay selection/build로 추가된 비용
- replay mixed actor batch 때문에 더 비싸진 actor update 비용

의 합이,

- 384에서 256으로 줄어든 rollout/old-log-prob 계열 비용 절감분

보다 작아야 한다.

### 16.3 현재 로그가 말하는 바

현재 Qwen2.5 main run과 baseline-b384 로그를 보면, 이 조건이 **현재 구현에서는 대체로 만족되지 않는다.**

이유는 다음과 같다.

1. rollout/old-log-prob 절감은 분명히 존재한다
   - baseline-b384는 384개를 rollout한다
   - replay는 256개만 rollout한다
   - 따라서 generation과 old_log_prob는 실제로 더 작다

2. 하지만 현재는 `S`가 너무 크다
   - 특히 late-stage에는 `replay_used=0`이어도 `m2_build_actor_batch`가 100초 이상 드는 구간이 있다
   - 즉 utility가 거의 없는데 selection 비용을 거의 full로 지불한다

3. 또한 `A_mix`도 현재는 너무 크다
   - 최종 actor batch가 baseline이 보장하던 balancing을 잃는다
   - replay sample 길이 tail도 섞이므로 `A384`보다 더 커진다

정리하면, 현재 구현에서는

- 좌변: `S + (A_mix - A384)` 가 크고
- 우변: `384 -> 256`로 줄어든 rollout 절감분이 이를 충분히 못 덮는다

그래서 replay가 baseline-b384보다 per-step 기준으로 느려지는 것이다.

### 16.4 그렇다면 Package B는 이 불등식을 어떻게 바꾸나

Package B는 위 불등식의 **좌변만 줄이는 패키지**다.

1. `S`를 줄임
   - debug 성격 계산 제거
   - ingress eager ZVP
   - selection superchunking
   - selection 내부 길이 재배치

2. `A_mix - A384`를 줄임
   - 최종 actor batch rebalance를 통해
   - baseline 대비 replay actor update의 추가 불이익을 낮춤

중요한 점은,
Package B는 우변을 키우는 패키지가 아니다.
즉 rollout 자체를 더 줄이거나, replay target을 줄이거나, tau 정책을 바꾸지 않는다.

따라서 Package B의 본질은

- replay의 의미는 그대로 두고
- replay가 baseline-b384보다 느려지는 원인을 만드는 추가 비용을 줄여서
- 위 불등식이 성립할 가능성을 높이는 것

이라고 보는 것이 가장 정확하다.

### 16.5 그래서 현실적으로 가능한 목표는 어디까지인가

현재 단계에서 가장 현실적인 목표는 세 수준으로 나뉜다.

1. **1차 목표**
   - replay가 baseline-b384보다 "덜 느려지게" 만들기
   - 즉 replay overhead를 큰 폭으로 줄이는 것

2. **2차 목표**
   - 특정 구간에서는 replay step time이 baseline-b384와 비슷하거나 더 빨라지게 만들기
   - 특히 selection이 잘 통과되고, replay actor batch가 충분히 잘 정렬되는 mid-stage에서 가능성이 있다

3. **3차 목표**
   - 거의 모든 구간에서 replay가 baseline-b384보다 확실히 빠르기
   - 이건 Package B만으로는 장담하기 어렵다
   - 이 수준까지 가려면 결국 Stage 2 계열의 selection policy 변경이나 더 강한 시스템 수준 최적화가 필요할 가능성이 높다

즉 Package B는 매우 가치 있지만,
그 자체만으로 "반드시 baseline-b384보다 더 빨라진다"를 보장하는 패키지는 아니다.

다만 현재 구현처럼 replay가 baseline-b384보다 과하게 느린 상태를,
**비교 가능한 수준까지 끌어내리는 데는 가장 정당한 1차 패키지**라고 볼 수 있다.
