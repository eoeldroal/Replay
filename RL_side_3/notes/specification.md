# **M₂-Gated Replay for GRPO (KL-free)**

### **안정성 우선 오프폴리시 리플레이 선택/학습 명세서**

## **-1. 현재 실행 파라미터 반영 메모 (2026-02-26)**

- 본 문서의 개념 명세와 별개로, 현재 실험 러닝 파라미터는 아래와 같다.
  1. on-policy 그룹 수: 256/step
  2. replay target: 256/step (1:1)
  3. buffer 크기: 1024 QueryGroup
  4. `tau=0.04`
  5. 관찰 동작: step1 replay 0, step2+ replay 256 full-use
  6. 관찰 시간: `timing_s/update_actor`가 baseline 대비 증가(대략 44.6s -> 103~112s)
- 따라서 “B=4, R=4” 예시는 동작 설명용 소형 예시로만 해석해야 하며, 실제 러닝은 위 대형 설정을 따른다.

## **0. 한 줄 요약**

**오프폴리시 리플레이를 “항상 쓰는” 것이 아니라, 정책 변화량을 측정하는** M_2**로 안전한 QueryGroup만 선별해 리플레이로 사용**하고, 나머지는 버리는 방식이다. 리플레이 후보는 **정보량이 큰 구간(정답률이 0.5에 가까운 그룹)**을 우선 탐색하되, 최종 결정은 **안정성 제약( M_2 \le \tau )**이 내린다.

---

## **1. 문제 정의**

대규모 롤아웃/비동기 환경에서 **stale(오래된) 데이터**를 사용하면, GRPO/PPO류 업데이트가 불안정해지거나, 안정성을 위해 클리핑을 강하게 걸면 유익한 신호가 잘려 **학습이 왜곡**될 수 있다.

우리는 다음 목표를 가진다.

- **목표:** 이미 존재하는 과거 데이터(리플레이)를 활용해 샘플 효율을 올리고 싶다.
- **제약:** 정책이 변한 정도가 큰(오프폴리시 gap 큰) 데이터는 학습을 깨뜨릴 수 있으므로, **안전한 범위에서만** 재사용하고 싶다.

---

## **2. 핵심 아이디어**

### **2.1 정책 변화량을** M_2**로 측정**

토큰별 importance ratio는

$r_{i,t} = \exp(\log \pi_\theta(a_{i,t}) - \log \pi_{\text{old}}(a_{i,t}))$

이고, 로그비를

$\ell_{i,t}=\log r_{i,t}$

로 두면, 정책 변화량은 \ell_{i,t}의 크기(특히 분산/outlier)에 반영된다.

우리는 **QueryGroup(쿼리 + G개 생성)** 단위로 전체 응답 토큰의 평균 제곱을 사용한다.

$M_2^{\text{group}} =
\frac{\sum_{i=1}^{G}\sum_t \mathbf{1}_{\text{resp}}(i,t)\,\ell_{i,t}^2}
{\sum_{i=1}^{G}\sum_t \mathbf{1}_{\text{resp}}(i,t)}$

- $\mathbf{1}_{\text{resp}}:$ response_mask (pad 제외, EOS 포함/제외 정책 고정)

### **2.2 리플레이는 “게이팅(선별)”로만 사용**

M_2를 **loss 내부에서 감쇠/클리핑에 쓰지 않는다.**

오직 리플레이 데이터를 **사용할지 말지(accept/reject)** 결정하는 기준으로만 쓴다.

$\text{accept}(g) \iff M_2^{\text{group}}(g) \le \tau$

- 기본 \tau = 0.04 고정(추후 ablation으로 변경)

### **2.3 “정보량 우선” 후보 탐색:** |p-0.5| **정렬**

이진 reward(정답/오답)에서 QueryGroup의 정답률을

$p_g=\frac{1}{G}\sum_{i=1}^G \mathbb{1}[\text{correct}(i)]$

라 하면, $p\approx 0.5$ 근방은 그룹 내 분산이 커져 GRPO에서 유효한 상대 비교 신호가 최대가 되기 쉽다.

따라서 replay 후보는

s_g = |p_g-0.5|

가 작은 순으로 우선 평가한다. 단, 최종 통과 여부는 M_2가 결정한다(안정성 우선).

---

## **3. 데이터 단위(리플레이 버퍼의 원자 단위)**

리플레이는 반드시 **QueryGroup 단위(원자 단위)** 로 저장/사용한다.

### **3.1 QueryGroup 구성**

- query_id
- prompt_token_ids, prompt_len
- generations[1..G]:
    - response_token_ids
    - response_mask
    - reward (또는 score)
    - old_logprob_tokens (**HF 기준 canonical**, response 토큰 위치만)

권장 메타:

- policy_version, timestamp ← policy version은 굳이… 싶긴 하다.

### **3.2 버퍼 운영**

- 고정 크기 FIFO 큐(예: N=64 QueryGroup)
- 새 QueryGroup push → N 초과 시 가장 오래된 항목 pop
- 리플레이가 0인 iteration이 발생해도 **버퍼를 비우지 않는다.** FIFO로 최신 샘플이 유입되며 자연스럽게 “재사용 가능한 샘플”이 다시 나타날 수 있다.

---

## **4. 학습 루프(운영 파라미터 합의)**

- rollout batch size B = 4 (새 on-policy QueryGroup 수)
- replay target R = 4 (이번 iteration에서 사용하고 싶은 replay QueryGroup 수)
- micro batch size =2 QueryGroup
- 기본적으로 train batch =B+R가 되지만,
    - 실제 사용 replay 수 R_{\text{use}}는 gate 통과 수에 따라 감소 가능
    - R_{\text{use}}는 micro batch의 배수로 내림: \{0,2,4\}
- 철학: **on-policy는 항상 포함**, replay는 보너스(부족분을 억지로 채우기 위해 추가 rollout 생성 X)

→ 이러한 배수 관계, 즉 rollout batch = replay 를 가진다는 점만 유념하고, 또 replay 수는 (부족하든, 가득 차든 상관없이) micro batch size로 나눠져야 한다는 사실만 기억하면 된다. 수치는 언제든지 변할 수 있음. 

---

## **5. 알고리즘 상세(2-pass + replay-first schedule)**

### **5.1 Pass 0: 온폴리시 롤아웃 및 저장**

1. vLLM 등으로 QueryGroup B=4개 생성(각 query당 G개 응답)
2. reward 계산
3. GRPO advantage(그룹 기반) 계산
4. HF teacher-forcing으로 old_logprob_tokens 계산 후 저장
    
    (on-policy 샘플도 이후 ratio 계산을 위해 old logprob를 canonical로 저장)
    

### **5.2 Pass 1: 리플레이 후보 선택 +** M_2 **게이팅 (no_grad, policy θ₀ 고정)**

**목표:** optimizer step을 밟기 전 정책 \theta_0 기준으로 “안전한 replay”만 선택한다.

**사전 필터**

- 그룹 내 reward 분산 0(adv=0)인 QueryGroup은 제외

**후보 우선순위**

- 후보를 s_g = |p_g-0.5| 오름차순 정렬
- tie-break: 최신(timestamp/policy_version) 우선 → 이후 고정 seed 랜덤

**게이팅**

- 정렬 순서대로:
    - new_logprob_tokens(no_grad) 계산
    - \ell = new-old, M_2^{group}=\text{mean}(\ell^2)
    - M_2^{group}\le \tau이면 accept
- 목표 R=4개 모이면 중단(early stop)
- max_scan은 버퍼 전체 길이로 설정하여 순회(=버퍼 전체 스캔)
- 버퍼 끝까지 봐도 R에 못 미치면 accept된 수만 사용하고, replay 부족 처리 규칙을 적용

**replay 내부 학습 순서(현재 고정)**

- Pass 1에서 accept된 replay 그룹은 **추가 정렬 없이**, 후보 스캔/accept 순서를 유지해 학습에 투입
- 필요 시 후속 ablation에서 replay 내부 정렬을 다시 도입

### **5.3 Pass 2: 학습(grad-enabled) — replay 먼저, on-policy 나중**

이번 iteration의 학습 데이터:

- replay 그룹 R_{\text{use}}개 + on-policy 그룹 B개

**미니배치 스케줄(고정)**

- replay-only 미니배치를 먼저 배치
- on-policy-only 미니배치를 뒤에 배치

예: B=4, R_{\text{use}}=4, \text{micro}=2

- step1: replay 2개
- step2: replay 2개
- step3: on-policy 2개
- step4: on-policy 2개

**각 미니 step마다 수행**

- new_logprob_tokens(grad-enabled) 재계산 (필수)
- ratio 계산
- GRPO(PPO-style) clipped objective로 업데이트(ε=0.2 유지)

> 설계 의도: replay 게이팅은 \theta_0에서 측정되므로, replay를 앞쪽에 배치해 학습 중 drift로 인해 M_2가 급증하는 위험을 줄인다.
> 

---

## **6. 수치/정확성 규칙(고정)**

- dropout 없음
- 모델 forward dtype: bf16 고정
- M_2 후처리 산술은 fp32로 수행
    - ell = (new-old).float()
    - M2 = masked_mean(ell*ell)
- EOS는 별도 커스텀 정책을 두지 않고, 내부 verl의 response_mask/attention_mask 규칙을 그대로 사용
- query-group 미완성(G개 미달)은 discard

---

## **7. replay 부족(0개 포함) 처리 규칙**

- R_{\text{accept}} < R이면 R_{\text{use}} = 2\lfloor \min(R_{\text{accept}},R)/2 \rfloor (0/2/4)
- R_{\text{use}} = 0이면 해당 iteration은 **on-policy만**으로 업데이트 진행
- 버퍼는 FIFO로 계속 갱신되며 비우지 않는다.

> 분석 포인트: “replay가 0이 되는 구간”은 정책 변화가 매우 큰 구간일 수 있으며, 이때 리플레이가 자동으로 꺼지면서 update-to-data ratio를 낮추는 효과가 생긴다.
> 

---

## **8. 로깅/분석 지표(필수)**

### **Pass 1(게이팅)**

- replay acceptance rate (전체/시간)
- M_2^{group} 분포(평균/상위 분위수)
- p, s_g 분포 및 s_g–acceptance 관계
- max_scan hit rate
- R_{\text{accept}}, R_{\text{use}}

### **Pass 2(학습)**

- outside rate: r\notin[0.8,1.2]
- effective clip rate: (A>0 \land r>1.2) 또는 (A<0 \land r<0.8)
- replay vs on-policy 분리 기록
- replay-first step index에 따른 M_2/clip-rate drift
- 유효 토큰 수(마스크 적용 후), replay 토큰 비중

---

## **9. 기대되는 동작/관찰 가설**

- p \approx 0.5 근방의 그룹이 우선 후보가 되지만, **정책이 급변하는 구간에서는** M_2**가 커져 replay acceptance가 급락**할 수 있다.
- 이는 “학습이 가파른 성장 구간에서 오프폴리시 재사용이 구조적으로 어려워질 수 있음”을 시사하며, 논문 분석 파트의 핵심 포인트가 될 수 있다.
- replay-first 스케줄은 gate 평가 시점(\theta_0)과 학습 시점(\theta)의 차이를 줄여 drift를 완화한다.

---

## **10. 비교/ablation 계획(명세 수준)**

- \tau: 0.04 vs 0.08 (=(2·0.2)²)
- replay 내부 정렬(후속 필요 시): M_2 오름차순 vs 랜덤 vs 내림차순
- replay 부족 처리: 가변(train batch 감소) vs on-policy 추가 생성으로 고정 train batch 유지
- M_2 집계: mean(기본) vs (향후) top-k/pctl (분석/후속)

---

## **11. (향후 노벨티 확장 포인트)**

현재 방식은 “정렬→게이팅” 휴리스틱이다. 이를 다음과 같이 **문제화**할 수 있다:

- 목표: 난이도(정보량) 우선 선택 (예: |p-0.5| 최소)
- 제약: M_2^{group}\le \tau (안정성)
- 해결: 정렬-게이팅은 제약 최적화의 단순 근사해법

추후에는 “선택된 집합의 평균 M_2 budget” 등으로 확장 가능.

---

# **부록: 용어 정리**

- **QueryGroup**: 하나의 query + G개의 생성 응답 묶음(원자 단위)
- **old_logprob**: QueryGroup 생성 시점(rollout policy) 기준 HF canonical logprob
- **new_logprob**: 현재 정책 기준 HF logprob (게이팅은 no_grad, 학습은 grad-enabled)
- M_2^{group}: 그룹 전체 응답 토큰의 (\log r)^2 평균
