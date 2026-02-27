# 1) 클래식 파이프라인(현 단계 KL 없는 GRPO 기준)

## 0) 현재 구현/실행 상태 (2026-02-26)

- 현재 코드에는 `m2_replay.py`, `m2_replay_adapter.py`, `ray_trainer.py` 통합이 반영되어 실제 학습 중이다.
- 실험 스크립트: `RL_side_3/grpo-qwen-math-7b-s8-m2-replay.sh`
- 초기 step 관찰:
  - step1: replay 0 (버퍼 워밍업)
  - step2+: replay `selected=256`, `used=256` (full-use)
  - `tau=0.04`에서 pass1 acceptance가 거의 1.0
- 시간 관찰:
  - `timing_s/update_actor`: step1 약 44.6s -> step2~6 약 103~112s
  - `timing_s/step`: step1 약 114s -> step2~6 약 200~214s
- query_id 디버그 출력(on-policy/replay/top-replayed)이 추가되어 재선택 패턴을 실시간 추적 가능하다.
- 현재 설정(`buffer=1024`, `new_groups=256/step`)에서는 관측 가능한 query 재사용 상한이 4회로 나타난다.
- 사용자 관찰 기준 초기 구간 baseline 대비 우세 신호가 있으나, 확정 결론은 장기 러닝 후 판단한다.


## (A) Rollout 생성: vLLM

- vLLM로 **토큰 시퀀스 자체**를 빠르게 생성
- 저장:
    - `prompt_token_ids`, `response_token_ids`(또는 concat된 `input_ids`)
    - `prompt_len`, `response_len`
    - stop 이유(eos/length), sampling params(temperature/top_p 등)
        - 나중에 HF 로 동일한 설정으로 log prob을 계산해야 하기 때문에 sampling params 저장
    - query_id(그룹 묶기용), sample_id

### 주의점(치명 포인트)

1. **토크나이저/템플릿 완전 동일성**
    - vLLM에 쓴 tokenizer/chat template와 HF(logprob 계산)에서 쓰는 tokenizer/template가 1:1로 같아야 함.
    - 다르면 “같은 텍스트”라도 tokenization이 달라져서, 이후 logprob/ratio/M₂ 전부 의미가 없어짐.
2. **EOS/특수토큰 처리**
    - vLLM이 생성한 eos 포함 여부, eos 이후 토큰이 있는지, pad 처리 규칙을 명확히.
    - HF logprob 계산 시에도 eos/pad를 mask에서 제외해야 함.
3. **정확히 ‘토큰’ 기반으로 저장**
    - 텍스트만 저장해두고 나중에 다시 tokenize하면 어긋날 확률이 큼(공백/개행/템플릿 차이).
    - **반드시 vLLM에서 나온 token_ids를 그대로로 저장**.

## (B) Scoring: reward 계산

- prompt+response에 대해 reward 산출(규칙 기반/검증기/모델 기반 등)
- GRPO라면 query당 G개 response의 reward 벡터를 얻어야 함.

### 주의점

1. **그룹 단위 정합성**
    - query_id별로 정확히 G개가 묶여야 함(중간 실패/타임아웃/빈 응답이 있으면 그룹이 깨짐).

## (C) Advantage 계산: GRPO

### 기본 GRPO

- query 그룹 평균/표준편차로 정규화:
    - $(\mu_q=\text{mean}(R)), (\sigma_q=\text{std}(R))$
    - $(A_i=(R_i-\mu_q)/(\sigma_q+\epsilon))$

### 주의점

1. **adv의 토큰 적용 범위**
    - GRPO는 흔히 “응답 전체 토큰에 동일 A”를 적용.
    - 반드시 `response_mask`로 **응답 토큰만** loss에 포함(프롬프트 제외).
2. **정규화 혼합 시 분포 왜곡**
    - 배치 std를 쓰면 안정적일 수 있지만,
    - 서로 다른 태스크/스코어 분포를 섞으면 std가 의미가 희석됨.
    - 필요하면 task/bucket별 std로 나누는 옵션 고려.

> 추가 ablation study
> 
> 1. lite-ppo 방식
> 2. adv = 0 인 샘플 제외 → 고정으로 픽스

## (D) HF(Transformers) 기준 logprob 계산: canonical

### 원칙

- vLLM logprob는 HF와 수치적 차이가 날 수 있으니,
- **“정확한/일관된 logprob”는 HF로 계산**해서 사용.

### 계산 방식(중요)

- 이미 token_ids가 고정되어 있으므로,
- **teacher-forcing 1회 forward**로 모든 토큰 위치 logits를 얻고,
- shift해서 토큰 logprob를 뽑는다:
    - logits[:, :-1] ↔ tokens[:, 1:]
- 저장할 것:
    - `old_logprob_tokens`(응답 토큰 위치)
    - `response_mask`(pad/eos 제외)

### 주의점(여기서 가장 많이 틀림)

1. **Shift 정합성**
    - “토큰 t의 확률”은 t 위치 logits가 아니라 **t-1 위치 logits**에서 나온다.
    - 실수가 한 번 나면 ratio/M₂ 전부 망가짐.
    - 왜 토큰 t의 확률은 t-1 위치의 logits 값으로부터 나올까?
        
        ## 1) 왜 `t`가 아니라 `t-1` 위치의 logits에서 토큰 logprob를 읽나?
        
        ### 핵심 원리: Causal LM의 logits는 “현재 토큰”이 아니라 “다음 토큰”을 예측한다
        
        길이 (T)의 토큰 시퀀스가 있을 때,
        
        $x = [x_0, x_1, \dots, x_{T-1}]$
        
        Causal LM에 `input_ids = x`를 넣으면, 모델은 각 위치 (t)에서
        
        $\text{logits}[t] \approx \text{분포 } p(\text{다음 토큰 } x_{t+1} \mid x_{\le t})$
        
        을 출력합니다.
        
        즉,
        
        - `logits[t]`는 **토큰 (x_t)** 의 확률이 아니라,
        - **다음 토큰 (x_{t+1})** 의 확률 분포입니다.
        
        그래서 “토큰 (x_t)의 logprob”는
        
        - (x_t)를 예측했던 위치, 즉 **직전 위치 (t-1)** 의 logits에서 읽게 됩니다:
        
        $\log p(x_t \mid x_{<t}) = \log\text{softmax}(\text{logits}[t-1])[x_t]$
        
        ---
        
        ## 2) 예시로 보면 바로 이해된다 (BOS 포함)
        
        토큰이 4개라고 합시다.
        
        $x = [x_0, x_1, x_2, x_3] = [\langle BOS\rangle,\ \text{"I"},\ \text{"love"},\ \text{"NLP"}]$
        
        모델 출력은 길이 4의 logits:
        
        - `logits[0]` : (p(\cdot \mid x_0)) → **다음 토큰 (x_1)** 을 예측하는 분포
        - `logits[1]` : (p(\cdot \mid x_0,x_1)) → **다음 토큰 (x_2)** 을 예측하는 분포
        - `logits[2]` : (p(\cdot \mid x_0,x_1,x_2)) → **다음 토큰 (x_3)** 을 예측하는 분포
        - `logits[3]` : (p(\cdot \mid x_0,x_1,x_2,x_3)) → “그 다음 토큰”을 예측(우리 시퀀스 밖)
        
        따라서 logprob는 이렇게 뽑습니다.
        
        - ( \log p(x_1 \mid x_0) ) = `log_softmax(logits[0])[ x1 ]`
        - ( \log p(x_2 \mid x_0,x_1) ) = `log_softmax(logits[1])[ x2 ]`
        - ( \log p(x_3 \mid x_0,x_1,x_2) ) = `log_softmax(logits[2])[ x3 ]`
        
        여기서 딱 질문 2번이 성립합니다:
        
        > **맞아요. `logits[0]`는 BOS를 본 뒤 “다음 토큰 분포”이고, 실제 다음 토큰이 `x[1]`이므로, 그 `x[1]`을 `logits[0]` 분포에서 찾아 logprob를 읽으면 됩니다.**
        > 
        
        ---
        
        ## 3) 그래서 코드가 `logits[:, :-1] ↔ tokens[:, 1:]`가 된다
        
        배치까지 포함해서 보면:
        
        - `tokens` shape: `[B, T]` = ([x_0, x_1, \dots, x_{T-1}])
        - `logits` shape: `[B, T, V]` (V는 vocab)
        
        우리가 점수화할 “정답 토큰”은 보통 첫 토큰을 제외한 `tokens[:, 1:]`입니다(= ([x_1,\dots,x_{T-1}])).
        
        그 토큰들의 분포는 `logits[:, 0:T-1]`에 들어 있으니:
        
        - `logits[:, :-1]` (0~T-2 위치) ↔ `tokens[:, 1:]` (1~T-1 토큰)
        
        이렇게 “한 칸 밀어서” 짝을 맞춰야 합니다.
        
        ---
        
        ## 4) 한 줄 요약
        
        - **`logits[t]`는 “다음 토큰” (x_{t+1})의 분포**다.
        - 그래서 **토큰 (x_t)의 logprob는 `logits[t-1]`에서 읽는다.**
        - 따라서 구현이 **`logits[:, :-1]`와 `tokens[:, 1:]`를 shift해서 매칭**하는 형태가 된다.
2. **prompt 길이 인덱싱**
- response의 첫 토큰 logprob는 “prompt 마지막 토큰 위치 logits”에서 읽는다.
- prompt_len 계산이 템플릿 포함 기준으로 정확해야 한다.
1. **pad/eos mask**
- pad 위치는 반드시 제외.
- eos를 포함할지 여부를 정책적으로 고정(보통 eos도 응답 토큰으로 포함하되, 길이제한으로 잘린 케이스와 일관되게 처리).
1. **dropout/정밀도/모델 모드**
- logprob 계산은 `model.eval()` + dropout off가 기본.
- 학습 업데이트용 new_logprob는 `train()`이어도 dropout이 있으면 ratio 노이즈가 생김 → 보통 actor는 dropout을 꺼두는 편이 안전.
- bf16/fp16 차이로 logprob가 미세하게 흔들릴 수 있음. “old/new 모두 동일 dtype/커널 경로”를 최대한 맞추기.
1. **attention_mask/position_ids**
- left padding을 쓰면 position_ids가 꼬일 수 있음.
- verl이 사용하는 padding 방향과 동일하게 맞추고, 필요하면 position_ids를 명시.
1. **캐시(KV-cache) 사용 여부**
- logprob 계산은 일반적으로 cache 없이 한 번에 처리(teacher forcing).
- cache 켜고 토큰별로 돌리면 느리고, 구현 오류가 늘어남.
1. **길이/메모리**
- “forward 1번”이어도 길고 배치 크면 OOM 가능.
- old_logprob(no_grad)은 비교적 가볍지만,
- new_logprob(grad 필요)는 activation 때문에 빡셈 → micro-batch/gradient checkpointing 고려.

## (E) Policy update: GRPO loss 적용

- $ratio: (r_{t}=\exp(\log\pi_{\theta}(a_t)-\log\pi_{\text{old}}(a_t)))$
- loss는 응답 토큰에 대해 마스크 평균/합(verl의 token-mean/seq-mean 옵션과 정합)

### 주의점

1. **old_logprob 재계산 금지**
- rollout 시점/저장 시점 HF old_logprob를 canonical로 저장해두고,
- 업데이트에서는 new_logprob만 계산하는 게 맞음(비용/일관성).
1. **gradient 흐름**
- old_logprob는 상수(no_grad).
- new_logprob는 autograd 그래프 위에 있어야 함.
- 실수로 detach하면 학습이 안 됨.
1. **클리핑/제약의 일관성**
- GRPO baseline을 먼저 구현한다면, clipping 여부/epsilon을 고정하고,
- 나중에 SAPO로 바꿀 때 실험 비교가 가능하도록 설계.
1. **분산 학습 동기화**
- 그룹 평균/배치 std 같은 통계는
    - “데이터 병렬 shard 내부”인지
    - “전체 world”인지
        
        를 명확히.
        
- 특히 배치 std를 쓰면 all-reduce 범위를 잘못 잡으면 재현성/스케일링이 깨짐.
1. **재현성**
- rollout sampling RNG, reward evaluator nondeterminism, hf forward nondeterminism(FlashAttention 등) 모두 로그/seed 정책 필요.

---

# 2) Experience Replay(현 단계: M₂-gated replay를 얹기 전 준비)

## 버퍼에 저장해야 하는 최소 항목(강추)

- token_ids: prompt+response
- prompt_len, response_mask
- reward, (query_id, group_index)
- **HF old_logprob_tokens** (응답 토큰 위치) → 필수!!
- (선택) advantage 값(단, “lite-ppo 방식”이면 배치 std가 달라져 재계산 필요할 수 있음 → 이럴 경우 각 generation에 대응되는 reward 값.)

### 주의점

1. **그룹 단위 보존**
- GRPO 구조상 query별 G개를 같이 꺼낼 수 있어야 advantage가 안정적.
- 샘플이 누락되면 그룹이 망가짐 → 버퍼 저장 시 “그룹 완성”을 원자적으로 처리.

> Ablation study : 0.5 우선 추출하기. → ablation 말고 고정
> 

---

# **3) M₂ 게이팅 기반 리플레이: 정의와 필요한 값**

## **목표**

- M₂를 **감쇠/클리핑의 대체**가 아니라, **리플레이 데이터의 승인/거절(게이팅)** 기준으로 사용한다.
- 안정성 최우선: \tau는 **고정**하며, 기본값은 **0.04**로 시작한다(추후 2^2 스케일 비교는 ablation).

## **리플레이 버퍼의 “원자 단위”**

- **QueryGroup 단위**로 저장한다(= query 하나에 대한 G개 생성 묶음).
    - GRPO advantage 계산과 M₂ 게이팅 모두 query-group 단위 정합성을 요구하므로, 그룹은 원자적으로 보존되어야 한다.

### **QueryGroup에 저장할 필드(최소)**

- query_id
- prompt_token_ids, prompt_len
- generations[1..G]:
    - response_token_ids
    - response_mask (응답 토큰만 1, pad는 0, eos 포함 여부는 정책으로 고정)
    - reward (또는 score)
    - old_logprob_tokens (**HF canonical**, response 토큰 위치만)
- (권장) policy_version, timestamp (버퍼 정렬/청소/재현성용)

> 참고: sampling params는 HF logprob 계산에 “필수”는 아니지만, 롤아웃 재현 및 분석을 위해 메타로 보관 가능.
> 

## **M₂ 계산에 필요한 값(준비 완료)**

- 저장된 old_logprob_tokens (HF 기준, response 토큰 위치)
- 게이팅 시점에 계산한 new_logprob_tokens (HF 기준)
    - 게이팅(선별) 단계는 **no_grad**로 계산 가능
- response_mask

## **토큰별 정의**

- 토큰별 로그비:
    
    $\ell_{i,t} = \log r_{i,t} = \log\pi_{\theta}(a_{i,t}) - \log\pi_{\text{old}}(a_{i,t})$
    
- 토큰별 제곱:
    
    $m_{i,t} = \ell_{i,t}^2$
    

## **그룹(QueryGroup) 단위 M₂ 정의(합의 버전)**

- **전체 응답 토큰에 대한 평균**:
    
    $M_2^{\text{group}}=\frac{\sum_{i=1}^{G}\sum_{t}\mathbf{1}_{\text{resp}}(i,t)\,\ell_{i,t}^{2}}{\sum_{i=1}^{G}\sum_{t}\mathbf{1}_{\text{resp}}(i,t)}$
    
- 여기서 $\mathbf{1}_{\text{resp}}$는 response_mask.

### **수치 안정성(필수 권고)**

- 모델 forward dtype은 bf16 고정이지만,
- M_2 후처리는 fp32로 계산:
    - ell = (new_logprob - old_logprob).float()
    - m2 = (ell * ell) 후 masked mean

# **4) M₂ 게이팅 기반 리플레이 선택/학습 절차**

## **4.0 배치/스텝 정의 (운영 합의 명시)**

- **rollout batch size = B = 4** (새 on-policy QueryGroup 수)
- **replay target batch size = R = 4** (게이팅 통과한 replay QueryGroup 목표 수)
- **train batch size = B + R = 8**
- **micro batch size = 2 QueryGroup**
- **optimizer step per iteration = (B+R)/2 = 4**
- *Replay buffer는 QueryGroup 단위로 고정 크기 N(예: 64) FIFO 큐로 유지하며, 새로운 QueryGroup이 들어오면 push하고 N을 초과하면 가장 오래된 항목을 pop한다.*

> 철학: on-policy는 항상 포함, replay는 보너스.
> 

> replay가 부족하면(게이팅/scan 제한) 해당 iteration의 train batch는 8보다 작아질 수 있으며, 부족분을 억지로 채우기 위해 추가 rollout을 생성하지 않는다(비용 최소화 우선).
> 

## **4.1 리플레이 후보 우선순위(버킷 없이 정렬)**

### **정답률과 우선순위**

- 각 QueryGroup g에 대해 이진 reward 기준 정답률:
    
    p_g = \frac{1}{G}\sum_{i=1}^{G}\mathbb{1}[\text{correct}(i)]
    
- 우선순위 점수:
    
    s_g = |p_g - 0.5|
    
- *s_g가 작은 순(=0.5에 가까운 순)**으로 replay 후보를 평가한다.

### **tie-break (고정)**

1. 최신(timestamp 또는 policy_version) 우선
2. 그 다음은 고정 seed 랜덤 (재현성)

### **사전 필터(고정)**

- **adv=0(또는 그룹 내 reward 분산 0) QueryGroup은 무조건 제외**
    - 학습 신호가 거의 없고, |p−0.5| 기반 우선순위와도 무관하게 낭비.

> 구현 팁: |p−0.5| 정렬은 reward만으로 가능하므로, HF 계산 전에 후보를 좁히는 “cheap filter”로 작동한다.
> 

## **4.2 승인 정책(합의 버전)**

- **하드 컷 고정 τ**:
    
    $\text{accept}(g)\iff M_2^{\text{group}}(g)\le \tau$
    
- 기본: \tau=0.04 (고정)
- 비교군(후속 ablation): \tau=0.08 등

### **그룹 단위 M₂ 정의(재명시)**

- **QueryGroup 전체(G개 응답)의 응답 토큰을 모두 합친 평균**:
    
    $M_2^{\text{group}}=
    \frac{\sum_{i=1}^{G}\sum_{t}\mathbf{1}_{\text{resp}}(i,t)\,\ell_{i,t}^{2}}
    {\sum_{i=1}^{G}\sum_{t}\mathbf{1}_{\text{resp}}(i,t)}
    \quad,\quad
    \ell_{i,t}=\log\pi_\theta(a_{i,t})-\log\pi_{\text{old}}(a_{i,t})$
    
- response_mask는 pad를 제외하며, EOS 처리는 내부 verl 기본 규칙을 그대로 따른다.

### **수치 처리(고정)**

- 모델 forward는 bf16 고정.
- M₂ 계산은 fp32로 캐스팅 후 수행:
    - ell = (new_logprob - old_logprob).float()
    - M2 = masked_mean(ell*ell)

## **4.3 계산 비용 최적화: 2-pass(필수)**

### **Pass 1: 게이팅용(no_grad) — “iteration 시작 시점 θ₀에서만” 평가**

**목표:** 현재 policy \theta_0 기준으로 replay 후보의 M_2^{group}를 측정하고, τ 이하만 선별한다.

이 평가는 **optimizer step을 밟기 전**에 수행한다.

1. replay buffer의 QueryGroup에 대해 s_g=|p_g-0.5| 기준으로 정렬(tie-break 적용)
2. 정렬된 순서대로 new_logprob_tokens를 no_grad로 계산 → M_2^{group} 계산 → τ 통과 여부 판정
3. τ를 통과한 그룹을 누적하여 목표 replay 개수 R을 채우면 **즉시 종료(early stop)**
4. max_scan은 버퍼 전체 길이로 설정해 순회한다(=버퍼 전체 스캔).
5. 버퍼 끝까지 봐도 목표 R을 못 채우면, accept된 수만 사용해 해당 iteration을 진행한다.

### **max_scan 상한(고정)**

- max_scan = 현재 버퍼 길이(버퍼 전체를 보되, 목표 R을 채우는 순간 즉시 종료)
- 전체 스캔 후에도 accept가 부족하면 replay를 덜 사용

### **추가: replay 내부 순서(학습 순서용) — 합의 반영**

- 현재는 Pass 1에서 accept된 replay 그룹을 **추가 정렬 없이** 그대로 학습에 넣는다.
- replay 내부 정렬(M_2 오름차순/랜덤/내림차순)은 후속 필요 시 ablation으로 추가한다.

### **Pass 2: 학습용(grad-enabled) — 미니배치 4스텝, “replay 먼저 → on-policy 나중”**

Pass 1에서 선택된 replay R개 + 새 on-policy B개를 합쳐 학습 배치를 구성한다.

### **학습 배치 구성(고정 운영)**

- replay 미니배치(2개 QueryGroup)들을 **앞쪽 스텝에 배치**
- on-policy 미니배치들을 **뒤쪽 스텝에 배치**

예: (R=4, B=4, micro=2)

- step1: replay 2개
- step2: replay 2개
- step3: on-policy 2개
- step4: on-policy 2개

> 목적: replay 게이팅이 \theta_0에서 측정되었으므로, replay를 앞쪽에 배치해 “게이팅 측정 시점과 학습 시점의 drift”를 최소화한다.
> 

### **각 미니배치 업데이트마다 수행(필수)**

각 step에서:

1. 해당 미니배치 샘플만 new_logprob_tokens를 **grad-enabled**로 재계산
2. ratio 계산:
    
    r_{i,t}=\exp(\log\pi_{\theta}(a_{i,t})-\log\pi_{\text{old}}(a_{i,t}))
    
3. GRPO(PPO-style) clipped objective로 업데이트(ε=0.2 유지)
4. optimizer.step()

> 핵심 합의:
> 
> 
> **매 미니 step마다 new logprob 재계산은 필수**
> 

> (policy가 step마다 바뀌기 때문)
> 

### **drift 관측(분석용 권장 로그)**

- replay 미니 step index가 뒤로 갈수록(특히 step2 이후) replay 샘플의
    - M_2^{group}, outside rate, effective clip rate
        
        가 증가하는지 추적(“게이팅 drift” 분석 소재)
        

## 부족분이 발생할 경우?

### **1) replay가 “부족”하다는 게 정확히 뭘 의미하나?**

여기서는 보통 두 가지 케이스가 있어.

1. **accept된 replay QueryGroup 수가 목표 R=4보다 작음**
    - (τ=0.04 + max_scan 제한) 때문에 생김
2. **accept 수는 있는데 micro_batch=2에 맞게 쪼개기 애매함(홀수 등)**
    - replay-only 미니배치를 앞에 두려는 스케줄에서 특히 문제
- rollout_batch(B)=4는 항상 수행
- replay_target(R)=4를 시도하되,
- 실제 사용 replay 수는
    
    $R_{\text{use}} = 2\lfloor \min(R_{\text{accept}},4)/2 \rfloor$
    
    (0/2/4 중 하나)
    
- train_batch = 4 + R_use
- optimizer step 수 = (4 + R_use)/2 → 2/3/4 중 하나
- 미니배치 스케줄: **replay-only step 먼저**, 이후 on-policy-only step

### **이게 왜 좋나?**

- replay 부족 시 억지로 채우지 않아 **비용/안정성 철학에 부합**
- replay-only 먼저라는 스케줄 규칙이 **항상 깨끗하게 유지**
- batch/step 변동이 있어도, 사실상 RL에서는 **토큰 수 변동이 원래 존재**하므로 구현이 크게 어렵지 않음
- 부족이 자주 발생하면 그 자체가 분석 포인트(“왜 이 구간에서 안전한 replay가 scarce한가?”)

## **4.4 로깅(필수: τ 고정 정당화 + 분석용)**

### **Pass 1(게이팅)에서 기록**

- replay acceptance rate (전체, step별)
- 후보들의 s_g 분포, accept된 s_g 분포
- M_2^{group} 분포(평균/상위 분위수)
- p 분포 및 s_g와 acceptance의 관계
- (선택) “fail cache”를 위한 연속 실패 횟수/eviction 통계

### **Pass 2(학습)에서 기록 — “클리핑 왜곡” 감시**

- outside rate: r\notin[0.8,1.2]
- effective clip rate: (A>0 \land r>1.2) 또는 (A<0 \land r<0.8)
- replay vs on-policy 각각의 clip 통계(분리 기록 권장)
- 실제 업데이트에서 replay 토큰 비중(유효 토큰 수)
- (권장) step index별(replay step1, step2, on-policy step3, step4)로 clip/M₂ 변화 기록

## **4.5 엣지 케이스 처리(고정 규칙)**

- 응답 토큰 수가 0(분모=0)이면 해당 그룹 reject
- 그룹 미완성(G개 미달)은 query-group 전체 discard
- EOS는 별도 커스텀 정책 없이 내부 verl response_mask 규칙을 그대로 사용
- replay 부족 시:
    - **실제 사용 replay 수** R_{\text{use}} 는 accept된 수에 따라 결정되며, **micro batch(=2)의 배수로 내림**하여 사용한다(예: 0/2/4).
    - 부족분을 억지로 채우기 위해 추가 rollout을 생성하지 않으며, 그 iteration은 **on-policy 업데이트만**으로 진행될 수 있다.

# **(추가) 문서에 한 줄 더 넣으면 좋은 “정확한 표현”**

- “replay가 on-policy보다 M₂가 작아서 뒤로 미룬다”가 아니라
    
    **“replay는 게이팅이 θ₀ 기준이므로, 학습에서 θ가 변하기 전에 replay를 먼저 소비해 drift를 줄인다”**
    
    라고 쓰는 게 엄밀하고 설득력 있습니다.
    

---

원하면, 이제 이 보강본을 기준으로 **실제로 ‘mini step 4’에서 어떤 텐서가 어떤 shape로 들어오고, M₂를 어떻게 그룹 평균으로 계산하는지**(PyTorch 의사코드 + shape 주석)까지 내려가서 완전히 구현 스펙으로 마무리해줄게.
