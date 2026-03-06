# Revising_6: rev2/rev3 해석 재정리 + ZVP replay 우선순위 브레인스토밍

## 상태 보정 (2026-03-06)

- 본 문서는 **ZVP 도입 의도와 해석 기준** 문서다.
- 현재 워크트리 기준으로 `zvp_recency`, sign-only 점수, batched M2 eval, one-turnover gate는 이미 코드에 반영되어 있다.
- 반면 `replay vs on-policy` 분리 actor 메트릭은 아직 직접 로깅되지 않았으므로, 아래 검증 설계는 일부 추가 구현을 전제로 한다.

## 1) 이번 라운드 핵심 정리

- `rev2 > rev3`로 보이는 관측은 존재하지만, 단일 런/초기 구간 기준이라 인과 단정은 보류.
- 논의 초점은 “누가 peak가 높았나”보다, **안정성/복원력/지속성**을 같이 보는 것으로 정리.
- `rev3`의 설계(adv=0 drop + fixed total)는 의도대로 동작 중이며, “오작동” 근거는 아직 없음.

## 2) 현재 합의된 사실 (코드/로그 기준)

- `recency_only`는 최근 학습 샘플 우선순위를 낮추는 방향으로 동작.
- `fixed_total_with_adv0_drop` 경로는 on-policy nonzero를 우선 사용하고 replay로 보충.
- 부족 시 on-policy adv=0 fallback으로 배치를 채우는 구조가 동작.
- `320/256 floor`(rev4)는 찌꺼기 배치(257~319) 구간을 피하기 위한 안전장치로 도입.

## 3) rev2/rev3 해석 시 주의점

- 배치 크기, replay 주입량, buffer 크기, adv=0 처리 방식이 동시에 바뀌어 교란요인이 큼.
- 따라서 “rev2가 더 좋다/나쁘다”를 단일 원인으로 설명하지 않기로 합의.
- 해석 단위는 아래 3개 축으로 분리:
  1. 성능(점수/검증)
  2. 안정성(grad_norm spike, 회복 속도)
  3. 효율(step time, throughput)

## 4) 신규 개선 방향 (구현 반영)

### 4.1 ZVP 기반 replay 우선순위 (rev5 반영)

- 목표: recency-only 단일 기준 대신, replay 샘플의 “현재 학습 가치”를 반영.
- 사용자 제안 요지:
  - `low logprob(=high entropy)` + `positive advantage` 샘플에 높은 우선순위 부여.
  - 선택되어 학습된 후 entropy가 내려가면, 다음 step에서 우선순위가 자연 하락(자기감쇠).
  - 기존 `old_log_prob`는 유지하고, 우선순위 점수(zvp)만 step 단위 갱신.

### 4.2 ZVP 점수의 대칭 확장(sign-only, rev5 반영)

- 철학: RL-ZVP의 “학습 신호 조절”과 유사하되, 본 프로젝트는 replay 선택 관점에서
  **잘한데 드문 경우**와 **틀렸는데 과신한 경우**를 동시에 높은 학습 가치로 본다.
- 대칭 규칙:
  - `A > 0` 이고 `p`가 낮은(=surprisal 높은) 토큰 -> 높은 점수
  - `A < 0` 이고 `p`가 높은(=surprisal 낮은) 토큰 -> 높은 점수
- 이번 합의: **adv 크기는 쓰지 않고 부호만 사용(sign-only)**.
- 그룹 점수 예시(개념식):
  - `p = exp(log_prob)` (0~1)
  - `zvp_pos = 1[A>eps] * (1-p)`
  - `zvp_neg = 1[A<-eps] * p`
  - `zvp_token = zvp_pos + lambda_neg * zvp_neg`
  - `zvp_group = mean(zvp_token over response_mask)`
- 스케일링 원칙:
  - 기존의 그룹 내 `min-max` 정규화는 제거
  - 그룹 간 절대 확률 스케일(`p`)을 직접 사용해 샘플 간 비교 가능성 유지
- 기대 효과:
  - `low-prob correct` 강화 + `high-prob wrong` 교정을 동시에 replay 우선순위에 반영
  - 기존 양수-adv 단일 점수 대비 교정 신호 보강
- 구현 원칙:
  - 기존 모드는 유지, 신규 스코어는 opt-in
  - `old_log_prob`는 유지, score 캐시만 갱신
  - 기본값은 `lambda_neg=1.0` (옵션으로 조정 가능)
  - sign-only 단계에서 outlier advantage 크기에 의해 선택이 지배되지 않도록 제어

### 4.3 현재 코드 반영 상태(핵심)

- 신규 모드: `selection.mode=zvp_recency`
- `zvp_use_recency=false`일 때 우선순위는 ZVP 단일 기준
- `zvp_ema_alpha=1.0` 설정 시 ZVP 캐시는 매 스텝 최신값으로 즉시 갱신
- 점수식은 sign-only 대칭:
  - `A>0`: `1-p` (저확률 정답) 우대
  - `A<0`: `p` (고확률 오답) 우대
- 기존 실행 파일 호환성 유지:
  - 기본 모드/기본값으로 기존 rev2~4 동작 불변
  - 신규 설정은 rev5에서만 활성화

### 4.4 우려사항 (분석 포인트로 고정)

- 우려 A: high-entropy replay가 많아지면 PPO ratio clipping 증가 가능.
- 우려 B: replay-first 순서에서 replay가 정책을 크게 밀면, 같은 step on-policy 구간 변형 가능.

## 5) 다음 검증 설계 (구현 후 확인)

1. source 분리 로깅 (replay vs on-policy)
- `pg_clipfrac`, `ppo_kl`, `grad 기여`

2. ZVP 관련 분포 추적
- replay 후보/선택 집합의 entropy 분포
- 샘플별 `(selection_count, entropy, zvp_score)` 시계열
- 양/음 분해 추적: `zvp_pos`, `zvp_neg`, `zvp_total` 분포
- sign-only 검증: `A` 크기와 무관하게 부호/entropy 조합으로 선택이 일관되는지 확인

3. 안정성 확인
- grad spike 빈도/크기/회복 step 수
- replay-first 영향: step 내 replay 후 on-policy 지표 변동

## 6) 실행 원칙

- 기존 실험(rev2/rev3/rev4) 재현성은 깨지지 않게 유지.
- 신규 로직은 opt-in 설정으로 추가.
- 구현 후에는 `ANALYSIS_POINTS.md` 가설 검증 항목부터 우선 추적.
- 이 문서를 SSOT로 쓰기보다, `FACTS_AND_DECISIONS.md`와 함께 “확장 축 설계 근거”로 읽는다.
