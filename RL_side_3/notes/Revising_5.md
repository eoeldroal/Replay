# Revising_5: rev3 저하 원인 재검증 + rev4(320/256 floor) 반영

본 문서는 최근 분석/논의/구현 변경을 한 번에 정리한 최신 요약본이다.  
목표는 `ADV=0 제외` 아이디어를 유지하면서, **배치 안정성(찌꺼기 방지)** 과 **성능 저하 원인 분리**를 동시에 달성하는 것이다.

---

## 0) 문제 정의

- rev2(legacy bonus) 대비 rev3(`fixed_total_with_adv0_drop`) 성능이 낮게 관측됨.
- rev3는 이론적으로 `ADV=0` 제거로 신호 밀도를 높여야 하므로 직관과 충돌.
- 추가 논의에서 `fixed_total=320` 확장 시 `257~319` 그룹 구간(찌꺼기)이 다시 생길 수 있음.

---

## 1) 엄격 검증 결과 (코드/로그)

### 1.1 recency-only 의도 검증

- `recency_only`는 `last_training_step` 오름차순 정렬로 **오래된 샘플 우선**이 맞다.
- 선택된 replay는 즉시 `last_training_step=current_step`로 갱신된다.
- 같은 step on-policy가 같은 step replay로 재사용되는 구조는 코드상 차단되어 있다.

해석:
- “최근 학습 샘플 우선순위 하향” 의도는 구현 정합성이 확인됨.

### 1.2 rev3 성능 저하가 “오동작”인지 검증

- 결론: 오동작보다는 **분포/양 변화의 부작용** 가능성이 큼.
- 공통 step 구간 비교에서 rev3는 rev2 대비:
  - 평균 replay 사용량이 더 적고
  - 사용 replay 평균 age가 더 높음
  - M2 평균은 유사(정책 거리 조건만으로 성능 보장되지 않음)

해석:
- M2는 “정책 plausibility” 제약이지 “학습 효용” 보장이 아니다.
- 즉, `M2 통과` + `다양성 높음`만으로 성능 우위가 자동으로 나오지 않는다.

### 1.3 “재사용 tail”과 다양성 논의 정리

- rev3는 재사용 억제가 강해 고유 샘플 비율이 높게 나오며, 과도한 top-tail 재사용은 완화됨.
- 그러나 이 특성 자체가 성능 우위를 보장하지는 않음.

해석:
- 현재 병목은 “재사용 과다” 단일 이슈가 아니라,  
  **(replay 투입량) x (age 분포) x (학습 타깃 구성)**의 결합 이슈.

### 1.4 계산 비용 관점

- 현재 순서가 `old_log_prob -> advantage -> replay 결합 -> update_actor` 이므로,
  `ADV=0` 제거는 주로 `update_actor` 절감에 기여.
- `old_log_prob` 비용은 구조상 그대로 남음(현재 합의 유지).

---

## 2) 핵심 논의 결론

### 2.1 rev3 아이디어 자체는 유효

- `ADV=0` 제거 + replay 보충 자체는 방향성이 타당함.
- 다만 고정 총량을 320으로 올릴 때 replay 부족 구간에서 배치 크기 가변이 다시 생김.

### 2.2 320 확장의 안정성 이슈

- `fixed_total=320`에서 replay가 충분하지 않으면 `256~319` 구간이 가능.
- 이 구간은 mini-batch(64) 기준 잔여 배치(찌꺼기) 위험을 다시 키움.

### 2.3 최종 합의

- **선호 타깃**: 320
- **안전 floor**: 256
- 즉, “320이 가능하면 320, 아니면 256으로 스냅”으로 배치 안정성 확보.

---

## 3) 구현 변경 사항 (반영 완료)

### 3.1 trainer 로직 변경 (`ray_trainer.py`)

- 새 스케줄 설정 추가:
  - `algorithm.m2_replay.schedule.fixed_total_floor_groups` (기본 0)
- 동작 규칙:
  1. `fixed_total_with_adv0_drop` 모드에서 우선 타깃은 `fixed_total_groups`(예: 320)
  2. `onpolicy_nonzero + replay`가 320 미만이면 타깃을 `fixed_total_floor_groups`(예: 256)로 적용
  3. 필요 시 replay를 trim해서 `257~319` 구간 제거
  4. 부족분은 기존대로 on-policy `adv=0` fallback으로 보충

- 안전장치:
  - `fixed_total_floor_groups < 0` 비활성화 경고
  - `fixed_total_floor_groups >= fixed_total_groups` 비활성화 경고

- 신규 로깅/메트릭:
  - `m2_replay/selection/fixed_total_floor_groups`
  - `m2_replay/selection/fixed_total_target_applied`
  - `m2_replay/selection/replay_trimmed_for_floor`
  - full score dump에도 동일 필드 저장

### 3.2 실행 스크립트 변경 (`rev4.sh`)

- `experiment_name`을 `rev4`로 분리 (rev3 로그/체크포인트 충돌 방지)
- 설정:
  - `fixed_total_groups=320`
  - `fixed_total_floor_groups=256`
  - `buffer.max_query_groups=1024`

---

## 4) 호환성/변인 통제

- 기존 실행 파일(rev2/rev3)은 **변경 없이 그대로 실행 가능**.
- floor 설정은 opt-in이며, 미설정 시 기존 동작과 동일.
- 즉, 실험 비교 체계(기존 baseline/rev2/rev3 재현성)는 유지됨.

---

## 5) rev4 실행 후 필수 점검 항목

1. `m2_replay/selection/fixed_total_target_applied`
- 값이 320/256으로만 동작하는지 확인

2. `m2_replay/selection/final_train_groups`
- 257~319 구간 제거 여부 확인

3. `m2_replay/selection/replay_trimmed_for_floor`
- floor 발동 빈도와 trim 강도 확인

4. 성능/효율 동시 확인
- `critic/score/mean`, validation acc
- `timing_s/update_actor`, `timing_s/step`, throughput

---

## 6) 현재 결론

- rev3 저하는 “구현 오류”라기보다, 분포/양/타깃 동시 변화에서 생긴 현상으로 보는 것이 타당하다.
- rev4의 320/256 floor 전략은 해당 문제를 최소 수정으로 완화하는 실용적 대안이다.
- 다음 단계는 rev4 실측 로그로 floor 발동 빈도와 성능 회복 여부를 검증하는 것이다.

