# Replay Analysis Points

## 현재 관찰 (step 1~6)

- step1은 replay 0, step2+는 replay 256 full-use
- `tau=0.04`에서 pass1 acceptance가 거의 1.0
- `update_actor`/`step` 시간이 baseline 대비 크게 증가
- 누적 재선택 상위 query_id가 빠르게 증가하며, 현재 설정에서는 max count가 4에서 상한
- 사용자 관찰상 초기 구간에서는 baseline 대비 성능 우세 신호가 있으나, 통계적 결론은 보류

## 최신 수치 스냅샷 (step 1 vs step 2~6)

- `timing_s/update_actor`: 약 `44.6s` -> `103~112s`
- `timing_s/step`: 약 `114s` -> `200~214s`
- `m2_replay/pass1/accepted_groups`: 거의 매 step `256`
- `m2_replay/pass1/acceptance_rate`: 거의 `1.0`

## 분석 목표

- 재사용 제한 없이도 M2 게이팅이 replay 재선택을 자연스럽게 줄이는지 확인

## 핵심 질문

1. 한 번 선택된 QueryGroup은 다음 step에서 `M2`가 올라가 재선택이 줄어드는가?
2. `tau`가 작을수록 이 효과가 강해지는가?
3. 재선택 감소가 M2 때문인지 FIFO/배치 제약 때문인지 분리 가능한가?

## 최소 지표

- step별 `acceptance_rate`, `accepted_groups`, `scanned_groups`
- QueryGroup별 선택 횟수(재사용 횟수) 분포
- 재사용 횟수별 `M2` 평균/분포
- step별 unique replay ratio (`unique_selected / total_selected`)

## 최소 비교 실험

1. `tau` 값 변화(예: 0.02 / 0.04 / 0.08)
2. replay on vs replay off
3. buffer 크기 변화( FIFO 영향 분리 )
4. replay 비율 축소(예: target 50%) vs full-use(100%)

## 현재 전제

- 재사용 제한(`max_reuse/cooldown/age_limit`) 없음
- 선택된 replay는 버퍼에서 제거하지 않음(FIFO로만 축출)
- same-step 혼입은 `select -> append`로 방지
