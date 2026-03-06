# Retrieval Server API Spec (Port 10001)

## 1. 서버 식별 정보

- 문서 기준 시각: **2026-03-02**
- 프로세스: `RL_side_1_LNS/retrieval_server.py`
- 바인딩 포트: `10001`
- 확인된 로컬 NIC IP: `172.17.0.3` (`eth0`)
- Base URL (현재 런타임): `http://172.17.0.3:10001`

> 주의: `172.17.0.3`는 컨테이너/내부 네트워크 IP일 수 있다. 실제 학습 서버에서 접근 가능한 IP인지 반드시 학습 서버 측에서 `curl http://<IP>:10001/health`로 검증할 것.

---

## 2. 엔드포인트 요약

### 2.1 `GET /health`

- 목적: 서버 기동 및 인덱스 로드 상태 확인
- 요청 본문: 없음
- 정상 응답 예시:

```json
{"status":"ok","rows":80292}
```

의미:
- `status == "ok"`: 웹 서버 응답 가능
- `rows == 80292`: `deck_index.pt` 로드 완료 상태

### 2.2 `POST /search`

- 목적: 질의 텍스트 + `row_index` 기반 20개 후보의 순위(score) 반환
- Content-Type: `application/json`
- 요청 본문은 **JSON 배열(list)** 이어야 함

요청 스키마 (배열 원소 1개):

```json
{
  "query": "string",
  "request_idx": 0,
  "row_index": 0
}
```

필드 설명:
- `query` (권장 필수): 검색 질의 문자열
- `request_idx` (선택): 호출자 추적용 인덱스. 서버 응답에는 그대로 반환되지 않음
- `row_index` (실질 필수): deck row 인덱스 (`0 <= row_index < 80292`)

---

## 3. 응답 형식

응답은 요청과 동일한 길이의 배열이며, 각 원소는 아래 형식:

```json
{
  "results": [
    {"position": 19, "score": 11.0},
    {"position": 16, "score": 10.75}
  ]
}
```

`results` 규칙:
- 기본적으로 길이 `20`
- `position`: `0~19` 정수 (해당 deck 내 이미지 위치)
- `score`: float
- 정렬: `score` 내림차순

실응답 샘플 (`row_index=0`):

```json
[{"results":[{"position":19,"score":11.0},{"position":16,"score":10.75},{"position":10,"score":10.5625},{"position":3,"score":10.375},{"position":6,"score":10.375},{"position":9,"score":10.3125},{"position":0,"score":10.25},{"position":4,"score":10.25},{"position":5,"score":10.25},{"position":11,"score":10.25},{"position":12,"score":10.25},{"position":7,"score":10.1875},{"position":1,"score":10.125},{"position":2,"score":10.125},{"position":8,"score":10.125},{"position":13,"score":10.125},{"position":15,"score":10.125},{"position":18,"score":10.125},{"position":17,"score":10.0625},{"position":14,"score":9.9375}]}]
```

---

## 4. 실제 검증된 동작/에러 계약

아래는 현재 서버 구현에서 **실제 확인된** 동작이다.

1. 정상 요청 (`list`, 유효 `row_index`)  
   - HTTP `200`
   - 본문: `[ { "results": [...] } ]`

2. 요청 본문이 객체(dict)일 때 (`{"query":"x","row_index":0}`)  
   - HTTP `200`
   - 본문: `[]`  
   - 이유: 서버는 payload를 list로 가정하고 순회함

3. `row_index` 누락 (`[{"query":"x"}]`)  
   - HTTP `200`
   - 본문: `[{"results":[]}]`

4. `row_index` 범위 초과/음수 (`999999`, `-1`)  
   - HTTP `500 Internal Server Error`
   - 본문: `Internal Server Error`

5. 잘못된 JSON (`not-json`)  
   - HTTP `500 Internal Server Error`
   - 본문: `Internal Server Error`

운영 관점 권고:
- 클라이언트는 요청 전 `row_index` 범위를 사전 검증할 것
- 서버가 500을 반환하면 재시도 전에 payload 유효성부터 확인할 것

---

## 5. 에이전트용 호출 절차 (권장)

1. 시작 시 1회 헬스체크

```bash
curl -sS http://172.17.0.3:10001/health
```

2. `rows == 80292` 확인 후 검색 호출

```bash
curl -sS -X POST http://172.17.0.3:10001/search \
  -H 'Content-Type: application/json' \
  -d '[{"query":"What is the website?","request_idx":0,"row_index":0}]'
```

3. 응답 파싱
- `results[0]`이 최상위 후보
- `position`을 현재 샘플의 `document_images[position]`에 매핑해서 실제 이미지 선택

4. 실패 처리
- HTTP 5xx 또는 timeout: 짧은 backoff 재시도
- `results` 빈 배열: 해당 턴은 검색 실패로 간주하고 fallback 로직 진행

---

## 6. 서버 런타임/튜닝 파라미터 (현 구현)

- 바인드: `0.0.0.0:10001`
- GPU 디바이스: `cuda:0`
- 모델: `vidore/colqwen2-v1.0-hf`
- 마이크로배칭:
  - `max_batch=64`
  - `max_wait_ms=20.0`

Startup 단계:
1. `deck_index.pt` 로드
2. 임베딩 샤드 877개 로드
3. ColQwen2 모델 로드
4. `Server ready!` 이후 요청 수신

---

## 7. 운영 명령

### 상태 확인

```bash
pgrep -af '[r]etrieval_server.py'
lsof -iTCP:10001 -sTCP:LISTEN -n -P
curl -sS http://127.0.0.1:10001/health
```

### 로그 확인

```bash
tail -f /home/work/DDAI_revised/verl/logs/retrieval_server_10001.log
```

### 종료

```bash
pkill -f 'RL_side_1_LNS/retrieval_server.py --port 10001'
```

