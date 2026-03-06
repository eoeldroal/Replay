# ColQwen2 검색 서버 배포 가이드

> **대상**: H100 × 2 서버 관리자
> **목적**: RL 학습 서버의 SearchTool이 호출하는 검색 서버를 기동한다.

---

## 1. 개요

학습 서버에서 GRPO 훈련이 실행되면, 모델이 `search` 도구를 호출할 때마다
SearchTool이 이 검색 서버에 HTTP POST를 보낸다.
검색 서버는 쿼리 텍스트와 20개 이미지 임베딩의 유사도를 계산하여 순위를 반환한다.

```
학습 서버 (8-GPU)                          검색 서버 (H100 × 2) ← 이 문서
├── SearchTool.execute()                   ├── retrieval_server.py
│   POST {"query":"...", "row_index":42}   │   ├── ColQwen2 쿼리 인코딩 (GPU)
│   ────────────── HTTP ──────────────▶    │   ├── MaxSim 유사도 계산 (CPU)
│   ◀── {"results":[{"position":3,...}]}   │   └── 순위 반환
```

---

## 2. 필요 파일

검색 서버에 아래 파일들이 존재해야 한다.
**학습 서버와 NFS를 공유하면 이미 존재**하며, 그렇지 않으면 복사한다.

| 파일/디렉토리 | 크기 | 용도 |
|---|---|---|
| `RL_side_1_LNS/retrieval_server.py` | 24 KB | 서버 코드 (단일 파일) |
| `data/Visual_Document_Rag/VDR_final/deck_index.pt` | 107 MB | row_index → 임베딩 위치 매핑 |
| `data/Visual_Document_Rag/VDR_processed_filtered_2/embeddings/colqwen2/` | 42 GB | 877개 임베딩 샤드 |

> **합계**: 디스크 ~43 GB

### 디렉토리 구조 (최소 요구)

```
DDAI_revised/verl/                          ← CWD (여기서 실행)
├── RL_side_1_LNS/
│   └── retrieval_server.py
└── data/Visual_Document_Rag/
    ├── VDR_final/
    │   └── deck_index.pt
    └── VDR_processed_filtered_2/
        └── embeddings/colqwen2/
            ├── OpenDocVQA/    (473 shards)
            ├── VDR_ibm/       (200 shards)
            └── SlideVQA/      (204 shards)
```

NFS 공유가 아닌 경우 복사 예시:
```bash
# 학습 서버에서 → 검색 서버로
rsync -avP /home/work/DDAI_revised/verl/RL_side_1_LNS/retrieval_server.py \
    <H100_USER>@<H100_IP>:/home/work/DDAI_revised/verl/RL_side_1_LNS/

rsync -avP /home/work/DDAI_revised/verl/data/Visual_Document_Rag/VDR_final/deck_index.pt \
    <H100_USER>@<H100_IP>:/home/work/DDAI_revised/verl/data/Visual_Document_Rag/VDR_final/

rsync -avP /home/work/DDAI_revised/verl/data/Visual_Document_Rag/VDR_processed_filtered_2/embeddings/colqwen2/ \
    <H100_USER>@<H100_IP>:/home/work/DDAI_revised/verl/data/Visual_Document_Rag/VDR_processed_filtered_2/embeddings/colqwen2/
```

---

## 3. 필요 환경

### 3.1 conda 환경

학습 서버와 동일한 `verl` conda 환경이 필요하다.
없으면 아래 패키지를 설치한다:

| 패키지 | 최소 버전 | 비고 |
|---|---|---|
| `torch` | 2.8.0+ | CUDA 지원 필수 |
| `transformers` | 4.56.1+ | ColQwen2ForRetrieval 클래스 포함 |
| `fastapi` | 0.128+ | HTTP 서버 |
| `uvicorn` | 0.40+ | ASGI 서버 |
| `uvloop` | 0.22+ | 고성능 이벤트 루프 |
| `flash-attn` | 2.8+ | 선택사항, 없으면 SDPA로 자동 fallback |

```bash
# verl 환경이 이미 있는 경우 확인만
conda run -n verl python -c "
from transformers import ColQwen2ForRetrieval, ColQwen2Processor
import fastapi, uvicorn, uvloop
print('All packages OK')
"
```

### 3.2 ColQwen2 모델 가중치

서버 첫 기동 시 `vidore/colqwen2-v1.0-hf` 모델을 HuggingFace에서 자동 다운로드한다 (~4.2 GB).
다운로드가 완료되면 `~/.cache/huggingface/hub/models--vidore--colqwen2-v1.0-hf/`에 캐시된다.

인터넷 접속이 안 되는 경우, 학습 서버에서 모델 캐시를 복사한다:
```bash
rsync -avP /home/work/.cache/huggingface/hub/models--vidore--colqwen2-v1.0-hf/ \
    <H100_USER>@<H100_IP>:~/.cache/huggingface/hub/models--vidore--colqwen2-v1.0-hf/
```

---

## 4. 서버 기동

### 4.1 기본 실행 (GPU 1개)

```bash
cd /home/work/DDAI_revised/verl

conda run -n verl python RL_side_1_LNS/retrieval_server.py \
    --port 5002 \
    --device cuda:0
```

### 4.2 GPU 2개 활용 (2-인스턴스)

GPU 1개로 충분하지만, 처리량을 높이려면 2개 인스턴스를 띄울 수 있다.
단, **CPU RAM을 2배 사용**한다 (~70 GB).

```bash
cd /home/work/DDAI_revised/verl

# 인스턴스 1: cuda:0, port 5002
conda run -n verl python RL_side_1_LNS/retrieval_server.py \
    --port 5002 --device cuda:0 &

# 인스턴스 2: cuda:1, port 5003
conda run -n verl python RL_side_1_LNS/retrieval_server.py \
    --port 5003 --device cuda:1 &
```

2-인스턴스 시 학습 서버의 `LNS_tool_config.yaml`에서
`retrieval_service_url`을 로드밸런서 주소로 변경하거나,
nginx reverse proxy를 설정해야 한다 (§4.3 참조).

### 4.3 nginx 로드밸런서 (2-인스턴스용, 선택사항)

```nginx
# /etc/nginx/conf.d/retrieval.conf
upstream retrieval_backend {
    server 127.0.0.1:5002;
    server 127.0.0.1:5003;
}

server {
    listen 5000;
    location / {
        proxy_pass http://retrieval_backend;
    }
}
```

이 경우 학습 서버의 `retrieval_service_url`을 `http://<H100_IP>:5000/search`로 변경한다.

> **권장**: 우선 GPU 1개(§4.1)로 시작하고, 학습 중 병목이 확인되면 2-인스턴스로 확장한다.

### 4.4 백그라운드 실행 (tmux/screen)

```bash
# tmux 세션 생성
tmux new -s retrieval

cd /home/work/DDAI_revised/verl
conda run -n verl python RL_side_1_LNS/retrieval_server.py --port 5002 --device cuda:0

# Ctrl+B, D 로 detach
# 재접속: tmux attach -t retrieval
```

---

## 5. Startup 과정 및 소요 시간

서버 기동 시 아래 순서로 초기화가 진행된다:

```
[1/4] deck_index.pt 로드          ~2초       CPU RAM 107 MB
[2/4] 877 임베딩 샤드 전수 로드    ~2-5분     CPU RAM ~35 GB
[3/4] ColQwen2 모델 GPU 로드       ~10-30초   GPU VRAM ~4 GB
[4/4] MicroBatcher 시작            즉시

"Server ready!" 로그가 나오면 요청 수신 가능
```

### 리소스 사용량 (정상 상태)

| 리소스 | 사용량 | 비고 |
|---|---|---|
| GPU VRAM | ~4 GB | ColQwen2 모델 (bf16) |
| CPU RAM | ~36 GB | deck_index + 877 샤드 |
| 디스크 | ~43 GB | 임베딩 샤드 + deck_index |
| 네트워크 | port 5002 | 학습 서버에서 접근 가능해야 함 |

---

## 6. 정상 동작 확인

### 6.1 Health Check

```bash
curl http://localhost:5002/health
```

기대 응답:
```json
{"status": "ok", "rows": 80292}
```

- `rows: 80292`이면 deck_index 정상 로드.
- `rows: 0`이면 startup 미완료 또는 실패.

### 6.2 검색 테스트

```bash
curl -X POST http://localhost:5002/search \
    -H "Content-Type: application/json" \
    -d '[{"query": "What is the website?", "request_idx": 0, "row_index": 0}]'
```

기대 응답 (position/score 값은 다를 수 있음):
```json
[{
    "results": [
        {"position": 3, "score": 85.5},
        {"position": 7, "score": 82.1},
        {"position": 0, "score": 79.3},
        ...
    ]
}]
```

확인 사항:
- `results` 배열에 **20개** 항목이 있어야 함 (덱 크기)
- `position`은 0~19 정수
- `score`는 float (보통 50~150 범위)
- `position` 순서가 score **내림차순**이어야 함

### 6.3 학습 서버에서 연결 확인

학습 서버에서 실행:
```bash
curl http://<H100_IP>:5002/health
```

응답이 오면 학습 서버 → 검색 서버 네트워크 연결 정상.

---

## 7. 학습 서버 연동 설정

검색 서버가 정상 기동되면, 학습 서버의 설정 파일에서 IP를 확인한다.

**파일**: `RL_side_1_LNS/LNS_tool_config.yaml`
```yaml
retrieval_service_url: http://163.239.28.21:5002/search
```

이 IP(`163.239.28.21`)가 **검색 서버(H100)의 실제 IP**와 일치해야 한다.
다르면 수정한다:
```yaml
retrieval_service_url: http://<실제_H100_IP>:5002/search
```

그 후 학습 서버에서 훈련 시작:
```bash
cd /home/work/DDAI_revised/verl
bash RL_side_1_LNS/run_qwen2.5-3b_instruct_search_multiturn.sh
```

---

## 8. 트러블슈팅

### 서버가 시작되지 않음

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: No module named 'transformers'` | verl 환경 미활성화 | `conda run -n verl ...` 또는 `conda activate verl` |
| `ModuleNotFoundError: No module named 'uvloop'` | uvloop 미설치 | `conda run -n verl pip install uvloop` |
| `FileNotFoundError: deck_index.pt` | 데이터 파일 미복사 | §2 디렉토리 구조 확인 |
| `CUDA out of memory` | GPU에 다른 프로세스 | `nvidia-smi`로 확인 후 정리, 또는 `--device cuda:1` |
| `OSError: Could not load model` | ColQwen2 모델 미다운로드 + 인터넷 차단 | §3.2 모델 캐시 복사 |

### Startup 중 메모리 부족

877 샤드 로드에 CPU RAM ~35 GB 필요. `free -h`로 가용 메모리를 확인한다.
2-인스턴스 시 ~70 GB 필요.

### 학습 서버에서 timeout

| 증상 | 원인 | 해결 |
|---|---|---|
| `ConnectionRefusedError` | 서버 미기동 또는 포트 불일치 | 검색 서버에서 `curl localhost:5002/health` |
| `TimeoutError` | 네트워크 차단 또는 방화벽 | 검색 서버에서 `ufw allow 5002` 또는 방화벽 규칙 확인 |
| 응답이 느림 (>10초) | 샤드 로드 미완료 | "Server ready!" 로그 확인 후 재시도 |

### 로그 확인

서버는 stdout에 로그를 출력한다. tmux/screen에서 확인 가능.
주요 로그:
```
Loading deck_index from ...              ← [1/4] 시작
  deck_index loaded in 1.5s — 80292 rows ← [1/4] 완료
Loading 877 embedding shards from ...    ← [2/4] 시작
  loaded 100/877 shards...               ← 진행 중 (100개마다 출력)
  All 877 shards loaded in 180.0s        ← [2/4] 완료
Loading ColQwen2 model: vidore/...       ← [3/4] 시작
  Model loaded in 15.0s                  ← [3/4] 완료
MicroBatcher worker started              ← [4/4]
Server ready!                            ← 요청 수신 가능
  Rows: 80292
  Device: cuda:0
```

---

## 9. 서버 종료

```bash
# foreground에서 실행 중이면
Ctrl+C

# 백그라운드 프로세스면
pkill -f "retrieval_server.py"
```

---

## 10. 요약 체크리스트

```
[ ] 1. 파일 존재 확인
      [ ] RL_side_1_LNS/retrieval_server.py
      [ ] data/Visual_Document_Rag/VDR_final/deck_index.pt
      [ ] data/Visual_Document_Rag/VDR_processed_filtered_2/embeddings/colqwen2/ (877 shards)

[ ] 2. 환경 확인
      [ ] conda verl 환경 존재 (torch, transformers, fastapi, uvicorn, uvloop)
      [ ] ColQwen2 모델 캐시 또는 인터넷 접속 가능

[ ] 3. 서버 기동
      [ ] cd /home/work/DDAI_revised/verl
      [ ] conda run -n verl python RL_side_1_LNS/retrieval_server.py --port 5002 --device cuda:0
      [ ] "Server ready!" 로그 확인

[ ] 4. 정상 동작 확인
      [ ] curl localhost:5002/health → rows: 80292
      [ ] curl POST localhost:5002/search → results 20개 반환

[ ] 5. 학습 서버 연동
      [ ] 학습 서버에서 curl <H100_IP>:5002/health 응답 확인
      [ ] LNS_tool_config.yaml의 IP가 H100 실제 IP와 일치하는지 확인
```
