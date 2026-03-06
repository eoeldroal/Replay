# AGENTS.md — LNS (Learn aNd Search) 프로젝트

> Visual Document Retrieval을 위한 강화학습 기반 멀티턴 검색 에이전트 프로젝트.
> 모델이 `<search>`, `<bbox>`, `<search_complete>` 도구를 사용해
> 80,292개 QA 샘플의 20장 이미지 덱에서 정답 문서를 찾도록 학습한다.

---

## 1. 프로젝트 구조

```
DDAI_revised/verl/
├── RL_side_1_LNS/                    ← 이 프로젝트의 설정/실행 파일
│   ├── AGENTS.md                     ← 이 문서
│   ├── retrieval_server.py           ← ColQwen2 검색 서버 (FastAPI, §9.1)
│   ├── LNS_tool_config.yaml          ← SearchTool, ImageCropper 설정
│   ├── search_multiturn_grpo.yaml    ← GRPO 훈련 설정
│   ├── run_qwen2.5-3b_instruct_search_multiturn.sh  ← 실행 스크립트
│   ├── Reference/                    ← 참고 논문 (GLM-4.5V, InternVL3.5)
│   └── ToDo/                         ← 향후 계획 (vLLM 마이그레이션 등)
│
├── verl/tools/LNS/                   ← LNS 전용 도구 (자유롭게 수정 가능)
│   ├── search_tool.py                ← SearchTool — 검색 서버 호출 + position 해석
│   └── bbox_tool.py                  ← ImageCropper — 이미지 영역 크롭
│
├── verl/utils/reward_score/
│   └── format_ndcg_reward.py         ← NDCG 보상 함수 (LNS 전용, 수정 가능)
│
├── verl/utils/dataset/
│   └── rl_dataset.py                 ← 데이터 로더 (verl 프레임워크, 수정 금지)
│
├── verl/experimental/agent_loop/
│   └── tool_agent_loop.py            ← 멀티턴 에이전트 루프 (verl 프레임워크, 수정 금지)
│
├── data/Visual_Document_Rag/
│   ├── VDR_final/
│   │   ├── train.parquet             ← 80,292행 훈련 데이터 (155.9 MB, 11컬럼)
│   │   ├── add_rl_columns.py         ← parquet RL 컬럼 생성 스크립트
│   │   ├── build_deck_index.py       ← 덱 인덱스 빌더 스크립트
│   │   ├── deck_index.pt             ← 덱 인덱스 (106.4 MB, row→임베딩 위치)
│   │   └── corpus/img/               ← 이미지 파일 (docvqa/, infovqa/, slidevqa/ 등)
│   │
│   └── VDR_processed_filtered_2/
│       └── embeddings/colqwen2/      ← 877 임베딩 샤드 (~41GB)
│           ├── OpenDocVQA/           ← 473 shards
│           ├── VDR_ibm/              ← 200 shards
│           └── SlideVQA/             ← 204 shards
│
└── tests/LNS/                        ← LNS 전용 테스트 (129개, 전체 통과)
    ├── conftest.py
    ├── test_deck_lookup_on_cpu.py
    ├── test_deck_index_build_on_cpu.py  ← 덱 인덱스 빌드 검증 (26개)
    ├── test_ndcg_reward_on_cpu.py
    ├── test_parquet_rl_columns_on_cpu.py
    ├── test_search_tool_id_on_cpu.py
    ├── test_search_tool_unit_on_cpu.py
    └── test_retrieval_server_on_cpu.py  ← 검색 서버 테스트 (24개 CPU + 3개 통합)
```

### 코드 소유권 — 수정 가능 여부 판단 기준

| 파일 | 소유 | 수정 가능? |
|---|---|---|
| `verl/tools/LNS/*.py` | LNS Team (Copyright 2026) | **자유롭게 수정 가능** |
| `verl/utils/reward_score/format_ndcg_reward.py` | LNS Team (commit 6256c77에서 추가) | **자유롭게 수정 가능** |
| `data/Visual_Document_Rag/VDR_final/add_rl_columns.py` | 우리 (데이터 생성) | **자유롭게 수정 가능** |
| `data/Visual_Document_Rag/VDR_final/build_deck_index.py` | 우리 (인덱스 빌드) | **자유롭게 수정 가능** |
| `verl/utils/dataset/rl_dataset.py` | Bytedance/SGLang/ModelBest | **수정 금지** |
| `verl/experimental/agent_loop/tool_agent_loop.py` | Bytedance/SGLang/ModelBest | **수정 금지** |
| `verl/protocol.py`, `verl/__init__.py` 등 | verl 프레임워크 | **수정 금지** |

> 판단 기준: 파일 상단 Copyright 헤더를 확인한다. `LNS Team`이면 수정 가능, `Bytedance/SGLang/ModelBest`이면 수정 금지.

---

## 2. 데이터 파이프라인

### 2.1 train.parquet 스키마 (80,292행, 11컬럼)

```
원본 6개:  id, query, document_images, answer, gti, metadata
RL 5개:   prompt, images, reward_model, extra_info, data_source
```

### 2.2 핵심 데이터 흐름

```
add_rl_columns.py ──(1회 생성)──▶ train.parquet ──(매 스텝 읽기)──▶ rl_dataset.py
                                                                        │
                              extra_info에서 tools_kwargs 추출           │
                                                                        ▼
                              tool_agent_loop.py → tool.create(create_kwargs={...})
                                                        │
                                                        ▼
                              SearchTool.execute() → 서버 호출 → position 해석
```

### 2.3 extra_info 구조 (각 행마다)

```python
{
    "index": 0,                              # row_index (0~80291)
    "split": "train",
    "question": "What is the website?",
    "need_tools_kwargs": True,
    "reference_documents": ["infovqa/38032.jpeg"],  # GTI (정답 이미지)
    "document_images": ["infovqa/38032.jpeg", ...],  # 20장 덱
    "tools_kwargs": {
        "search": {
            "create_kwargs": {
                "row_index": 0,              # 서버에 전송할 정수 ID
                "document_images": [...],     # position→경로 변환용 덱
                "ground_truth": "answer text",
                "question": "What is the website?",
                "data_source": "VDR_lns",
            }
        }
    }
}
```

### 2.4 이미지 경로 형식

모든 이미지 경로는 **상대경로** (corpus/img/ 기준):
```
docvqa/hzym0020_14.png
infovqa/38032.jpeg
slidevqa/accel-deck_95/slide_4_1024.jpg
vdr/4f0d89780c8fb747ca03398424d76635f5f270da.png
coyo/sample1.jpg
chartqa/chart1.png
visualmrc/page1.png
mpmqa/doc1.png
openwikitable/table1.png
```

유효 소스 접두사: `docvqa/`, `infovqa/`, `slidevqa/`, `vdr/`, `coyo/`, `chartqa/`, `visualmrc/`, `mpmqa/`, `openwikitable/`

### 2.5 Parquet numpy 직렬화 주의사항

Parquet round-trip 시 dict 안의 Python list가 **numpy array로 변환**된다. 이는 정상 동작이며, 다운스트림 코드에서 문제없이 처리된다:
- `format_ndcg_reward._to_list()`: `.tolist()` 명시적 변환 (line 129-131)
- `search_tool.py`: 인덱싱만 사용 (`deck[position]`) — numpy/list 모두 동작
- `rl_dataset.py`: `.get()`으로 dict 추출만 함

---

## 3. 검색 파이프라인 (Position-Based Protocol)

### 3.1 기존 방식의 문제점

기존: `sample_id.split("_")[-1]`로 서버에 ID 전송
- **11,218개 ID 충돌**: `infovqa-train_1`, `docvqa-train_1`, `visualmrc-train_1`이 모두 `"1"`로 변환
- NDCG 정규화 버그: 모든 SlideVQA 슬라이드가 `"1024"`로 정규화 → 아무 슬라이드나 정답 처리

### 3.2 현재 방식 (Position-Based Protocol)

```
SearchTool                          Server
    │                                 │
    │  POST {"query": "...",          │
    │        "row_index": 42}         │
    │ ───────────────────────────────▶│
    │                                 │  deck_map[42] → 20장 이미지
    │                                 │  유사도 계산 → 순위 매기기
    │  {"results": [                  │
    │    {"position": 3, "score": 0.95},
    │    {"position": 7, "score": 0.82},
    │    ...                          │
    │  ]}                             │
    │◀─────────────────────────────── │
    │                                 │
    │  position 3 → document_images[3]
    │  → "slidevqa/deck/slide_4_1024.jpg"
    │  (상대경로를 image_paths에 저장)
```

### 3.3 SearchTool 핵심 메서드

```python
class SearchTool(BaseTool):
    async def create(self, instance_id=None, **kwargs):
        """create_kwargs에서 row_index, document_images를 인스턴스별로 저장."""
        create_kwargs = kwargs.get("create_kwargs", {})
        self._instance_data[instance_id] = {
            "row_index": create_kwargs.get("row_index"),
            "document_images": create_kwargs.get("document_images", []),
        }

    async def execute(self, instance_id, parameters, **kwargs):
        """row_index로 서버 호출 → position을 상대경로로 변환."""
        inst = self._instance_data.get(instance_id, {})
        row_index = inst.get("row_index")
        document_images = inst.get("document_images", [])

        payload = [{"query": query, "request_idx": 0, "row_index": row_index}]
        # ... HTTP 호출 ...
        # 서버 응답의 position → document_images[position] → 상대경로
        doc_id = document_images[position]
        image_paths_found.append(doc_id)  # 상대경로 저장
```

### 3.4 NDCG 보상 계산

```python
# format_ndcg_reward.py
def compute_score(data_source, solution_str, ground_truth, extra_info):
    format_score = simple_format_checker(...)     # 0.0 또는 1.0
    retrieved_norm = [_normalize_doc_id(x) for x in retrieved]
    reference_norm = [_normalize_doc_id(x) for x in reference]
    ndcg_value = _ndcg(retrieved_norm, reference_norm)
    final_score = 0.1 + 0.9 * ndcg_value         # format 10% + NDCG 90%
```

`_normalize_doc_id`: 확장자만 제거, 전체 경로 보존.
- `"slidevqa/deck/slide_4_1024.jpg"` → `"slidevqa/deck/slide_4_1024"` (고유)
- `"docvqa/hzym0020_14.png"` → `"docvqa/hzym0020_14"` (소스 구별 가능)

---

## 4. 테스트

### 4.1 실행 방법

```bash
# verl conda 환경에서 실행 (ray 등 의존성 필요)
conda run -n verl python -m pytest tests/LNS/ -v --tb=short

# 특정 파일만
conda run -n verl python -m pytest tests/LNS/test_ndcg_reward_on_cpu.py -v
```

**중요**: GPU를 사용하는 기존 프로세스가 실행 중일 수 있으므로, GPU를 사용하는 테스트나 프로세스를 실행하지 않는다.

### 4.2 테스트 파일별 역할 (총 129개)

| 파일 | 테스트 수 | 검증 대상 |
|---|---|---|
| `test_deck_lookup_on_cpu.py` | 12 | parquet 구조, 덱 구성(20장), 덱맵 생성, 이미지 파일 존재, 임베딩 커버리지 |
| `test_deck_index_build_on_cpu.py` | 26 | strip_ext() 정규화, 룩업 테이블 구축, VDR_ibm 매칭, 80,292행 커버리지, doc_id 보존, 샤드 크로스 검증 |
| `test_ndcg_reward_on_cpu.py` | 33 | `_normalize_doc_id` (SlideVQA 구별, 크로스소스 충돌), `_basename_no_ext`, `_ndcg`, `compute_score` E2E |
| `test_parquet_rl_columns_on_cpu.py` | 11 | extra_info 구조 (row_index, document_images, reference_documents), create_kwargs, 일관성 |
| `test_search_tool_id_on_cpu.py` | 14 | ID 충돌 증명 (`split("_")[-1]`), row_index 고유성, position 해석 개념, NDCG 파이프라인 |
| `test_search_tool_unit_on_cpu.py` | 9 | SearchTool.create() kwargs 저장, payload에 row_index 포함, position→상대경로 변환 |

### 4.3 conftest.py 주요 fixture

- `parquet_path`: `VDR_final/train.parquet` 경로 (없으면 skip)
- `corpus_img_root`: `VDR_final/corpus/img` 경로 (없으면 skip)
- `colqwen_row`: infovqa 기반 샘플 (GTI: `infovqa/38032.jpeg`)
- `slidevqa_row`: SlideVQA 샘플 (20장 슬라이드 덱, GTI: `slide_4_1024.jpg`)
- `vdr_row`: VDR 해시 기반 샘플

---

## 5. 훈련 설정

### 5.1 핵심 하이퍼파라미터

| 항목 | 값 |
|---|---|
| 모델 | Qwen2.5-VL-7B-Instruct |
| 알고리즘 | GRPO (GSPO loss mode) |
| 학습률 | 1e-6, warmup 28.5% |
| 롤아웃 | n=8 (프롬프트당 8개 응답) |
| 최대 턴 수 | 5 (assistant turns) |
| 최대 응답 길이 | 16384 tokens |
| 배치 크기 | train=64, val=32 |
| 보상 구성 | format(10%) + NDCG(90%) |
| Clip ratio | 0.0003 ~ 0.0004 |
| 저장 주기 | 100 steps |

### 5.2 실행 명령

```bash
cd /home/work/DDAI_revised/verl
bash RL_side_1_LNS/run_qwen2.5-3b_instruct_search_multiturn.sh
```

### 5.3 도구 설정 (`LNS_tool_config.yaml`)

```yaml
tools:
  - class_name: verl.tools.LNS.search_tool.SearchTool
    type: native
    config:
      retrieval_service_url: "http://127.0.0.1:10001/search"  # SSH 터널 경유
      timeout: 30
      num_workers: 120
      rate_limit: 120
      local_image_root: "./data/Visual_Document_Rag/VDR_final/corpus/img"

  - class_name: verl.tools.LNS.bbox_tool.ImageCropper
    type: native
    config:
      crops_dir: "./agent_crops"
```

> `local_image_root`는 SearchTool의 `project_root` (= `DDAI_revised/verl/`) 기준 상대경로로 resolve된다.
> 실제 해석: `/home/work/DDAI_revised/verl/data/Visual_Document_Rag/VDR_final/corpus/img` (94,076 이미지)

---

## 6. 임베딩 및 검색 서버

### 6.1 ColQwen2 임베딩 샤드

```
data/Visual_Document_Rag/VDR_processed_filtered_2/embeddings/colqwen2/
├── OpenDocVQA/    473 shards (docvqa, infovqa, coyo, chartqa, visualmrc, mpmqa, openwikitable)
├── VDR_ibm/       200 shards (vdr 해시 기반 이미지)
└── SlideVQA/      204 shards (slidevqa 슬라이드 이미지)
```

- 총 877 shards, 123,877개 고유 이미지
- 임베딩 차원: variable token length x 128 dim, bf16
- 디스크: ~41GB, 메모리: ~17.7GB

### 6.2 덱 인덱스 (`deck_index.pt`)

서버가 `row_index`를 받으면 해당 행의 20개 이미지 임베딩을 찾아야 한다.
임베딩은 877개 샤드에 흩어져 있으므로, **사전 구축된 역인덱스**로 O(1) 룩업을 제공한다.

**VDR_ibm 확장자 불일치 처리**:
- parquet의 doc_id: `vdr/HASH.png` (확장자 있음)
- 임베딩의 doc_id: `vdr/HASH` (확장자 없음)
- `strip_ext()`로 양쪽 모두 정규화하여 매칭, 원본도 보존

```python
deck_index = torch.load("VDR_final/deck_index.pt", map_location="cpu", weights_only=False)

# 구조:
{
    "deck_locations": [              # len=80,292, 각 원소 = 20개 튜플
        [(ds_idx, shard_idx, local_idx), ...],  # × 20
    ],
    "datasets": ["OpenDocVQA", "VDR_ibm", "SlideVQA"],
    "shard_paths": {                 # dataset별 샤드 파일명 목록
        "OpenDocVQA": ["shard_0000000.pt", ...],  # 473개
        "VDR_ibm":    ["shard_0000000.pt", ...],  # 200개
        "SlideVQA":   ["shard_0000000.pt", ...],  # 204개
    },
    "doc_ids_original":   [...],     # parquet 원본 doc_id (확장자 포함)
    "doc_ids_normalized": [...],     # 매칭용 (확장자 제거)
}
```

**사용법** — row_index=500의 3번째 이미지 임베딩 찾기:
```python
ds_idx, shard_idx, local_idx = deck_index["deck_locations"][500][3]
ds_name = deck_index["datasets"][ds_idx]                      # → "VDR_ibm"
shard_file = deck_index["shard_paths"][ds_name][shard_idx]    # → "shard_0000042.pt"
shard = torch.load(f"embeddings/colqwen2/{ds_name}/{shard_file}", ...)
embedding = shard["embeddings"][local_idx]                    # → Tensor[401, 128]
```

**빌드**: `conda run -n verl python VDR_final/build_deck_index.py`
- 877 샤드 스캔 → 224,013 유니크 임베딩 등록 → 80,292행 × 20 매핑 → 누락 0건
- 소요 시간: ~13초, 출력: 106.4 MB

### 6.3 서버 API (Position-Based Protocol)

**요청:**
```json
[{"query": "검색 쿼리", "request_idx": 0, "row_index": 42}]
```

**응답:**
```json
[{"results": [
    {"position": 3, "score": 0.95},
    {"position": 7, "score": 0.82},
    ...
]}]
```

서버(`retrieval_server.py`)는 startup 시:
1. `deck_index.pt` → row_index → 20개 임베딩 위치 매핑 로드 (~106 MB)
2. 877 임베딩 샤드 전수 로드 → CPU RAM (~35-38 GB)
3. ColQwen2 쿼리 인코더 → GPU (~4 GB)

> parquet을 직접 로드하지 않는다. `deck_index.pt`가 이미 모든 매핑을 포함한다.

---

## 7. 완료된 작업 및 버그 수정 이력

### 7.1 데이터셋 수정 (add_rl_columns.py)

**문제**: 기존 extra_info에 `document_images`, `row_index` 누락
**수정**: extra_info와 create_kwargs에 두 필드 추가
**결과**: train.parquet 69.2MB → 155.9MB (document_images 중복 저장)

### 7.2 NDCG 보상 함수 수정 (format_ndcg_reward.py)

**`_normalize_doc_id` 버그**:
- 기존: `split("_")[-1]`로 마지막 숫자만 추출 → SlideVQA 전체가 `"1024"`, 크로스소스 충돌
- 수정: 확장자만 제거, 전체 경로 보존

**`_basename_no_ext` 버그**:
- 기존: `os.path.basename()` + `.jpg`만 제거 → 디렉토리 소실, .png/.jpeg 미처리
- 수정: 전체 경로 보존, `.jpg`/`.jpeg`/`.png` 모두 처리

### 7.3 SearchTool 수정 (search_tool.py)

**ID 전송 방식**:
- 기존: `sample_id.split("_")[-1]` → 11,218개 충돌
- 수정: `create_kwargs["row_index"]` (정수 0~80291, 충돌 0)

**결과 해석 방식**:
- 기존: 서버가 `image_file` 경로 반환 → 절대경로로 변환
- 수정: 서버가 `position` (0-19) 반환 → `document_images[position]` 상대경로 저장

**경로 저장 형식**:
- 기존: `/home/work/.../corpus/img/docvqa/abc.png` (절대경로)
- 수정: `docvqa/abc.png` (상대경로, reference_documents와 동일 형식 → NDCG 비교 가능)

### 7.4 덱 인덱스 구축 (build_deck_index.py)

**문제**: 서버가 `row_index`를 받으면 20개 이미지 임베딩을 찾아야 하는데, 877개 샤드에 흩어져 있어 매번 전체 스캔이 필요함

**해결**: `build_deck_index.py`로 사전 역인덱스 구축
1. 877 샤드 전수 스캔 → normalized doc_id → (dataset_idx, shard_idx, local_idx) 룩업 테이블
2. train.parquet 80,292행 순회 → 행별 20개 이미지를 `strip_ext()`로 정규화하여 룩업
3. 검증: 누락 0건, 224,013개 유니크 임베딩 전부 매핑

**VDR_ibm 확장자 불일치 처리**:
- parquet: `vdr/HASH.png` → 임베딩: `vdr/HASH`
- `strip_ext()`가 양쪽 `.png`/`.jpg`/`.jpeg` 제거하여 통일
- `doc_ids_original`에 parquet 원본, `doc_ids_normalized`에 정규화 버전 모두 보존

**결과**: `deck_index.pt` (106.4 MB), 빌드 ~13초, 테스트 26개 전부 통과

### 7.5 설정 파일 경로 수정 (8만건 데이터셋 정합)

기존 설정 파일들은 `LNS/` 디렉토리와 `data/rag/` 디렉토리를 참조하고 있었으나,
실제 구조는 `RL_side_1_LNS/`와 `data/Visual_Document_Rag/VDR_final/`이다.
8만건 데이터셋으로 학습하기 위해 아래 3개 파일을 수정했다.

**`run_qwen2.5-3b_instruct_search_multiturn.sh`**:

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| TRAIN_DATA | `data/rag/slidevqa_train_6667.parquet` (미존재) | `data/Visual_Document_Rag/VDR_final/train.parquet` (80,292행) |
| VAL_DATA | `data/rag/overall_test_crop.parquet` (미존재) | `data/Visual_Document_Rag/VDR_final/train.parquet` (추후 val parquet으로 교체) |
| `--config-path` | `$PROJECT_DIR/LNS` (미존재) | `$PROJECT_DIR/RL_side_1_LNS` |

**`search_multiturn_grpo.yaml`**:

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| `tool_config_path` | `LNS/LNS_tool_config.yaml` (미존재) | `RL_side_1_LNS/LNS_tool_config.yaml` |

> `tool_config_path`는 `OmegaConf.load()`에 직접 전달되므로 CWD(`verl/`) 기준 상대경로다.

**`LNS_tool_config.yaml`**:

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| `local_image_root` | `./LNS/corpus/img` (미존재) | `./data/Visual_Document_Rag/VDR_final/corpus/img` (94,076 이미지) |

> `local_image_root`는 SearchTool의 `project_root`(`verl/`) 기준. `./`로 시작하면 `project_root + 나머지`로 resolve.

### 7.6 검색 서버 버그 수정 (retrieval_server.py)

**패딩 토큰 MaxSim 오염 (P1)**:
- ColQwen2는 배치 쿼리 인코딩 시 **left-padding** 사용 (짧은 쿼리 앞에 패드)
- 패딩 위치의 임베딩은 0이 아닌 **임의 값** → MaxSim 점수에 기여 → 절대 점수 왜곡
- 수정: `attention_mask`로 유효 토큰만 추출

```python
# 수정 전 (버그)
result.append(embeddings[i])           # 패딩 토큰 포함

# 수정 후
mask = attention_mask[i].bool()
result.append(embeddings[i][mask])     # 유효 토큰만
```

**`torch_dtype` deprecated (P2)**:
- transformers 4.56.1에서 `from_pretrained(torch_dtype=...)` 사용 시 경고
- `dtype=torch.bfloat16`으로 변경

---

## 8. 자주 발생하는 문제와 해결

### Q: 테스트에서 `ModuleNotFoundError: No module named 'ray'`
**A**: `conda run -n verl python -m pytest ...`로 verl 가상환경에서 실행해야 한다.

### Q: Parquet에서 읽은 list가 numpy array로 돼 있음
**A**: 정상 동작. Parquet round-trip의 특성이다. `_to_list()`이나 `.tolist()`로 변환하면 된다.

### Q: SlideVQA에서 아무 슬라이드나 검색해도 NDCG=1.0
**A**: `_normalize_doc_id` 버그. 이미 수정 완료. 수정 전 코드로 회귀하지 않도록 `test_wrong_slide_retrieved_scores_zero_ndcg` 테스트가 보호한다.

### Q: GPU 프로세스가 실행 중인데 테스트를 돌려도 되나?
**A**: `tests/LNS/` 테스트는 모두 CPU-only이다. GPU를 사용하지 않으므로 안전하다. 단, `conda run -n verl`에서 `import torch`가 일어나면 CUDA 초기화가 발생할 수 있으니, 명시적으로 `CUDA_VISIBLE_DEVICES=""`를 설정하는 것이 더 안전하다.

### Q: verl 프레임워크 코드를 수정해야 할 것 같은데?
**A**: Copyright 헤더를 확인한다. `Bytedance/SGLang/ModelBest`이면 수정 금지. 대신 우리 코드(LNS 전용)에서 우회하거나, parquet 데이터 포맷으로 해결한다.

---

## 9. 미완료 작업

- [x] ~~**덱 인덱스 사전 구축**: 877 shards를 단일 인덱스로 병합~~ → `deck_index.pt` 완성 (§7.4)
- [x] ~~**검색 서버 구축**: FastAPI 서버 — `deck_index.pt`를 활용한 position-based 검색 구현~~ → `retrieval_server.py` 완성 (§9.1)
- [ ] **vLLM 마이그레이션**: SGLang → vLLM fully async policy (ToDo/ 참조)
- [ ] **bbox_tool 테스트 추가**: 현재 단위 테스트 없음

### 9.1 검색 서버 (`retrieval_server.py`) — 완성됨

**파일**: `RL_side_1_LNS/retrieval_server.py`
**테스트**: `tests/LNS/test_retrieval_server_on_cpu.py` (24개 CPU 테스트 통과)

#### 아키텍처

```
                    120 concurrent requests
                            │
                    ┌───────▼────────┐
                    │  FastAPI /search│
                    │  (uvicorn+uvloop)│
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  MicroBatcher  │  ← asyncio.Queue, max_wait=20ms, max_batch=64
                    │  (수집 → 배치)  │
                    └───────┬────────┘
                            │
              ┌─────────────▼──────────────┐
              │  GPU: Batch Query Encoding  │  ← ColQwen2 model, H100
              │  N queries → [N, T, 128]   │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  Per-Query MaxSim Scoring   │  ← CPU에서 수행
              │  query [T,128] vs 20 docs   │
              │  → 20 scores → 정렬         │
              └─────────────┬──────────────┘
                            │
                    ┌───────▼────────┐
                    │  각 Future에    │
                    │  결과 반환      │
                    └────────────────┘
```

#### 클래스 구조

```
EmbeddingStore
├── __init__(deck_index_path, embedding_base)
│   └── _load_all_shards() → emb_store[ds_idx][shard_idx] = list[Tensor]
├── get_deck_embeddings(row_index) → list[Tensor] × 20
│   └── deck_locations[row_index] → 20개 (ds,sh,loc) → 20 Tensors
└── num_rows (속성)

QueryEncoder
├── __init__(model_name, device)
│   └── ColQwen2ForRetrieval + ColQwen2Processor 로드
├── encode_batch(queries: list[str]) → list[Tensor]
│   └── process_queries → model forward → embeddings
└── device, dtype (속성)

MaxSimScorer
├── score(query_emb, doc_embs) → list[(position, score)]
│   └── query @ doc.T → max(dim=1).values.sum() → 내림차순 정렬
└── score_batch(query_embs, doc_embs_list) → list[list[(pos, score)]]

MicroBatcher
├── __init__(encoder, scorer, store, max_batch=64, max_wait_ms=20)
├── submit(query, row_index) → Future → await → results
├── start(loop) → _batch_worker task 생성
└── _batch_worker() → 무한 루프: 수집 → encode → score → 분배

FastAPI App
├── startup: EmbeddingStore + QueryEncoder + MaxSimScorer + MicroBatcher 초기화
├── POST /search → payload 파싱 → batcher.submit() → 응답
└── GET /health → 상태 확인
```

#### 메모리 레이아웃 (Startup 시)

| 항목 | 위치 | 크기 |
|------|------|------|
| 877 샤드 임베딩 (전수) | CPU RAM | ~35-38 GB |
| deck_index.pt | CPU RAM | ~106 MB |
| ColQwen2 model (bf16) | GPU | ~4 GB |

#### 실행 방법

```bash
# H100 서버에서 실행
conda run -n verl python RL_side_1_LNS/retrieval_server.py

# 옵션
conda run -n verl python RL_side_1_LNS/retrieval_server.py \
    --port 5002 --device cuda:0 --max-batch 64 --max-wait-ms 20

# curl 테스트
curl -X POST http://localhost:5002/search \
    -H "Content-Type: application/json" \
    -d '[{"query":"What is the website?","request_idx":0,"row_index":0}]'
```

#### MaxSim 계산 (ColBERT-style late interaction)

```python
# query_emb: [T, 128] (쿼리 토큰별 임베딩, T≈30-50)
# doc_emb:   [P, 128] (이미지 패치별 임베딩, P≈700-780)
sim = query_emb @ doc_emb.T          # [T, P]
score = sim.max(dim=1).values.sum()   # scalar
# 20개 문서에 대해 반복 → 20 scores → 내림차순 정렬
```

#### 테스트

```bash
# CPU 테스트 (현재 서버에서 실행 가능, GPU 불필요)
CUDA_VISIBLE_DEVICES="" conda run -n verl python -m pytest tests/LNS/test_retrieval_server_on_cpu.py -v -k "not integration"

# 통합 테스트 (서버 실행 중일 때)
conda run -n verl python -m pytest tests/LNS/test_retrieval_server_on_cpu.py -v -k "integration"
```

| 테스트 클래스 | 테스트 수 | 검증 대상 |
|---|---|---|
| `TestEmbeddingStore` | 9 | deck_index 로드, row→20 텐서, 형상 [T,128], 경계 검증 |
| `TestMaxSimScorer` | 6 | MaxSim 정확성, 정렬, 가변 길이, 수동 계산 검증 |
| `TestRequestResponseFormat` | 6 | payload 파싱, 응답 포맷, position=int, score=float |
| `TestEdgeCases` | 3 | 빈 docs, 누락 query/row_index |
| `TestIntegration` | 3 | /health, E2E 검색, 마지막 row (서버 필요) |

---

## 10. 배포 아키텍처 및 데이터 정합성

### 10.1 2-머신 배포 구조

```
┌──────────────────────────────────┐    ┌─────────────────────────────────┐
│  학습 서버 (8-GPU)               │    │  검색 서버 (H100 × 2)           │
│                                  │    │                                 │
│  run_*.sh                        │    │  retrieval_server.py            │
│  ↓                               │    │  ├── EmbeddingStore (CPU RAM)   │
│  verl.trainer.main_ppo           │    │  │   ├── deck_index.pt (106 MB) │
│  ├── rl_dataset.py               │    │  │   └── 877 shards (~35 GB)    │
│  │   └── train.parquet (80,292)  │    │  ├── QueryEncoder (GPU, ~4 GB)  │
│  ├── tool_agent_loop.py          │    │  └── MicroBatcher               │
│  │   └── SearchTool 초기화       │    │      └── POST /search           │
│  └── SearchTool.execute()        │    │          port 5002              │
│      ├── POST → HTTP ─────────────────┤                                 │
│      │   row_index + query       │    │  ColQwen2 유사도 계산           │
│      ├── ← position + score ──────────┤                                 │
│      └── 이미지 로드 (로컬 파일) │    │                                 │
│          corpus/img/ (94,076)    │    │                                 │
│                                  │    │                                 │
│  CUDA_VISIBLE_DEVICES=0,...,7    │    │  --device cuda:0                │
└──────────────────────────────────┘    └─────────────────────────────────┘
                                   ↑
                        retrieval_service_url:
                        http://163.239.28.21:5002/search
```

- **학습 서버**: GPU 8개로 GRPO 학습, SearchTool은 HTTP 클라이언트로만 동작
- **검색 서버**: GPU 1개(쿼리 인코딩) + CPU RAM(임베딩 저장), 별도 머신에서 `retrieval_server.py` 실행
- 두 서버 모두 `DDAI_revised/` 폴더를 공유 (NFS 또는 동일 구조 복제)
- `retrieval_service_url`의 IP(`163.239.28.21`)가 검색 서버의 실제 IP와 일치해야 함

### 10.2 데이터 정합성 — 검증 완료

8만건 데이터셋(`VDR_final/train.parquet`)을 기준으로 전체 체인을 검증했다.

**parquet ↔ deck_index**:
- parquet 80,292행 = deck_index 80,292행: **일치**
- `row_index` 범위: 0~80,291, 전수 고유, 누락 0: **완벽 연속**
- parquet `document_images[row][pos]` = deck_index `doc_ids_original[row][pos]`: 7개 행 크로스 검증 **전부 일치**

**deck_index ↔ 임베딩 샤드**:
- `deck_locations[row][pos]` → `(ds_idx, shard_idx, local_idx)` → 실제 샤드 파일 존재: **확인**
- 임베딩 형상: `[T, 128]`, dtype=bf16, ragged list 형식: **정상**

**이미지 파일**:
- `corpus/img/` 아래 9개 서브디렉토리, 94,076개 파일
- row 0, 100, 50000, 80291의 20장 전부 존재: **확인**

**경로 resolve 체인** (수정 후):
```
run_*.sh
  TRAIN_DATA=$PROJECT_DIR/data/Visual_Document_Rag/VDR_final/train.parquet     ← 존재 ✓
  --config-path=$PROJECT_DIR/RL_side_1_LNS                                     ← 존재 ✓
    → search_multiturn_grpo.yaml                                               ← 존재 ✓
      tool_config_path: RL_side_1_LNS/LNS_tool_config.yaml                    ← 존재 ✓ (CWD 기준)
        retrieval_service_url: http://163.239.28.21:5002/search                ← 서버 port 5002 매칭 ✓
        local_image_root: ./data/Visual_Document_Rag/VDR_final/corpus/img      ← 존재 ✓ (project_root 기준)
```

### 10.3 데이터셋 상세 스펙

**`VDR_final/train.parquet`** (155.9 MB, 80,292행 × 11컬럼):

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `id` | str | 샘플 ID (`infovqa-train_1` 등) |
| `query` | str | 검색 질문 |
| `document_images` | list[str] × 20 | 덱 이미지 상대경로 |
| `answer` | str | 정답 텍스트 |
| `gti` | list[str] | Ground Truth Image (정답 이미지 경로) |
| `metadata` | dict | 원본 메타데이터 |
| `prompt` | list[dict] | 채팅 메시지 형식 프롬프트 |
| `images` | list | 프롬프트에 포함된 이미지 |
| `reward_model` | dict | 보상 함수 설정 |
| `extra_info` | dict | row_index, tools_kwargs, reference_documents 등 |
| `data_source` | str | `"VDR_lns"` |

**임베딩 샤드** (~41 GB 디스크, ~35 GB RAM):

| 데이터셋 | 샤드 수 | 이미지 소스 |
|----------|---------|------------|
| OpenDocVQA | 473 | docvqa, infovqa, coyo, chartqa, visualmrc, mpmqa, openwikitable |
| VDR_ibm | 200 | vdr 해시 기반 이미지 |
| SlideVQA | 204 | slidevqa 슬라이드 이미지 |

- 총 877 샤드, 224,013개 이미지 (중복 포함, 123,877개 고유)
- 샤드 형식: `{"doc_ids": [...], "embeddings": [Tensor, ...], "embeddings_format": "list", "dtype": "torch.bfloat16"}`
- 각 임베딩: `[T, 128]` (T = 패치 토큰 수, 대부분 401~780)
- 샤드 네이밍: `shard_{start:07d}.pt` (예: `shard_0000000.pt`, `shard_0000256.pt`)
- 샤드당 최대 256개 이미지

**`deck_index.pt`** (106.4 MB):

| 키 | 타입 | 설명 |
|---|---|---|
| `deck_locations` | list[list[tuple]] | 80,292행 × 20개 `(ds_idx, shard_idx, local_idx)` |
| `datasets` | list[str] | `["OpenDocVQA", "VDR_ibm", "SlideVQA"]` |
| `shard_paths` | dict[str, list[str]] | 데이터셋별 샤드 파일명 목록 |
| `doc_ids_original` | list[list[str]] | parquet 원본 doc_id (확장자 포함) |
| `doc_ids_normalized` | list[list[str]] | 매칭용 정규화 doc_id (확장자 제거) |

**`corpus/img/`** (94,076 파일):

| 서브디렉토리 | 예시 파일 |
|-------------|----------|
| `docvqa/` | `hzym0020_14.png` |
| `infovqa/` | `38032.jpeg` |
| `slidevqa/` | `accel-deck_95/slide_4_1024.jpg` (2-depth) |
| `vdr/` | `4f0d89780c8fb747ca03398424d76635f5f270da.png` |
| `coyo/` | `sample1.jpg` |
| `chartqa/` | `chart1.png` |
| `visualmrc/` | `page1.png` |
| `mpmqa/` | `doc1.png` |
| `openwikitable/` | `table1.png` |

### 10.4 ColQwen2 쿼리 인코딩 상세

- 모델: `vidore/colqwen2-v1.0-hf` (`ColQwen2ForRetrieval`, bf16, ~4 GB)
- 프로세서: `ColQwen2Processor` — `processor(text=queries, ...)` 호출 시 자동으로:
  - `"Query: "` 접두사 추가
  - 10개 `<|endoftext|>` augmentation 토큰 접미사 추가
  - 배치 시 **left-padding** (짧은 쿼리 왼쪽에 패드)
- `process_queries()`와 `__call__(text=...)` 는 동일 (내부에서 동일 코드 호출)
- 출력: `model(**inputs).embeddings` → `[batch, max_seq_len, 128]`
- **중요**: `attention_mask`로 패딩 토큰을 제거한 뒤 MaxSim에 사용해야 함 (§7.6)

### 10.5 심볼릭 링크 구조 (`data/` 디렉토리)

```
data/
├── VDR → Visual_Document_Rag/raw_datasets           (심볼릭 링크)
├── VDR_processed_filtered_2 → Visual_Document_Rag/VDR_processed_filtered_2  (심볼릭 링크)
└── Visual_Document_Rag/                              (실제 디렉토리)
    ├── VDR_final/
    │   ├── train.parquet       (80,292행, 155.9 MB)
    │   ├── deck_index.pt       (106.4 MB)
    │   ├── corpus/img/         (94,076 이미지)
    │   ├── add_rl_columns.py
    │   ├── build_deck_index.py
    │   └── assemble_dataset.py
    └── VDR_processed_filtered_2/
        └── embeddings/colqwen2/ (877 샤드, ~41 GB)
```

`retrieval_server.py`는 `VDR_final/deck_index.pt`와 `VDR_processed_filtered_2/embeddings/colqwen2/`를 직접 참조한다.
심볼릭 링크 `data/VDR_processed_filtered_2`를 통해서도 접근 가능하지만, 서버는 정규 경로를 사용한다.

### 10.6 남은 확인 사항

- [x] ~~`retrieval_service_url`의 IP 확인~~ → SSH 터널 방식으로 해결 (§11)
- [x] ~~H100 서버에서 `retrieval_server.py` 실제 구동 후 통합 테스트~~ → E2E + 부하 테스트 완료 (§11)
- [ ] VAL_DATA를 별도 검증 parquet으로 교체 (현재는 train과 동일하게 설정됨)

---

## 11. 검색 서버 배포 및 검증 결과 (2026-03-03)

### 11.1 네트워크 구성: SSH 터널 방식

검색 서버가 Docker 컨테이너 내부에서 실행되어 포트가 호스트로 노출되지 않았다.
직접 연결(`10.90.20.120:10001`, `163.239.28.21:10001`) 불가 → SSH 터널로 해결.

```
학습 서버 (main1)                    Docker 호스트               컨테이너 (bai-vscode-2)
                                     10.90.20.120:10522
127.0.0.1:10001  ── SSH tunnel ──▶  127.0.0.1:10001  ────▶  172.17.0.3:10001
     ↑                                                            ↑
  SearchTool                                              retrieval_server.py
  (RL 학습)                                               (ColQwen2, H100 GPU)
```

**터널 생성 명령** (학습 시작 전 실행):
```bash
ssh -fN -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -p 10522 -L 10001:127.0.0.1:10001 work@10.90.20.120
```

**터널 확인**:
```bash
curl -sS http://127.0.0.1:10001/health
# → {"status":"ok","rows":80292}
```

**터널 종료**:
```bash
pkill -f "ssh.*10001:127.0.0.1:10001"
```

**설정 변경** (`LNS_tool_config.yaml`):
```
수정 전: retrieval_service_url: http://163.239.28.21:5002/search
수정 후: retrieval_service_url: http://127.0.0.1:10001/search
```

### 11.2 E2E 검증 — 실제 RL 코드 흐름 재현

`search_tool.py`의 `call_search_api()` + `_normalize_doc_id()`를 직접 사용하여
train.parquet의 6개 행(첫행, 중간, 마지막)을 테스트했다.

**흐름**: parquet 로드 → extra_info에서 create_kwargs 추출 → payload 구성 → HTTP POST → 응답 파싱 → position→doc_id 변환 → 이미지 파일 존재 확인 → GT 매칭

| row_idx | 소스 | GT rank | score gap | 파일 존재 |
|---------|------|---------|-----------|----------|
| 0 | infovqa | **rank 1** | 9.0 | O |
| 100 | infovqa | **rank 1** | 8.6 | O |
| 500 | infovqa | rank 13 | 7.9 | O |
| 10000 | vdr | **rank 1** | 8.5 | O |
| 50000 | vdr | rank 7 | 10.6 | O |
| 80291 | slidevqa | **rank 1** | 10.5 | O |

- **6건 전부 성공**, 응답 50~80ms, 이미지 파일 전부 존재
- 6건 중 4건 Top-1 정답 (67%)
- infovqa, vdr, slidevqa 모든 소스 정상

### 11.3 부하 테스트 — RL 학습 규모 시뮬레이션

**RL 학습 설정 기준**:
- `train_batch_size=64`, `rollout.n=8`, `max_assistant_turns=5`
- 1 step 평균 검색: 64 × 8 × 2.5턴 = **1,280건**
- 1 step 최대 검색: 64 × 8 × 5턴 = **2,560건**
- `rate_limit=120` (동시 요청 제한)

**결과** (SSH 터널 경유, `localhost:10001`):

| 시나리오 | 요청 수 | Wall-clock | Throughput | p50 | p95 | 실패 |
|----------|---------|-----------|-----------|-----|-----|------|
| 단일 요청 | 1 | 52ms | - | - | - | 0 |
| 120 동시 | 200 | 0.66s | 305 req/s | 317ms | 380ms | 0 |
| **1 step (평균)** | **1,280** | **3.5s** | **369 req/s** | **310ms** | **365ms** | **0** |
| 1 step (최악) | 2,560 | 10.8s | 238 req/s | 336ms | 1,385ms | 0 |

**학습 영향 분석**:
```
1 step 전체 ~2-3분 기준:
  검색 3.5s (평균) = 전체의 ~2-3%  → 병목 아님
  검색 10.8s (최악) = 전체의 ~6-8% → 여전히 소수
  실패율 0/2,560 = 0%              → 안정적
```

> 최악 케이스의 p95=1.4s는 SSH 터널 하나에 120개 동시 연결이 집중되어 발생한다.
> Docker 포트 노출(직접 연결)로 전환하면 더 줄어들 것으로 예상.

### 11.4 학습 전 체크리스트

```
[x] 1. 검색 서버 기동 확인 (bai-vscode-2, port 10001, rows=80292)
[x] 2. SSH 터널 생성 (localhost:10001 → 원격 127.0.0.1:10001)
[x] 3. E2E 검증 통과 (6건, 실제 RL 코드 흐름)
[x] 4. 부하 테스트 통과 (2,560건, 0 실패, 10.8s)
[x] 5. LNS_tool_config.yaml URL 수정 (127.0.0.1:10001)
[ ] 6. 학습 시작 전 터널 생성 확인
[ ] 7. VAL_DATA parquet 교체 (현재 train 동일)
```
