# JF 샘플 - 뷰티/화장품 프로젝트 국가별 1개 (2026.11.10)

Journey Finder 뷰티/화장품 프로젝트 3개(국가별 1개)의 단계별 결과를 **키워드 한 줄 = 한 행** JSONL.gz로 풀어 둔 것입니다.
S2 Span 판정 유형 번호(0~7)가 남아 있고, S3~S5 오류가 0건인 run만 골랐습니다.

다른 모델을 바로 돌려 볼 수 있게 **S2·S4·S5는 `tasks/`에 CSV로 따로 있습니다** — 프롬프트에 들어간 입력 컬럼 + 결과 컬럼([태스크 CSV](#태스크-csv-tasks)).

| 국가 | job_id | 끝난 시각(UTC) | 카테고리 | 시드 | S1 키워드 | S3~S6 키워드 |
|---|---|---|---|---|---|---|
| kr | `8f228742-7a57-4ed0-9cd2-4829decf5791` | 2026-11-10 09:25 | 남성 기초 스킨케어 | 남자 화장품 | 1,651 | 1,237 |
| jp | `06ab26ab-bcce-4911-a18e-4bcbf9922684` | 2026-11-10 04:11 | 韓国コスメのファンデーション | wakemake 외 4개 | 2,247 | 870 |
| us | `4a8c361b-6608-49a2-ad37-6e8d4c3b3f5d` | 2026-11-10 06:14 | Anti-Aging Skincare | antiaging | 754 | 474 |

배치가 아니라 수동으로 등록한 프로젝트입니다. 국가마다 카테고리와 시드가 달라 국가끼리 바로 비교되지는 않습니다.

## 고른 기준

0~7 유형 번호는 2026-11-09부터 저장됩니다. 그 뒤에 끝난 뷰티/화장품 run 중에서 아래를 모두 만족하는 것을 골랐습니다.

- run 상태 DONE, S2 LLM 실패 0건, Span `unknown` 0건
- Span 트랙 키워드 전부에 `type_num`이 있음 (kr 1,409 · jp 1,961 · us 388건)
- S3·S4·S5 오류 0건. 오류가 나면 그 키워드는 S3 이후 결과에서 빠집니다.

jp는 조건을 만족하는 뷰티/화장품 run이 이것 하나뿐입니다.
kr·us는 키워드가 더 많은 스킨케어 run도 있지만 S3·S4 오류가 있어 빼었습니다.

## 출처와 한계

- 단계별 캐시는 7일 뒤 지워져서 없습니다. **최종 산출물 `result.json.gz`에서 다시 풀어낸 것**입니다.
  - S1·S2는 `keyword_meta`에서 가져왔습니다(확장된 키워드 전체).
  - S3~S5는 `cdj_results`에서 가져왔습니다(S2에서 유관으로 넘어간 키워드만). S3·S4 값은 S5까지 그대로 넘어온 값입니다.
  - S6은 `data.{단계}.tree`에서 가져왔습니다.
- 단계 캐시에만 있던 값은 없습니다. 예: LLM 원문 응답, S2 유형 1·2·7에 붙던 설명(detail), CEP 트랙 판정 근거. SERP는 hbase에서 다시 찾아 `tasks/` CSV에만 넣었습니다.
- S2 `relevant`는 결과 파일에 저장된 `keyword_meta.rel` 그대로입니다. CEP 트랙은 같은 키워드를 상황·질문마다 따로 판정하는데, `rel`에는 마지막 판정만 남습니다. 그래서 `relevant`가 `false`인데 S3~S6에 있는 CEP 키워드가 있습니다(kr 8건, jp 2건). Span 트랙은 키워드마다 한 번만 판정해서 이런 일이 없습니다.
- 단계별 집계(stats)는 국가별 `meta.json`의 `steps`에 있습니다.

## 파일

`{kr,jp,us}/` 아래에 같은 이름으로 들어 있습니다. 모든 파일은 `keyword`로 이어 붙일 수 있습니다.

| 파일 | 행 | 필드 |
|---|---|---|
| `s1_keyword_expansion.jsonl.gz` | 확장 키워드 전체 | `volume` 월 검색량, `seeds` 이 키워드를 낳은 시드, `sources` 출처(`Seed`·`ES`·`graphDB`·`CEP`), `graph_hops` 시드별 그래프 홉(+ 정방향, - 역방향, 그래프 키워드만) |
| `s2_relevance.jsonl.gz` | 확장 키워드 전체 | `track` 판정 트랙(`span`·`cep`·`seed`), `relevant` 유관 여부, `type_num` Span 판정 유형(0~7, Span이 아니면 `null`) |
| `s3_attributes.jsonl.gz` | S3로 넘어간 키워드 | `target`, `intent`, `site` |
| `s4_brand.jsonl.gz` | S3로 넘어간 키워드 | `brand` 브랜드 키워드 여부, `brand_part` 브랜드 부분 |
| `s5_cdj.jsonl.gz` | S3로 넘어간 키워드 | `brand`, `cdj_stage`, `stage_num` 모델 출력 번호, `error` |
| `s6_topics.jsonl.gz` | S3로 넘어간 키워드 | `cdj_stage`, `topic_path` 루트→leaf 라벨, `cluster_id` leaf 노드 id, `method`(`llm`·`auto`) |
| `meta.json` | - | 시드, 산업군·카테고리, 단계별 이름·끝난 시각·stats, 파일별 행 수 |

### S2 `type_num` (Span 트랙)

`0`이면 유관, `1`~`7`이면 무관입니다. Span 트랙에서는 `relevant == (type_num == 0)`이 항상 맞습니다.

| 번호 | 뜻 | 예 (치약) |
|---|---|---|
| 0 | 유관 | 치약 추천, 2080 치약 성분 |
| 1 | 전혀 다른 시장 카테고리 | 자동차 왁스, 게임기 추천 |
| 2 | 관련 브랜드명은 있지만 다른 시장 카테고리(동명 업종·브랜드 포함) | 한우리 독서토론 |
| 3 | 채용·취업 | LG생활건강 채용 |
| 4 | 주식·금융 | 페리오 주가 |
| 5 | 뉴스·홍보 | 페리오 뉴스, 2080 광고 모델 |
| 6 | 특정 인물 | 페리오 대표이사, 치약 개발자 |
| 7 | 1~6에 없지만 구매 여정과 무관 | 치약 역사, 치약 공장 견학 |

| 국가 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| kr | 1,079 | 263 | 2 | 17 | 1 | 7 | 2 | 38 |
| jp | 726 | 1,189 | 7 | 3 | 0 | 9 | 20 | 7 |
| us | 188 | 152 | 2 | 0 | 0 | 2 | 6 | 38 |

CEP 트랙은 유관/무관만 남고 유형 번호가 없습니다.

### S5 `stage_num`

브랜드 여부에 따라 뜻이 다릅니다.

- 브랜드: 1=browsing, 2=experience, 3=confirmation, 4=own_and_service, 5=other
- 논브랜드: 1=initial_exploration, 2=own_and_service, 3=other

`other` 단계 키워드는 S6 클러스터링을 하지 않아 `topic_path`가 `null`입니다.

## 태스크 CSV (`tasks/`)

`s2.csv`·`s4.csv`·`s5.csv` 세 파일에 세 나라가 `country` 컬럼으로 함께 들어 있습니다(UTF-8 BOM, 엑셀에서 바로 열림).
컬럼은 **식별 → 프롬프트 입력 → SERP → 결과** 순서입니다. 목록 값(시드·target 등)은 프롬프트에 들어간 모양대로 `, `로 이었습니다.

| 파일 | 행 | 프롬프트 입력 | 결과 |
|---|---|---|---|
| `s2.csv` | Span 트랙 키워드 전부 (kr 1,409 · jp 1,961 · us 388) | `industry`, `category`, `market`(프롬프트 표기 `대분류 > 소분류/소분류`), `seeds`(프롬프트의 related entities = job 시드 전체), `keyword` | `type_num`(0~7), `type_label`, `relevant` |
| `s4.csv` | S3로 넘어간 키워드 (kr 1,237 · jp 870 · us 474) | `keyword`, `target`(소문자) | `llm_called`, `brand`, `brand_part` |
| `s5.csv` | S3로 넘어간 키워드 (위와 같음) | `keyword`, `brand`(S4 결과, 브랜드/논브랜드 프롬프트를 고름), `target`, `intent`, `site`, (`industry`, `category`는 참고용) | `cdj_stage`, `stage_num` |

세 파일 모두 SERP 컬럼 4개가 붙습니다.

| 컬럼 | 뜻 |
|---|---|
| `serp_top3` | 상위 3개 결과(featured snippet·organic)의 제목·스니펫, `[1] 제목\n스니펫` 형식 |
| `ai_overview` | AI overview 본문(`AI overview (sources: …):` 머리줄 포함), 없으면 빈 칸 |
| `serp_source` | SERP를 어디서 다시 찾았는지 (아래) |
| `serp_requested_at` | 그 SERP가 수집된 시각(UTC) |

이 run들(11-10)의 프롬프트에는 `serp_top3`만 들어갔습니다. AI overview가 있으면 그것만 넣는 규칙은 2026-11-11 코드부터입니다. 디시전 모델 설정(`configs/tasks.yaml`의 `serp_overview_or_top`)처럼 AI overview를 먼저 쓰려면 `ai_overview`가 있을 때 그것을, 없으면 `serp_top3`를 쓰면 됩니다.

### SERP를 다시 찾은 방법

run이 성공하면 SERP는 단계 파일에서 지워집니다. 그래서 hbase SERP 저장소에서 다시 열었습니다.

- `serp_key`: S4가 AI overview 집계용으로 남긴 키로 그 시점 SERP를 그대로 열었습니다. S3로 넘어간 키워드 중 AI overview가 있던 것입니다(kr 1,020 · jp 562 · us 335).
- `hbase_before_run`: 키가 없으면 **run 시작 시각 이전에 수집된 가장 최근 SERP**를 골랐습니다. S1은 hbase 최신본을 쓰기 때문입니다.
  - 이 규칙으로 고른 SERP가 저장된 키 1,917개와 전부 같았습니다.
  - S4·S5 행 중 이 방식으로 찾은 SERP에는 AI overview가 하나도 없습니다. S4가 "AI overview가 없었다"고 남긴 기록과도 맞습니다.
- SERP를 못 찾은 키워드는 없습니다.

S1이 hbase가 비어 수집 API로 직접 받은 SERP는 hbase에 남지 않습니다. 이 세 run에서는 그런 키워드가 없었습니다.

### 결과 컬럼 주의

- **S2**: CEP 트랙 키워드는 넣지 않았습니다(유형 번호가 없습니다). Span 트랙은 `relevant == (type_num == 0)`입니다.
- **S4**: `target`이 비면 LLM을 부르지 않고 논브랜드로 둡니다(`llm_called = false`). LLM 판정 뒤에는 **같은 target끼리 brand를 다수결로 맞춥니다**(동점이면 논브랜드). 그래서 `brand`는 LLM 원답이 아니라 이 보정까지 거친 최종값입니다. 원답은 단계 캐시와 함께 지워졌습니다.
- **S5**: `stage_num`은 `brand`에 따라 뜻이 다릅니다([S5 `stage_num`](#s5-stage_num)).

## 다시 만들기

```bash
# jf 서버에서, /data/prefect-results 읽기 권한 필요. S3~S5 오류가 있는 run은 거부한다.
python export_stages.py <출력 폴더> kr=<job_id> jp=<job_id> us=<job_id>

# 태스크 CSV — SERP 정리에 JF 백엔드의 app.serp를 쓴다 (그 venv의 python으로)
python export_task_csv.py tasks --jf-backend <ds-journey-finder-async/backend> kr=<job_id> jp=<job_id> us=<job_id>
```

## 읽기

```python
import pandas as pd
s5 = pd.read_csv("tasks/s5.csv", keep_default_na=False)   # 빈 칸을 NaN 대신 ""로
s2 = pd.read_json("kr/s2_relevance.jsonl.gz", lines=True)
s4 = pd.read_json("kr/s4_brand.jsonl.gz", lines=True)
s5 = pd.read_json("kr/s5_cdj.jsonl.gz", lines=True)
df = s2.merge(s4, on="keyword", how="left").merge(s5.drop(columns="brand"), on="keyword", how="left")
```
