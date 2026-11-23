# JF 샘플 - 뷰티/화장품 프로젝트 국가별 1개 (2026.11.10)

Journey Finder 뷰티/화장품 프로젝트 3개(국가별 1개)의 단계별 결과를 **키워드 한 줄 = 한 행** JSONL.gz로 풀어 둔 것입니다.
S2 Span 판정 유형 번호(0~7)가 남아 있고, S3~S5 오류가 0건인 run만 골랐습니다.

| 국가 | job_id | 끝난 시각(UTC) | 카테고리 | 시드 | S1 키워드 | S2 유관 = S3~S6 키워드 |
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
- S2 유관 키워드 수 = S3~S6 키워드 수 (단계 요약의 `relevant_total`과 같음)

jp는 조건을 만족하는 뷰티/화장품 run이 이것 하나뿐입니다.
kr·us는 키워드가 더 많은 스킨케어 run도 있지만 S3·S4 오류가 있어 빼었습니다.

## 출처와 한계

- 단계별 캐시는 7일 뒤 지워져서 없습니다. **최종 산출물 `result.json.gz`에서 다시 풀어낸 것**입니다.
  - S1·S2는 `keyword_meta`에서 가져왔습니다(확장된 키워드 전체).
  - S3~S5는 `cdj_results`에서 가져왔습니다(S2 유관 키워드만). S3·S4 값은 S5까지 그대로 넘어온 값입니다.
  - S6은 `data.{단계}.tree`에서 가져왔습니다.
- 단계 캐시에만 있던 값은 없습니다. 예: SERP 스니펫, LLM 원문 응답, S2 유형 1·2·7에 붙던 설명(detail), CEP 트랙 판정 근거.
- S2 `relevant`는 `keyword_meta.rel`이 아니라 **S3 입력에 들어갔는지**로 정했습니다. CEP 트랙에서 같은 키워드가 두 번 판정되면 `keyword_meta.rel`에는 마지막 판정만 남습니다. 그래서 실제로는 유관으로 넘어간 키워드가 `false`로 보이는 경우가 있습니다(kr 8건, jp 2건). S3~S5 오류가 0건인 run이라 S3 입력 = S2 유관 목록입니다.
- 단계별 집계(stats)는 국가별 `meta.json`의 `steps`에 있습니다.

## 파일

`{kr,jp,us}/` 아래에 같은 이름으로 들어 있습니다. 모든 파일은 `keyword`로 이어 붙일 수 있습니다.

| 파일 | 행 | 필드 |
|---|---|---|
| `s1_keyword_expansion.jsonl.gz` | 확장 키워드 전체 | `volume` 월 검색량, `seeds` 이 키워드를 낳은 시드, `sources` 출처(`Seed`·`ES`·`graphDB`·`CEP`), `graph_hops` 시드별 그래프 홉(+ 정방향, - 역방향, 그래프 키워드만) |
| `s2_relevance.jsonl.gz` | 확장 키워드 전체 | `track` 판정 트랙(`span`·`cep`·`seed`), `relevant` 유관 여부, `type_num` Span 판정 유형(0~7, Span이 아니면 `null`) |
| `s3_attributes.jsonl.gz` | S2 유관 키워드 | `target`, `intent`, `site` |
| `s4_brand.jsonl.gz` | S2 유관 키워드 | `brand` 브랜드 키워드 여부, `brand_part` 브랜드 부분 |
| `s5_cdj.jsonl.gz` | S2 유관 키워드 | `brand`, `cdj_stage`, `stage_num` 모델 출력 번호, `error` |
| `s6_topics.jsonl.gz` | S2 유관 키워드 | `cdj_stage`, `topic_path` 루트→leaf 라벨, `cluster_id` leaf 노드 id, `method`(`llm`·`auto`) |
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

## 다시 만들기

```bash
# jf 서버에서, /data/prefect-results 읽기 권한 필요. S3~S5 오류가 있는 run은 거부한다.
python export_stages.py <출력 폴더> kr=<job_id> jp=<job_id> us=<job_id>
```

## 읽기

```python
import pandas as pd
s2 = pd.read_json("kr/s2_relevance.jsonl.gz", lines=True)
s4 = pd.read_json("kr/s4_brand.jsonl.gz", lines=True)
s5 = pd.read_json("kr/s5_cdj.jsonl.gz", lines=True)
df = s2.merge(s4, on="keyword", how="left").merge(s5.drop(columns="brand"), on="keyword", how="left")
```
