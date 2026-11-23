# JF 배치 샘플 - 뷰티/화장품 > 스킨케어 기초 (2026.11.07~08)

Journey Finder 배치 run 3개(국가별 1개)의 단계별 결과를 **키워드 한 줄 = 한 행** JSONL.gz로 풀어 둔 것입니다.

| 국가 | job_id | 끝난 시각(UTC) | 시드 | S1 키워드 | S3~S6 키워드 |
|---|---|---|---|---|---|
| kr | `254afd99-68f5-4983-9725-8b7495732fa6` | 2026-11-08 | 10 | 9,138 | 6,388 |
| jp | `ef8f8a56-d45e-4776-92d1-b7dfcc3f9f6f` | 2026-11-07 | 5 | 43,603 | 38,262 |
| us | `a397ec90-6692-423a-a02f-5cc703a9ecbb` | 2026-11-07 | 9 | 29,727 | 25,853 |

## 출처와 한계

- 단계별 캐시(`jf-cache/{job}/{n}.*.json.gz`)는 7일 뒤 지워져서 없습니다. **최종 산출물 `result.json.gz`에서 다시 풀어낸 것**입니다.
  - S1·S2는 `keyword_meta`에서 가져왔습니다(확장된 키워드 전체).
  - S3~S5는 `cdj_results`에서 가져왔습니다(S2 유관 키워드만). S3·S4 값은 S5까지 그대로 넘어온 값입니다.
  - S6은 `data.{단계}.tree`에서 가져왔습니다.
- 단계 캐시에만 있던 값은 없습니다. 예: S1 SERP 스니펫, 하위 단계 중간 결과, LLM 원문 응답.
- S2 Span 판정 유형 번호(`type_num`)는 이 run들 다음에 추가된 필드라 들어 있지 않습니다.
- 단계별 집계(stats)는 국가별 `meta.json`의 `steps`에 있습니다.

## 파일

`{kr,jp,us}/` 아래에 같은 이름으로 들어 있습니다. 모든 파일은 `keyword`로 이어 붙일 수 있습니다.

| 파일 | 행 | 필드 |
|---|---|---|
| `s1_keyword_expansion.jsonl.gz` | 확장 키워드 전체 | `volume` 월 검색량(ES `ads_metrics.volume_avg`), `seeds` 이 키워드를 낳은 시드, `sources` 출처(`Seed`·`ES`·`graphDB`·`CEP`), `graph_hops` 시드별 그래프 홉(+ 정방향, - 역방향, 그래프 키워드만) |
| `s2_relevance.jsonl.gz` | 확장 키워드 전체 | `relevant` 유관 여부 |
| `s3_attributes.jsonl.gz` | 유관 키워드 | `target`, `intent`, `site` |
| `s4_brand.jsonl.gz` | 유관 키워드 | `brand` 브랜드 키워드 여부, `brand_part` 브랜드 부분 |
| `s5_cdj.jsonl.gz` | 유관 키워드 | `cdj_stage`, `stage_num` 모델 출력 번호, `error` |
| `s6_topics.jsonl.gz` | 유관 키워드 | `cdj_stage`, `topic_path` 루트→leaf 라벨, `cluster_id` leaf 노드 id, `method`(`llm`·`auto`) |
| `meta.json` | - | 시드, 산업군·카테고리, 단계별 이름·끝난 시각·stats, 파일별 행 수 |

`stage_num`은 브랜드 여부에 따라 뜻이 다릅니다.

- 브랜드: 1=browsing, 2=experience, 3=confirmation, 4=own_and_service, 5=other
- 논브랜드: 1=initial_exploration, 2=own_and_service, 3=other

`other` 단계 키워드는 S6 클러스터링을 하지 않아 `topic_path`가 `null`입니다(kr 39, jp 85, us 281건).

## 다시 만들기

```bash
python export_stages.py <출력 폴더>   # jf 서버에서, /data/prefect-results 읽기 권한 필요
```

## 읽기

```python
import pandas as pd
s1 = pd.read_json("kr/s1_keyword_expansion.jsonl.gz", lines=True)
s5 = pd.read_json("kr/s5_cdj.jsonl.gz", lines=True)
df = s1.merge(s5, on="keyword", how="left")
```
