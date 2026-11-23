# JF 샘플 - 식품 프로젝트 us (2026.11.11)

Journey Finder 식품 프로젝트 1개(us)의 결과입니다. 파일 구성과 컬럼 정의는 [뷰티/화장품 샘플](../beauty_cosmetics/README.md)과 같습니다.

- `tasks/s2.csv`·`s4.csv`·`s5.csv`: 프롬프트 입력 + 결과 컬럼, 한 행 = 키워드 하나
- `us/s1~s6_*.jsonl.gz`: 단계별 결과 전체
- `us/meta.json`: 시드, 단계별 stats, S3~S5 오류 건수(`errors`)

| 국가 | job_id | 끝난 시각(UTC) | 카테고리 | 시드 | S1 키워드 | S2 Span 키워드 | S4·S5 키워드 |
|---|---|---|---|---|---|---|---|
| us | `1d2d58f1-e5fe-40b6-a6be-36e81ac2deed` | 2026-11-11 16:39 | Greek Yogurt Snack Bars | clio snacks, clio greek yogurt bars, yogurt bars 외 7개 | 3,092 | 2,879 | 2,253 |

kr·jp 식품은 올리지 않았습니다. 일본은 S2 유형 번호(0~7)가 남은 식품 프로젝트가 없습니다.
us 식품 run 20개 중 이것을 고른 이유는 S3 오류가 6건으로 가장 적고 S4 오류가 0건이기 때문입니다.

## 다른 샘플과 다른 점

- **S3 오류가 6건 있습니다.** 그 6개 키워드는 S3 이후 결과(s3~s6 파일, `s4.csv`·`s5.csv`)에 없습니다. S2는 Span 키워드 전부에 유형 번호가 있습니다.
- **이 run은 AI overview를 우선하는 코드로 돌았습니다.** 이 규칙은 2026-11-11 09:03(UTC)에 worker 브랜치에 들어갔고, run은 16:20에 시작했습니다. 그래서 `serp`가 실제 프롬프트 입력과 거의 같습니다. 실제 프롬프트는 `AI overview (sources: …):` 머리줄이 붙고 최대 2,000자였습니다.
- **SERP가 모두 최근 것입니다.** 수집 시점이 2025년 이전인 행이 없습니다.
- `run_serp_query`(run이 다른 검색어의 SERP를 받은 키워드)는 0건입니다. SERP 키 1,662개가 모두 키워드 자신의 SERP였습니다.

## 행 수

| 파일 | 행 | AI overview | 상위 결과 |
|---|---|---|---|
| `s2.csv` | 2,879 | 2,083 | 796 |
| `s4.csv`·`s5.csv` | 2,253 | 1,662 | 591 |

S2 유형 분포:

| 유형 | us |
|---|---|
| 0 related | 2,106 |
| 1 other_category | 567 |
| 2 same_name_other_category | 77 |
| 3 hiring | 8 |
| 4 stock_finance | 11 |
| 5 news_pr | 9 |
| 6 person | 14 |
| 7 other_unrelated | 87 |
