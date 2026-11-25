# dataset/reports - 데이터 작업 회차 기록

라벨링·검수·재검수 한 회차마다 폴더 하나를 만듭니다. 다음에 같은 작업을 할 때 이 기록이 출발점입니다.

```
dataset/reports/YYYY-MM-DD-<태스크>-<주제>/     예) 2026-11-26-s4-human-review
  README.md        방법, 결과, 많이 고친 유형, 팀이 정할 것
  summary.txt      datagen.py validate 결과
  fixes/*.json     반영한 수정 목록 (id, 이전 값, 새 값, 이유)
```

## README 틀

```markdown
# <태스크> <주제> (YYYY-MM-DD)

**결론:** 무엇을 몇 건 바꿨고, 지금 상태는 어떤지 한두 문장.

## 방법
대상(칸·파티션·선택 조건), 덩이 수, 사용한 기준서 버전(커밋), 사람이 본 범위.

## 결과
| 항목 | 전 | 후 |
바꾼 건수, 라벨 분포, Gemma 일치율(dev·test), 남은 보류.

## 많이 고친 유형
1. …

## 새로 정한 규칙
dataset/criteria/<task>.md 통일 규칙에 넣은 것.

## 팀이 정할 것
- …
```
