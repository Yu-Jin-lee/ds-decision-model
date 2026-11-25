# dataset/tools - 학습데이터 구축 도구

Claude가 기준서대로 라벨을 붙이고, 도구가 그 결과를 기계적으로 검사한 뒤 반영합니다.
사람은 불일치·보류 건과 회차 보고서만 봅니다. 구조는 `kj-hong-ascentko/aiov-tagging-dataset`의 `tools/`를 따릅니다.

> 아직 도구는 없습니다. 아래는 만들 명령과 파일 형식입니다. 첫 대상(S4 남은 검수 또는 S2 불일치 검수)의 데이터를 가져오면서 구현합니다.

## 한 회차의 흐름

```
export ──▶ (Claude 에이전트가 덩이마다 라벨) ──▶ check ──▶ apply ──▶ validate ──▶ dataset/reports/ + 커밋
   │                                              │ ERROR가 0이 될 때까지 고침
   └─ work/<회차>/guide.md + chunk_XX.json         └─ 반영하지 않고 검사만
```

| 명령 | 하는 일 |
|---|---|
| `python dataset/tools/datagen.py status` | 태스크·칸별 진행률 (상태, 사람 검수 남은 건수) |
| `python dataset/tools/datagen.py export TASK --select disagree\|hold\|all --chunks N --out work/<회차>` | 기준서 전문(`guide.md`)과 덩이 파일 생성. 덩이에는 id, 입력(키워드·target·SERP 요약), 현재 라벨이 들어감 |
| `python dataset/tools/datagen.py check TASK work/<회차>/out_XX.json ...` | 쓰지 않고 검사만. 문제가 있으면 ERROR와 종료 코드 1 |
| `python dataset/tools/datagen.py apply TASK work/<회차>/out_XX.json ... --tag <태그>` | 검사를 통과하면 `dataset/labels/TASK/labels.jsonl`에 반영. 바꾼 행의 memo 끝에 `<태그>: <이유>` |
| `python dataset/tools/datagen.py validate TASK` | 전체 검사: 칸별 건수, 라벨 분포·몫, 쏠림 제한, dev·test Gemma 일치율, 같은 target의 라벨 충돌 |
| `python dataset/tools/datagen.py build-master TASK` | 라벨 + S3 입력 → 마스터 (`train/scripts/prepare_data.py` 입력) |

`work/`는 Git에 넣지 않습니다. 반영한 수정 목록은 `dataset/reports/<회차>/fixes/`로 옮겨 커밋합니다.

## 에이전트 출력 형식

```json
{
  "kr:에코백스 최저가": {"label": "confirmation", "hold": false, "memo": null},
  "kr:스티바 a": {"label": "browsing", "hold": false, "memo": "근거: SERP 1위가 스티바 a 제품 페이지 | 낯선 이름"},
  "kr:첫만남 어색": {"label": "initial_exploration", "hold": true, "memo": "보류: 소개팅앱 카테고리 엔트리 포인트인지 애매"}
}
```

## check가 잡는 것

- 덩이의 id가 **빠짐없이** 들어 있고, 덩이 밖 id는 없음
- 라벨 값이 태스크 라벨 집합 안에 있음 (S5는 브랜드 구분별 허용 라벨: 브랜드에 initial_exploration 금지 등)
- 보류(`hold: true`)면 memo에 이유가 있음
- 기존 라벨과 다르게 판정하면 memo에 근거가 있음
- memo의 따옴표 인용은 입력(키워드·SERP)에 글자 그대로 있음
- 기준서에 없는 규칙을 근거로 들면 경고 (통일 규칙으로 올릴지 사람이 판단)

## 회차 끝내기

1. `validate` 결과를 `dataset/reports/<YYYY-MM-DD-태스크-주제>/summary.txt`로 저장
2. 회차 README: 방법, 결과(바꾼 건수·분포 변화), 많이 고친 유형, **팀이 정할 것**
3. 새로 정한 규칙은 `dataset/criteria/<task>.md` 통일 규칙 + `docs/decisions.md`
4. 커밋: `S4 human review kr dev: 32 changes` 처럼 무엇을 몇 건 바꿨는지
