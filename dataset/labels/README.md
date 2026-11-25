# dataset/labels - 태스크별 라벨

> 아직 `labels.jsonl`은 없습니다. 지금은 태스크별 `manifest.yaml`(상태와 서버 저장소의 원천 위치)만 있고, 첫 작업 회차에서 라벨을 가져오며 채웁니다.

라벨만 Git에 둡니다. 키워드 입력(SERP 포함)은 무거워서 S3에 두고, 각 태스크의 `manifest.yaml`에 위치와 sha256을 적습니다.
라벨이 Git에 있으니 누가 언제 무엇을 왜 바꿨는지 커밋으로 남습니다.

```
dataset/labels/{s2,s4,s5}/
  manifest.yaml      상태, 입력 위치(S3)·sha256, 원천 생성 커밋·날짜, 마스터 버전
  labels.jsonl       키워드 1행 (아래 형식). id 순으로 정렬해 diff를 작게 유지
```

## labels.jsonl 형식

```json
{"id": "kr:에코백스 최저가", "lang": "ko", "partition": "train",
 "industry": "가전제품", "category": "로봇청소기",
 "labels": {"gemma": "confirmation", "claude": "confirmation", "human": null},
 "claude_hold": false,
 "final": "confirmation", "final_source": "agree",
 "case": "representative", "memo": null}
```

| 필드 | 내용 |
|---|---|
| `id` | `국가:키워드` (kr·jp·us). 입력 파일과 이 값으로 잇습니다 |
| `lang`, `partition`, `industry`, `category` | 분할·분류 ([criteria/sampling/common.yaml](../criteria/sampling/common.yaml)) |
| `labels` | 출처별 라벨. 태스크에 따라 `rule`(규칙 사전)이 더 있을 수 있음 |
| `claude_hold` | Claude가 기준서로 답이 안 나와 보류했는지 |
| `final`, `final_source` | 최종 라벨과 출처: `human` / `agree` / `bulk`(기준 일괄 적용) |
| `case` | `representative` / `hard` / `failure` |
| `memo` | 판정 근거·애매한 이유. 사람이 고친 행은 반드시 채움 |
| (S5만) `kind`, `kind_changed` | 브랜드 구분(S4 결과)과 구분이 바뀌었는지 |

마스터(`train/scripts/prepare_data.py`의 입력)는 앞으로 `dataset/tools/build_master.py`가 이 라벨과 S3 입력을 합쳐 만듭니다. 지금까지의 마스터는 서버 저장소에서 만들었습니다.
지금 S5 마스터 필드 이름(`stage`, `stage_source`, `kind_source`)은 [train/configs/tasks.yaml](../../train/configs/tasks.yaml)이 읽는 이름이라 마스터에서는 그대로 둡니다.
