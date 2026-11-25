# dataset - 학습데이터·골든셋 구축

이 저장소가 받는 마스터(`s5_master` 등)가 어떻게 만들어졌는지, 앞으로 어떻게 만들지 정리합니다.

| 폴더 | 내용 |
|---|---|
| [criteria/](criteria/README.md) | 라벨링 기준서·표본 기준 (기준의 원본) |
| [labels/](labels/README.md) | 태스크별 라벨과 manifest (상태, S3 입력 위치) |
| [tools/](tools/README.md) | 구축 도구: export → check → apply → validate → build-master |
| [reports/](reports/README.md) | 데이터 작업 회차 기록 |

아래 흐름·설계 내용은 Drive 구축 방안 문서(2026.11.03~11.10)를 요약한 것입니다.

- 프로젝트 문서 폴더: [결정모델 전환](https://drive.google.com/drive/folders/1w7Otw8HgTJp7Cqca0P2WlFGjXeE-POFQ) (`00 리서치·배경` / `01 데이터 준비` / `02 학습·튜닝` / `03 평가` / `04 배포·관리`)
- 데이터 준비 문서 폴더: [01 데이터 준비](https://drive.google.com/drive/folders/1vRoyPoALbtpqE-IOwV_AsAKYe2wcYau2)

## 전체 흐름

모든 태스크가 같은 5단계를 거칩니다. 라벨은 Gemma 예측 → Claude 검수 → 사람 검수 순으로 확정합니다.

```
① 기준 정의 → ② 표본 추출 → ③ 모델 라벨 (Gemma 예측 + Claude 검수) → ④ 사람 검수 → ⑤ 마스터 고정
                                                                                  │
                                    이 저장소: train/scripts/prepare_data.py ◀────┘
                                    (마스터 → data/{task}/{lang}/{partition}.jsonl)
```

| 단계 | 할 일 | 산출물 |
|---|---|---|
| ① 기준 정의 | 라벨 기준과 경계 사례, 표본 기준과 제외 규칙 작성 | 라벨링 기준서, 표본 기준 |
| ② 표본 추출 | 표본 기준에 따라 원천에서 추출. 앞 단계(S3 추출, S4 구분)가 정확한 키워드만 넣거나 고쳐서 넣음 | 언어 × 파티션 표본 |
| ③ 모델 라벨 | Gemma(운영 배치 라벨)와 Claude(기준서로 전수 판정, 애매하면 보류) 비교 | 불일치 목록 |
| ④ 사람 검수 | 불일치·보류 건만 판정. 새 사례는 기준서에 추가 | 사람 라벨 |
| ⑤ 마스터 고정 | 최종 라벨을 키워드 1행으로 저장. 이후 마스터가 원본 | `cdj_dataset/{task}_master/` |

**최종 라벨 규칙**: 사람 라벨이 있으면 그 값, 없으면 모델 라벨이 일치하는 값. 행마다 라벨 출처(`stage_source`, `kind_source`)와 구분 변경 여부를 남깁니다.

S5 절차서는 다음 구축부터 **브랜드·논브랜드 구분과 target·intent를 라벨링 전에 확정**하도록 순서를 바꿨습니다(이번 구축에서 구분을 마지막에 고쳐 147건이 바뀌고 사람 라벨 35건을 다시 저장함).

## 공통 설계

### 데이터 분할 (2026.11.04 확정, 모든 태스크 공통)

처음 보는 산업군에서 성능을 재기 위해 산업군 단위로 나눕니다. 칸은 태스크(S2·S4·S5) × 언어(ko·ja·en) 9개이고, 칸당 약 1,680건입니다.

| 파티션 | 산업군 | 칸당 건수 | 용도 |
|---|---|---|---|
| train | 나머지 산업군 | 1,000 | 어댑터·헤드 학습 |
| calibration | train 산업군에서 카테고리 단위로 분리 | 214 | temperature만 맞춤 |
| development | 주류, 헬스케어 서비스, 뷰티 서비스, 출산/유아동, 의류 | 214 | 실험 비교, 설정 선택, ECE |
| test | 뷰티/화장품, 패션잡화/액세서리, 여행, 식품, 반려동물식품, 구독 서비스 | 252 | 채택 모델을 마지막에 한 번 평가 |

- 여러 산업군에 걸친 키워드는 test > development > calibration > train 순으로 배정합니다.
- 한 국가에서 10%를 넘는 산업군(자동차·은행·컴퓨터)과 카테고리가 하나뿐인 통신 서비스는 test·development에서 뺍니다.

### 사례 구성

| 사례 | 정의 | 비율 |
|---|---|---|
| 대표 | 기준표만 보고 누구나 같은 라벨을 고르는 일반적인 키워드 | 칸의 85% |
| 어려운 | 기준서 규칙만으로는 하나로 정해지지 않아 맥락·SERP·사람 검수로 확정 | 칸의 15% |
| 실패 | Gemma 오답 중 기준서로 답이 분명한 것 | dev·test에만, Gemma 일치율 약 85%가 되게 |

85%는 **설계값**입니다. Gemma의 실제 성능으로 인용하지 않습니다. 실패 사례는 한 종류가 칸 오답의 절반을 넘지 않게 섞습니다.

### 마스터 형태

| 필드 | 내용 |
|---|---|
| id | `국가:키워드` (예: `kr:아디 제로 evo sl 무게`) |
| 분할·분류 | lang, partition, industry, category |
| 입력 | keyword, target, intent, site, serp |
| 라벨 | 출처별 라벨 (gemma, claude, human) |
| 최종 라벨 | 최종 값과 출처 (S5: `stage`, `stage_source` / S4: `kind`, `kind_source`) |
| 메타 | 사례 유형(대표·어려운), 구분 변경 여부 |

- 학습 입력은 운영 추론과 같은 함수로 만듭니다 (`train/configs/tasks.yaml`의 `state` 규칙, TODO 남아 있음).
- 선택지는 숫자가 아니라 "라벨 + 한 줄 정의"로 넣습니다.
- few-shot 예시, soft label, 출처별 가중치는 넣지 않습니다.

## 태스크별 상태

상태는 Drive 문서 작성 시점 기준입니다. 이후 진행은 확인이 필요합니다.

| 태스크 | 판정 | 라벨 | 상태 (문서 기준) | 원천 데이터 위치 |
|---|---|---|---|---|
| S5 CDJ 단계 | 키워드의 구매 여정 단계 | 브랜드: Browsing·Experience·Confirmation·Own & Service·Other / 논브랜드: Initial Exploration·Own & Service·Other | **마스터 고정** (5,040건, 82%가 라벨 일치로 확정) | `cdj_dataset/s5_master/`, 표본 `cdj_dataset/tasks/s5_v2/` |
| S4 브랜드/논브랜드 | target이 브랜드인지 | brand · nonbrand | 표본 완료, Claude와 최종이 다른 143건 사람 검수 대기, `s4_master` 예정 (11.10) | `cdj_dataset/tasks/s4_v2/` (판정 규칙 `_claude/rules.md`) |
| S2 Span 유무관 | 확장 키워드가 카테고리와 유관한지, 무관이면 유형 | 0 유관 / 1 다른 카테고리 · 2 같은 이름 다른 카테고리 · 3 채용 · 4 주식·금융 · 5 뉴스·홍보 · 6 특정 인물 · 7 기타 무관 | 불일치·보류 건 사람 검수, `s2_master` 예정 (11.09) | `cdj_dataset/tasks/s2_v2/` |

태스크 사이의 의존:

- **S4**는 S5 마스터 키워드와 파티션을 그대로 쓰고, 정답은 S5 검수에서 고친 최종 구분(`kind`)입니다.
- **S2**는 S5의 산업군·카테고리 파티션을 그대로 씁니다. S4·S5에 있는 키워드는 그 파티션을 따릅니다.
- **S5**의 브랜드 구분은 S4 결과입니다. target·intent·site는 S3 추출 기준(S5 구축 방안 부록 2)대로 뽑힌 키워드만 씁니다.

## 코드와 데이터 위치

데이터 구축 코드와 산출물은 아직 이 저장소에 없고, 서버 저장소 `ds-journey-finder-async`에 있습니다.

| 무엇 | 위치 |
|---|---|
| 원천 수집 (키워드·SERP·Gemma 라벨) | `backend/scripts/build_cdj_dataset.py` → `cdj_dataset/master/` |
| 표본 추출 | `backend/scripts/build_cdj_task_sets.py`, `build_cdj_s5_v2.py` → `cdj_dataset/tasks/{task}_v2/` |
| Claude 검수 | `backend/scripts/cdj_claude_api.py` → `cdj_dataset/claude_api/` |
| 사람 라벨 | `cdj_dataset/human_label/{task}/{lang}/{partition}.jsonl` |
| 마스터 고정 | `cdj_dataset/s5_master/build_master.py` → `s5_master.jsonl.gz`, `.csv`, `README.md`, 스냅샷 |
| 분포 대시보드 | [S5](https://claude.ai/artifact/QXNXmYLPZ7bNTudhGkFmkU) · [S4](https://claude.ai/artifact/YM3D5NijW9iwQ8b5kV1x36) · [S2](https://claude.ai/artifact/A2RY4epjNdpMwsWBJ2kKHp) |

일부 스크립트는 서버 저장소에서도 커밋되지 않은 상태였습니다. 옮길 때 먼저 확인합니다.

## 남은 일 (문서 기준)

- S4: 143건 사람 검수, `s4_master` 고정
- S2: 불일치·보류 건 사람 검수, `s2_master` 고정
- S5: 기준서의 '인기' 기준 추가 여부, 규칙 사전에 사람 기준 반영, 브랜드 Other가 적은 파티션의 평가 방식
- 공통: Claude 판정 결과 파일을 데이터 폴더에 저장 (S5에서 근거 파일을 찾지 못함), 기준 변경 시 관련 일치 건 재점검

## 원본 문서

| 문서 | 링크 |
|---|---|
| 구축 방안 - 공통 (2026.11.03) | [Google Docs](https://docs.google.com/document/d/1kdauKD-YYUhlCytE3v87ObDVecs9ihWjf5NlKrfmUzI/edit) |
| 구축 방안 - S5 CDJ 단계 (2026.11.06) | [Google Docs](https://docs.google.com/document/d/1-cuM7QwLERE8usUZYxH6Fomov4vjuVNL8LfFaFqH8Uo/edit) |
| 구축 절차 - S5 CDJ 단계 (2026.11.06) | [Google Docs](https://docs.google.com/document/d/1VTIpuMH3Ua_rbfGTyJ7v35SfcRYUkVDKTgu4Y-RwPtI/edit) |
| 구축 방안 - S4 브랜드/논브랜드 (2026.11.10) | [Google Docs](https://docs.google.com/document/d/16m2NJAPJelQtV4s6gA0YsqphzEy0kckqGBmSuEVPbwE/edit) |
| 구축 방안 - S2 Span 유무관 유형 (2026.11.09) | [Google Docs](https://docs.google.com/document/d/1EMPjiprPBBQ2rpbEJXpNTbm9c-aciOtWhWXm8Gr-rgQ/edit) |
| 라벨링 기준 - S5 CDJ 단계 (2026.11.05) | [Google Docs](https://docs.google.com/document/d/1rngWXMHQEiS_sCOguHZk05VpVC9RSdBVYLbNr5Tq0wo/edit) |
| 라벨링 기준 - S4 브랜드/논브랜드 (2026.11.12) | [Google Docs](https://docs.google.com/document/d/1td2rgFSx30ayAQqbkGzGMcx6-sN2GBH8-LheGgTNp6I/edit) |
| 라벨링 기준 - S2 Span 유무관 유형 (2026.11.09) | [Google Docs](https://docs.google.com/document/d/1ggjEmH4GjI1izXib-qiEmgC0Gh8CdCo3-0FG_nk3FQE/edit) |
| S2 라벨 2 사례 - 같은 이름 다른 카테고리 브랜드 (2026.11.17) | [Google Docs](https://docs.google.com/document/d/1S15hw0LunWNyyJyEprEHJiCECJigqleE-ByaQRz6gsE/edit) |
