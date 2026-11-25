# ds-decision-model

Journey Finder의 판단 단계(S2·S4·S5)를 운영 LLM(Gemma) 대신 작은 **결정모델**로 바꾸는 프로젝트입니다.
이 저장소 하나에 학습데이터 구축, 학습·실험, 결정 이력을 모두 둡니다. 작업을 시작하기 전에 이 파일과 [docs/decisions.md](docs/decisions.md)의 최근 항목을 읽습니다.

## 태스크

| 태스크 | 판정 | 라벨 | 기준서 |
|---|---|---|---|
| S2 Span 유무관 | 확장 키워드가 카테고리와 유관한지, 무관이면 유형 | 0 유관, 1~7 무관 유형 | [dataset/criteria/s2.md](dataset/criteria/s2.md) |
| S4 브랜드/논브랜드 | target이 브랜드인지 | brand, nonbrand | [dataset/criteria/s4.md](dataset/criteria/s4.md) |
| S5 CDJ 단계 | 키워드의 구매 여정 단계 | 브랜드 5단계 / 논브랜드 3단계 | [dataset/criteria/s5.md](dataset/criteria/s5.md) |

실험 모델은 `kev-4b`, `laya-multilingual`, `chon-1.0-e4b`, `chon-1.0-e2b`(B0만) 4개입니다. `clef-flash`, `kev-9b`는 설정·코드만 있고 쓰지 않았고, `gemma-4-26b-a4b`는 후보가 아니라 운영 Gemma 기준선(P0-*)입니다 ([train/README.md](train/README.md)의 모델 표).

언어는 ko·ja·en(파일·id에서는 kr·jp·us)입니다. 칸은 태스크 × 언어 9개, 칸당 1,680건(train 1,000 / calibration 214 / development 214 / test 252)입니다.

## 흐름과 폴더

```
── 학습데이터 구축 (dataset/) ────────────────────────────────────────────
① 기준 정의    dataset/criteria/*.md, dataset/criteria/sampling/*.yaml
② 표본 추출    dataset/tools/  (원천 수집은 서버 저장소 ds-journey-finder-async)
③ 모델 라벨    Gemma(운영 배치) + Claude(기준서로 전수 판정) → dataset/tools/
④ 사람 검수    불일치·보류 건만 → dataset/labels/{task}/
⑤ 마스터 고정  dataset/labels/{task}/ (라벨은 Git, 입력·SERP는 S3)
── 학습·평가 (train/) ────────────────────────────────────────────────────
⑥ 학습·평가    train/scripts/prepare_data.py → run.py / launch_sagemaker.py (train/src/decision_model/)
── 기록 ──────────────────────────────────────────────────────────────────
⑦ 기록         dataset/reports/ (데이터 작업 회차), experiments/ (학습 실험), docs/decisions.md (결정)
```

| 폴더 | 내용 |
|---|---|
| `dataset/` | 학습데이터 구축 전부 ([dataset/README.md](dataset/README.md)) |
| `dataset/criteria/` | 라벨링 기준서와 표본 기준. **기준의 원본** (Google Docs는 공유용 사본) |
| `dataset/labels/` | 태스크별 라벨과 manifest(S3 입력 위치·sha256) |
| `dataset/tools/` | 구축 도구: export → check → apply → validate → build-master |
| `dataset/reports/` | 데이터 작업 회차별 기록 (방법, 결과, 수정 목록, 팀이 정할 것) |
| `train/` | 학습·평가 파이프라인: `configs/`, `src/`, `scripts/`, `sagemaker/`, `tests/` ([train/README.md](train/README.md)). **명령은 `train/` 안에서 실행** |
| `experiments/` | 학습 실험 기록 |
| `docs/` | 결정 로그([decisions.md](docs/decisions.md)) |
| `jf_samples/` | 운영 Journey Finder 결과 샘플 (산업군별). `train/scripts/jf_to_master.py`로 마스터로 바꿔 A/B 비교에 씀 |

데이터 구축과 학습은 **마스터 파일 하나로만** 이어집니다: `dataset/tools`가 마스터를 만들고 `train/scripts/prepare_data.py`가 읽습니다.
`train/data/`(학습용 파일), `train/runs/`, `work/`는 Git에 넣지 않습니다(S3가 원본).
SageMaker 작업에는 `train/`만 올라가고, dirty 판정도 `train/`만 봅니다. `dataset/`을 고쳐도 학습 실행에는 영향이 없습니다.

## 지켜야 할 것

- **test 파티션은 최종 후보에 한 번만** 평가합니다(`--final`). 어떤 결정에도 쓰지 않습니다.
- dev·test의 Gemma 일치율 약 85%는 **설계값**입니다. Gemma 성능으로 인용하지 않습니다.
- 최종 라벨은 **사람 라벨 > 모델 라벨 일치 값**. 행마다 라벨 출처를 남깁니다.
- 브랜드·논브랜드 구분과 target·intent는 **라벨링 전에 확정**합니다(S5에서 늦게 고쳐 147건이 바뀜).
- Claude 판정 결과는 반드시 파일로 남깁니다. 근거 없는 라벨은 반영하지 않습니다.
- 기준을 바꾸면 기준서 통일 규칙 + `docs/decisions.md`에 같이 남기고, 그 기준에 걸리는 기존 일치 행을 다시 점검합니다.
- 모델 입력은 운영 추론과 같은 함수로 만듭니다. 선택지는 숫자가 아니라 "라벨 + 한 줄 정의"로 넣고, few-shot·soft label·출처별 가중치는 넣지 않습니다.
- 데이터 작업은 한 덩이(시트·칸·회차)를 끝낼 때마다 커밋합니다. 커밋 메시지에 무엇을 몇 건 바꿨는지 적습니다.

## 문서 쓰는 법

- 한국어, 결론 먼저, 짧고 쉬운 문장. 표를 적극적으로 씁니다.
- 날짜는 `YYYY-MM-DD`(문서 제목은 `(YYYY.MM.DD)`).
- Drive 문서를 근거로 쓸 때는 링크를 남깁니다.

## 외부 자료

- Drive 프로젝트 폴더: [결정모델 전환](https://drive.google.com/drive/folders/1w7Otw8HgTJp7Cqca0P2WlFGjXeE-POFQ) (00 리서치·배경 / 01 데이터 준비 / 02 학습·튜닝 / 03 평가 / 04 배포·관리)
- 실험 설계서: [결정모델 파인튜닝 (2026.11.16)](https://docs.google.com/document/d/1CQsWkZvSeysl-i-sbdY7u5qJDOJopjaNgnokIdAgZEc/edit)
- 모델 선정 기준: [결정 모델 선정 기준 (2026.11.17)](https://docs.google.com/document/d/1qnmq4TPnFhUlMApYhQlzwcSJCACz1MLPKJJavrvkCvY/edit)
- AWS: 서울 리전, 버킷 `ascentai-sagemaker-211132580794-ap-northeast-2-an`, 프리픽스 `ds-jf-decision-model/` ([train/README.md](train/README.md))
- 데이터 구축 원천: 서버 저장소 `ds-journey-finder-async`의 `cdj_dataset/`, `backend/scripts/build_cdj_*.py`
- 참고한 구조: `kj-hong-ascentko/aiov-tagging-dataset` (에이전트 라벨링 → check → apply → validate → 회차 보고서)
