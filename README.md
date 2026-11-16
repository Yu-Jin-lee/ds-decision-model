# ds-decision-model

결정모델(Kev, Clef-Flash, Laya 등)을 S2·S4·S5 × 한국어·일본어·영어로 파인튜닝하고 평가하는 코드입니다.
실험 설계는 [실험 설계서 - 결정모델 파인튜닝 (2026.11.16)](https://docs.google.com/document/d/1CQsWkZvSeysl-i-sbdY7u5qJDOJopjaNgnokIdAgZEc/edit)을 따릅니다.

**무엇을 돌릴지는 `configs/`에, 어떻게 돌릴지는 코드에 둡니다.** 모델이나 실험군을 바꿀 때는 설정 파일만 고칩니다.

## 구조

```
configs/
  models/*.yaml     모델별 체크포인트·학습 레시피·인스턴스 (모델 추가 = 파일 하나 추가)
  tasks.yaml        태스크별 최종 라벨, 질문 형식(수준), 질문 문구
  groups.yaml       실험군(B0, S-*, J+R-*), seed, 1단계에서 고른 형식
src/decision_model/
  formats.py        질문 생성, 모델 답 → 최종 라벨 분포 변환 (모든 모델 공통)
  backends/         모델 계열별 학습·추론 (kev.py, clef.py, laya.py)
  calibration.py    temperature 하나를 calibration 파티션으로 맞춤
  metrics.py        정확도, Brier, ECE, 오류 5% 이내 자동 처리 비율, bootstrap
  evaluation.py     예측 → 칸별(태스크/언어/형식) 지표
  pipeline.py       한 실행: 학습 → 보정 → development 평가 → 저장
scripts/
  prepare_data.py   태스크 마스터 → data/{task}/{lang}/{partition}.jsonl
  run.py            실행 하나 (또는 --list로 실행 목록)
  launch_sagemaker.py  실행 목록을 SageMaker 학습 작업으로 제출
sagemaker/entry.py  학습 작업 안에서 의존성 설치 후 run.py 실행
```

## 사용법

```bash
# 1. 데이터: 마스터를 칸별 파일로 (분할은 마스터의 partition을 그대로 씀)
#    마스터 형식(라벨·브랜드 구분·입력 필드)은 configs/tasks.yaml의 태스크별 설정을 따름
python scripts/prepare_data.py --task s5 --master s5_master.jsonl.gz --out data

# 2. 실행 목록 확인 (1단계 27회)
python scripts/run.py --model clef-flash --list --stage 1

# 3. 실행 하나 (GPU 장비에서)
python scripts/run.py --model clef-flash --group S-s4-noul --seed 0

# 4. SageMaker로 한꺼번에 제출 (끝난 실행은 건너뜀)
python scripts/launch_sagemaker.py --model clef-flash --stage 1 \
  --data s3://BUCKET/ds-jf-decision-model/data/ --output s3://BUCKET/ds-jf-decision-model/runs/ \
  --role <SageMaker 실행 역할 ARN> --image-uri <학습 이미지> --dry-run

# 스모크 테스트: 실험군 하나, 시드 하나만 (--group은 S-s5처럼 묶음 이름, S-s5-choice처럼 하나만 지정 가능)
python scripts/launch_sagemaker.py --model laya --group B0-s5-choice --seed 0 ...
```

현재 AWS 설정(서울, ap-northeast-2):

| 항목 | 값 |
|---|---|
| 버킷 | `ascentai-sagemaker-211132580794-ap-northeast-2-an` (이름에 `sagemaker`가 있어야 실행 역할이 쓸 수 있음) |
| 실행 역할 | `arn:aws:iam::211132580794:role/custom-sagemaker` |
| 학습 이미지 | `763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch:2.14-cu133-amzn2023-sagemaker` |

실행 결과는 `runs/{모델}/{실험군}/s{seed}/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `summary.json` | 실행 설정, temperature, 칸별 지표, git commit, 데이터 sha256 |
| `predictions_development.jsonl` | 보정 전 질문별 log 확률 |
| `scored_development.jsonl` | 보정 후 최종 라벨 분포와 정답 |
| `checkpoint/` | 학습 결과물 (어댑터 + 헤드) |

test 파티션은 `--final`을 줄 때만 평가합니다. 최종 후보에 한 번만 씁니다.

## 모델 추가하기

1. `configs/models/<이름>.yaml`을 만듭니다. `backend`로 학습 방식을 고릅니다.
2. 기존 backend(kev, clef, laya)로 안 되는 모델이면 `src/decision_model/backends/`에 `train()`과 `predict()`를 구현한 클래스를 추가하고 `backends/__init__.py`에 등록합니다.
3. 데이터, 질문 형식, 보정, 지표는 그대로 재사용됩니다.

## 모델별 메모

**Clef-Flash** (`backends/clef.py`)
- Cloudflare는 추론 코드(`joint_schema_model.py`)만 공개했습니다. 이 코드로 입력을 만들고, 학습 루프는 여기서 작성했습니다.
- 백본은 고정하고 언어 모델의 모든 선형 층에 LoRA를 붙입니다(비전 인코더 제외). joint head는 전부 학습합니다.
- 공식 학습 레시피가 없어 `configs/models/clef-flash.yaml`의 값은 시작점입니다. 파일럿에서 development로 조정합니다.
- Clef는 한 레코드의 질문을 한 시퀀스에서 함께 판단합니다. Kev는 질문끼리 서로 보지 못합니다. "동시" 수준을 해석할 때 이 차이를 감안합니다.
- replay를 쓰는 실험군(J+R-*)은 Kev 형식의 범용 데이터 파일이 필요합니다. `data/replay.jsonl`에 두거나 `--replay-file`로 지정합니다.

**Laya** (`backends/laya.py`)
- Laya 패키지의 학습 함수(`laya.train.train_model`)를 그대로 호출합니다. 체크포인트는 다국어용 `multilingual`입니다.
- 기본값은 공식 레시피입니다: 인코더와 헤드 전체를 학습하고, 손실은 RLCD입니다.
- `configs/models/laya.yaml`에서 `freeze_encoder: true`(헤드만 학습), `loss: soft-ce`(cross-entropy만)로 바꿀 수 있습니다. LoRA는 지원하지 않습니다.
- Laya 자체의 calibration 분할과 temperature는 쓰지 않고, 다른 모델과 같은 공통 보정을 씁니다.
- 질문마다 별도 시퀀스로 판단합니다(Kev와 같고 Clef와 다름).
- 입력이 `max_len`을 넘거나 선택지가 헤드 예산(`head_max_len`)에 안 들어가는 질문은 학습에서 빠지고, 평가에서는 균등 확률로 처리되며 `predictions_*.jsonl`의 `skipped`에 기록됩니다.

**Kev** (`backends/kev.py`)
- Kev 저장소의 `kev.train`과 `kev.serve`를 그대로 호출합니다. `KEV_DIR`(기본 `/opt/kev`)에 저장소가 있어야 합니다.
- 평가는 보정 전 확률(`KEV_TEMPERATURE=1.0`)과 fp32 경로로 받고, 보정은 공통 코드에서 합니다.

## 아직 GPU에서 확인하지 않은 것

- Clef-Flash 학습은 CPU에서 가짜 모델로 흐름만 검증했습니다(`tests/test_clef_backend.py`). Laya는 laya 패키지 대신 가짜 함수로 선택지 순서와 변환만 검증했습니다(`tests/test_laya_backend.py`). 실제 모델로는 스모크 학습에서 메모리, 커널, bf16 autocast 동작을 확인해야 합니다.
- Kev 학습 옵션은 실험 설계서의 명령을 따랐습니다. Kev 저장소 버전에 맞는지 스모크 학습에서 확인합니다.
- `configs/tasks.yaml`의 질문 문구와 선택지 설명은 임시입니다. 운영에서 보낼 문구로 바꿔야 합니다.

## 테스트

```bash
pip install -e ".[dev,clef,laya]"
pytest
```
