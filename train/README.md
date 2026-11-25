# train - 결정모델 학습·평가

결정모델(Kev, Laya, Chon)을 S2·S4·S5 × 한국어·일본어·영어로 파인튜닝하고 평가하는 코드입니다.
**아래 명령은 모두 이 `train/` 폴더에서 실행합니다** (`cd train`). `data/`, `runs/`도 이 폴더 아래에 생깁니다.
실험 설계는 [실험 설계서 - 결정모델 파인튜닝 (2026.11.16)](https://docs.google.com/document/d/1CQsWkZvSeysl-i-sbdY7u5qJDOJopjaNgnokIdAgZEc/edit)을 따릅니다.

**무엇을 돌릴지는 `configs/`에, 어떻게 돌릴지는 코드에 둡니다.** 모델이나 실험군을 바꿀 때는 설정 파일만 고칩니다.

## 모델

결정모델 후보로 실험한 모델은 아래 4개입니다. `clef-flash`와 `kev-9b`는 설정과 코드만 있고 돌리지 않았습니다. `gemma-4-26b-a4b`는 후보가 아니라 비교 기준선입니다.

| 모델 | 백엔드 | 실험 |
|---|---|---|
| `kev-4b` | kev | 사용 |
| `laya-multilingual` | laya | 사용 |
| `chon-1.0-e4b` | chon | 사용 |
| `chon-1.0-e2b` | chon | 사용 (B0만) |
| `clef-flash` | clef | **미사용**. 설정·백엔드 코드만 있음 |
| `kev-9b` | kev | **미사용**. 설정만 있음 (`base_revision` 미정) |
| `gemma-4-26b-a4b` | prompt_llm | **결정모델 후보 아님**. 운영 Gemma 기준선(P0-*). 평가·serving 실행 기록이 커밋에 있음 |

## 구조

```
configs/
  models/*.yaml     모델별 체크포인트·학습 레시피·인스턴스·serving 설정 (모델 추가 = 파일 하나 추가)
  tasks.yaml        태스크별 최종 라벨, 모델 입력(state) 조립 규칙, 질문 형식(수준), 질문 문구
  groups.yaml       실험군과 seed (아래 표)
  serving.yaml      속도 비교용 serving 프로필 (인스턴스, vLLM 버전)
  prompts/          Gemma 기준선(P0-*)이 쓰는 운영 프롬프트
src/decision_model/
  config.py         설정 읽기, 실험군 → 실행 목록 확장
  prepare.py        태스크 마스터 행 → 표준 레코드
  data.py           칸별 데이터 읽기·쓰기, 데이터 파일 sha256
  inputs.py         원본 입력 → 모델 입력(state) 조립 (SERP 형식 등)
  formats.py        질문 생성, 모델 답 → 최종 라벨 분포 변환 (모든 모델 공통)
  backends/         모델 계열별 학습·추론 (kev.py, laya.py, chon.py / prompt_llm.py: Gemma 기준선 / 미사용: clef.py)
  calibration.py    temperature 하나를 calibration 파티션으로 맞춤
  metrics.py        정확도, Brier, ECE, 오류 5% 이내 자동 처리 비율, bootstrap
  evaluation.py     예측 → 칸별(태스크/언어/형식) 지표
  latency.py        모든 모델 같은 조건(칸별 30건, 배치 1, 워밍업 10건)으로 추론 속도 측정
  pipeline.py       한 실행: 학습 → 보정 → development 평가 → 저장
scripts/
  prepare_data.py   태스크 마스터 → data/{task}/{lang}/{partition}.jsonl
  run.py            실행 하나 (또는 --list로 실행 목록)
  launch_sagemaker.py  실행 목록을 SageMaker 학습 작업으로 제출
  jf_to_master.py   jf_samples 태스크 CSV → 태스크 마스터 (A/B용)
  ab_launch.py      끝난 실행을 바깥 데이터로 채점하는 SageMaker 작업 제출
  ab_job.py         A/B 작업 안에서 예측·채점
  reshard_safetensors.py  Gemma 체크포인트를 4GB 파일로 다시 나눔 (serving 전)
sagemaker/entry.py  학습 작업 안에서 의존성 설치 후 run.py 실행
```

### 실험군 (`configs/groups.yaml`)

| 실험군 | 단계 | 내용 |
|---|---|---|
| `B0-{태스크}-{형식}` | 0 | 학습 없이 공개본 그대로, 태스크의 모든 질문 형식 |
| `P0-s2`, `P0-s4`, `P0-s5` | 0 (선택) | Gemma 기준선: 운영 프롬프트로 생성한 답을 라벨로 변환 |
| `S-s2-*`, `S-s4-*`, `S-s5-*` | 1 | 태스크 하나씩 질문 형식 비교 (s2·s5: choice/score/both, s4: noul/choice/both, `S-s4-score`는 선택) |
| `L-{태스크}-{언어}` | 2 (선택) | 한 언어로만 학습(choice, replay 없음), 세 언어로 평가. seed 0만 |
| `J-best` | 2 (선택) | 세 태스크를 한 모델로, 1단계에서 고른 형식(s2 choice, s4 both, s5 choice), replay 없음. seed 0만 |
| `J+R-all`, `J+R-ko/ja/en`, `J` | 2 | 실험 설계서의 2단계(replay 2000, `J`는 replay 없음). `selected_levels`를 채워야 실행 목록이 만들어짐 |

- 1단계에서 고른 형식은 `selected_levels`가 아니라 `J-best`에 직접 적혀 있습니다. `selected_levels`는 비어 있어서 `J+R-*`와 `J`는 지금 설정으로는 돌릴 수 없습니다.
- 2단계는 Kev-4B와 Laya가 `L-*`와 `J-best`로 돌렸습니다(커밋 205b301 기록). Chon은 replay를 지원하지 않아 `J+R-*`를 돌릴 수 없습니다.

## 사용법

```bash
# 1. 데이터: 마스터를 칸별 파일로 (분할은 마스터의 partition을 그대로 씀)
#    마스터 형식(라벨·브랜드 구분·입력 필드)은 configs/tasks.yaml의 태스크별 설정을 따름
python scripts/prepare_data.py --task s5 --master s5_master.jsonl.gz --out data
#    데이터에는 입력 필드를 원본 그대로(inputs) 저장하고, 모델 입력(state)은 학습 때
#    tasks.yaml의 state 규칙으로 조립함. SERP 형식 등을 바꿔도 데이터를 다시 올릴 필요 없음

# 2. 실행 목록 확인 (1단계 27회)
python scripts/run.py --model kev-4b --list --stage 1

# 3. 실행 하나 (GPU 장비에서)
python scripts/run.py --model kev-4b --group S-s4-noul --seed 0

# 4. SageMaker로 한꺼번에 제출 (끝난 실행은 건너뜀)
python scripts/launch_sagemaker.py --model kev-4b --stage 1 \
  --data s3://BUCKET/ds-jf-decision-model/data/ --output s3://BUCKET/ds-jf-decision-model/runs/ \
  --role <SageMaker 실행 역할 ARN> --image-uri <학습 이미지> --dry-run

# 스모크 테스트: 실험군 하나, 시드 하나만 (--group은 S-s5처럼 묶음 이름, S-s5-choice처럼 하나만 지정 가능)
python scripts/launch_sagemaker.py --model laya-multilingual --group B0-s5-choice --seed 0 ...
```

현재 AWS 설정(서울, ap-northeast-2):

| 항목 | 값 |
|---|---|
| 버킷 | `ascentai-sagemaker-211132580794-ap-northeast-2-an` (이름에 `sagemaker`가 있어야 실행 역할이 쓸 수 있음) |
| 실행 역할 | `arn:aws:iam::211132580794:role/custom-sagemaker` |
| 학습 이미지 | `763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch:2.14-cu133-amzn2023-sagemaker` |
| 비용 태그 | `project=ds-jf-decision-model`, `team=ds` (학습 작업은 런처가 자동으로, 버킷은 버킷 태그로 붙음. `--tag key=value`로 추가) |

실행 결과는 `runs/{모델}/{실험군}/s{seed}/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `summary.json` | 실행 설정, temperature, 칸별 지표, git commit, 데이터 sha256 |
| `model_inputs_{train,calibration,development}.jsonl.gz` | 모델이 실제로 받은 입력 그대로 (조립된 state, 질문, 정답). 해시는 `summary.json`의 `model_inputs_sha256` |
| `predictions_development.jsonl` | 보정 전 질문별 log 확률 |
| `scored_development.jsonl` | 보정 후 최종 라벨 분포와 정답 |
| `checkpoint/` | 학습 결과물 (어댑터 + 헤드) |

test 파티션은 `--final`을 줄 때만 평가합니다. 최종 후보에 한 번만 씁니다.

### 학습은 끝났는데 그 뒤에 실패한 실행 이어서 돌리기

학습 작업이 실패해도 SageMaker는 `/opt/ml/model`을 `_jobs/<작업 이름>/output/model.tar.gz`로 올립니다.
학습을 마친 실행이면 그 안에 `{모델}/{실험군}/s{seed}/checkpoint/checkpoint/`가 남아 있으니, 학습은 건너뛰고 보정·평가·저장만 다시 돌립니다.

```bash
# SageMaker: 이전 작업의 model.tar.gz에 체크포인트가 있는 실행은 학습을 건너뜀 (없는 실행은 처음부터 학습)
python scripts/launch_sagemaker.py --model kev-4b --group J-best --seed 0 --batch \
  --resume-from s3://BUCKET/ds-jf-decision-model/runs/_jobs/<실패한 작업 이름>/output/model.tar.gz ...

# 로컬: 체크포인트 폴더를 직접 지정
python scripts/run.py --model kev-4b --group J-best --seed 0 --checkpoint <압축 푼 폴더>/kev-4b/J-best/s0/checkpoint/checkpoint
```

체크포인트는 새 실행 폴더의 `checkpoint/checkpoint/`로 복사되어 함께 올라가고, `summary.json`의 `resumed_from`에 원래 위치가 남습니다.

### 속도 비교: serving 프로필로 다시 재기

평가 실행은 정확도를 위해 모델마다 정확한 경로로 읽습니다(Kev fp32, Laya fp32).
이 설정 그대로 잰 속도는 모델끼리 비교할 수 없습니다. `--serving`은 끝난 실행을 **운영에서 서빙하는 방식**으로 다시 잽니다.

| 조건 | 모든 모델 공통 (`configs/serving.yaml`) |
|---|---|
| 인스턴스 | `ml.g6e.xlarge` (L40S 48GB 1장), 평가와 같은 인스턴스 |
| 정밀도 | bf16 계산. Gemma(기준선)만 bf16 가중치(~52GB)가 48GB에 안 들어가 가중치를 int8로 둠(weight-only, 계산은 bf16) |
| 측정 | 같은 레코드(칸별 30건), 배치 1, 순서대로. 워밍업은 측정 표본 밖의 레코드 10건 |

| 모델 | 서빙 경로 (`configs/models/*.yaml`의 `serving`) |
|---|---|
| Chon | `chon.serve --reader vllm` (어댑터 병합 후 vLLM으로 선택지 글자를 읽음, Gemma와 같은 vLLM 버전) |
| Kev | `kev.serve` 기본 서빙 설정: bf16, CUDA graph, fused Qwen3.5 커널(`flash-linear-attention` 설치) |
| Laya | bf16 autocast |
| Gemma (기준선) | vLLM, 평가와 같은 플래그 (int8 가중치, bf16 계산) |

```bash
# 끝난 실행(B0는 공개본, 학습한 실행은 --output의 체크포인트)을 serving 프로필로 다시 잼
python scripts/launch_sagemaker.py --model kev-4b --group B0-s5-choice --group S-s5-choice --seed 0 \
  --batch --serving --data ... --output s3://BUCKET/ds-jf-decision-model/runs/ --role ... --image-uri ...
```

결과는 `runs/serving/{모델}/{실험군}/s{seed}/`에 따로 저장되고 평가 실행은 건드리지 않습니다.
`summary.json`의 `latency_development`가 속도이고, `metrics_development`는 serving 경로의 정확도입니다(평가 경로와 답이 같은지 확인용).
Chon의 vLLM이나 Kev의 fused 커널이 뜨지 않으면 조용히 느린 경로로 재지 않고 실행이 실패합니다.
serving 배치는 학습한 실행이 끝날 때마다 받은 체크포인트와 Chon이 병합한 모델을 지웁니다(병합본은 어댑터마다 전체 모델 하나라 쌓이면 디스크가 찹니다).

### A/B: 실제 데이터에서 기존 결과와 비교

끝난 실행 하나의 모델을 바깥 데이터(예: `jf_samples/`)로 돌려, 그 데이터의 기존 결과(운영 Gemma 판정)와 얼마나 같은지 봅니다.
지시문·선택지·라벨은 그 실행의 태스크 형식(`configs/tasks.yaml`), 입력 형식은 학습 데이터와 같고, 체크포인트와 보정 온도는 그 실행의 `summary.json`에서 읽습니다.
라벨이 사람 정답이 아니라 기존 시스템의 답이라, 정확도는 **일치율**입니다.

```bash
python scripts/jf_to_master.py --task s5 --csv ../jf_samples/beauty_cosmetics/tasks/s5.csv --country kr --out jf_s5_kr.jsonl.gz
python scripts/prepare_data.py --task s5 --master jf_s5_kr.jsonl.gz --out data_ab/jf-beauty-kr-s5
aws s3 sync data_ab/jf-beauty-kr-s5 s3://BUCKET/ds-jf-decision-model/data-ab/jf-beauty-kr-s5/
python scripts/ab_launch.py --model chon-1.0-e4b --group J-best --seed 0 --task s5 --lang ko --name jf-beauty-kr-s5 \
  --data s3://BUCKET/ds-jf-decision-model/data-ab/jf-beauty-kr-s5/ --output s3://BUCKET/ds-jf-decision-model/runs/ \
  --role ... --image-uri ... --spot
```

여러 태스크·언어는 한 작업에 넣으면(`--task s2,s4,s5 --lang ko,ja,en`) 모델을 한 번만 띄워 모두 예측하고, 태스크·언어마다 따로 채점해 저장합니다.
결과는 `runs/ab/{name}/{task}/{lang}/{run_id}/`: `summary.json`(일치율), `keywords.csv`(키워드별 기존 라벨·모델 답·확률·일치), `scored_development.jsonl`.
`--spot`은 따로 잡힌 스팟 한도를 써서 온디맨드 작업과 동시에 돕니다.

## 모델 추가하기

1. `configs/models/<이름>.yaml`을 만듭니다. `backend`로 학습 방식을 고릅니다.
2. 기존 backend(kev, laya, chon, prompt_llm / 미사용: clef)로 안 되는 모델이면 `src/decision_model/backends/`에 `train()`과 `predict()`를 구현한 클래스를 추가하고 `backends/__init__.py`에 등록합니다.
3. 데이터, 질문 형식, 보정, 지표는 그대로 재사용됩니다.

## 모델별 메모

**Laya** (`backends/laya.py`)
- Laya 패키지의 학습 함수(`laya.train.train_model`)를 그대로 호출합니다. 체크포인트는 다국어용 `convaiinnovations/laya-multilingual`입니다.
- 기본값은 공식 레시피입니다: 인코더와 헤드 전체를 학습하고, 손실은 RLCD입니다.
- `configs/models/laya-multilingual.yaml`에서 `freeze_encoder: true`(헤드만 학습), `loss: soft-ce`(cross-entropy만)로 바꿀 수 있습니다. LoRA는 지원하지 않습니다.
- Laya 자체의 calibration 분할과 temperature는 쓰지 않고, 다른 모델과 같은 공통 보정을 씁니다.
- 질문마다 별도 시퀀스로 판단합니다(Kev와 같음).
- 입력이 `max_len`을 넘거나 선택지가 헤드 예산(`head_max_len`)에 안 들어가는 질문은 학습에서 빠지고, 평가에서는 균등 확률로 처리되며 `predictions_*.jsonl`의 `skipped`에 기록됩니다.

**Chon** (`backends/chon.py`, `chon-1.0-e4b`)
- 공개본(`ninhnv/chon-1.0-e4b`)은 Gemma 4 E4B 위의 LoRA 어댑터이고, 서빙 코드(`chon` 패키지)만 공개되어 학습 루프는 여기서 작성했습니다.
- 새 LoRA를 붙이지 않고 공개 어댑터를 그대로 불러와 이어서 학습합니다(베이스는 bf16 고정, 어댑터는 fp32).
- 질문은 `chon.serve`와 똑같이 chon의 `DecisionModel.encode`로 토큰화하고(같은 토큰 id인지 테스트로 확인), 답 위치의 선택지 글자 logit에 cross-entropy를 겁니다.
- 결과는 공개본과 같은 패키지 형식(어댑터 + `chon_config.json`)으로 저장해 `chon.serve --model <checkpoint>`로 평가합니다. 공개본에 맞춰 둔 temperature·post-calibration·thinking은 빼고(temperature 1), 보정은 다른 모델과 같은 공통 보정만 씁니다.
- lr·epoch·batch×accum은 Kev-4B와 같게 두었습니다. replay(J+R-*)는 지원하지 않습니다. `chon-1.0-e2b`는 여전히 B0만 돌립니다.

**Kev** (`backends/kev.py`)
- Kev 저장소의 `kev.train`과 `kev.serve`를 그대로 호출합니다. `KEV_DIR`(기본 `/opt/kev`)에 저장소가 있어야 합니다.
- 평가는 보정 전 확률(`KEV_TEMPERATURE=1.0`)과 fp32 경로로 받고, 보정은 공통 코드에서 합니다.
- fp32 경로는 Kev의 서빙 기본값(bf16, CUDA graph, fused 커널)을 끄므로 속도 비교에는 쓰지 않습니다. 속도는 `--serving`으로 잽니다.

**Clef-Flash** (`backends/clef.py`) — 미사용
- Cloudflare는 추론 코드(`joint_schema_model.py`)만 공개했습니다. 이 코드로 입력을 만들고, 학습 루프는 여기서 작성했습니다.
- 백본은 고정하고 언어 모델의 모든 선형 층에 LoRA를 붙입니다(비전 인코더 제외). joint head는 전부 학습합니다.
- 공식 학습 레시피가 없어 `configs/models/clef-flash.yaml`의 값은 시작점입니다. 파일럿에서 development로 조정합니다.
- Clef는 한 레코드의 질문을 한 시퀀스에서 함께 판단합니다. Kev는 질문끼리 서로 보지 못합니다. "동시" 수준을 해석할 때 이 차이를 감안합니다.
- replay를 쓰는 실험군(J+R-*)은 Kev 형식의 범용 데이터 파일이 필요합니다. `data/replay.jsonl`에 두거나 `--replay-file`로 지정합니다.

## 실제 GPU 실행 확인 현황

커밋 기록으로 확인한 것과, 기록이 없어 아직 확인이 필요한 것입니다.

| 항목 | 상태 | 근거 |
|---|---|---|
| Kev-4B 학습·평가 | 실행됨 | 학습 데이터가 384토큰 제한에 잘려 `max_state 2048`로 고침(e520e9a), J-best 학습 작업 후 보정 오류 수정(9c4a893) |
| Laya 학습·평가 | 실행됨 | L-*, J-best 실행 설정을 groups.yaml에 옮김(205b301), Drive 실험 결과 문서(2026.11.18) |
| Chon-1.0-E4B 학습 | 실행됨 | serving 작업이 학습한 실행을 차례로 재다 4번째에서 디스크가 참(88b7282) |
| Chon serving (vLLM 리더) | 실행됨 | 같은 serving 작업 (88b7282) |
| Gemma serving | 실패 후 수정, 재실행 기록 없음 | FlashInfer sampler 컴파일 중 엔진 중단 → `VLLM_USE_FLASHINFER_SAMPLER=0` (88b7282) |
| Kev serving (bf16, CUDA graph, fused 커널) | 기록 없음 | `flash-linear-attention==0.5.2`가 L40S에서 도는지 확인 필요 |
| Laya serving (bf16 autocast) | 기록 없음 | |
| Clef-Flash | 돌리지 않음 | CPU 가짜 모델로 흐름만 검증(`tests/test_clef_backend.py`) |

- `configs/tasks.yaml`의 질문 문구와 선택지 설명은 임시입니다. 운영에서 보낼 문구로 바꿔야 합니다.

## 테스트

```bash
cd train
pip install -e ".[dev,clef,laya]"
pytest
```
