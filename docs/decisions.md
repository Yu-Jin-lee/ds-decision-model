# 결정 로그

프로젝트에서 정한 것을 날짜순으로 한 줄씩 남깁니다. 최신이 아래입니다.
형식: `날짜 | 영역 | 결정 | 근거`. 영역은 데이터·기준·학습·평가·운영 중 하나입니다.

2026-11-25 이전 항목은 Drive 문서에서 옮긴 것입니다.

| 날짜 | 영역 | 결정 | 근거 |
|---|---|---|---|
| 2026-11-03 | 데이터 | 모든 태스크를 같은 5단계(기준 → 추출 → Gemma·Claude 라벨 → 사람 검수 → 마스터 고정)로 구축. 최종 라벨은 사람 > 모델 일치 | [구축 방안 - 공통](https://docs.google.com/document/d/1kdauKD-YYUhlCytE3v87ObDVecs9ihWjf5NlKrfmUzI/edit) |
| 2026-11-03 | 데이터 | 칸마다 대표 85%·어려운 15%. dev·test는 Gemma 오답을 넣어 일치율 약 85% (설계값, 성능으로 인용 금지) | 같은 문서 2.2 |
| 2026-11-04 | 데이터 | 산업군 단위 분할 확정. test 6개·development 5개 산업군을 모든 칸에 고정 | 같은 문서 부록 1, [criteria/sampling/common.yaml](../dataset/criteria/sampling/common.yaml) |
| 2026-11-05 | 기준 | S5는 SERP가 아니라 키워드(intent·site) 자체로 판단. intent가 site보다 우선 | [criteria/s5.md](../dataset/criteria/s5.md) |
| 2026-11-06 | 데이터 | S5 마스터 고정: 5,040건(언어별 1,680), 82%가 라벨 일치로 확정 | [구축 방안 - S5](https://docs.google.com/document/d/1-cuM7QwLERE8usUZYxH6Fomov4vjuVNL8LfFaFqH8Uo/edit) |
| 2026-11-06 | 데이터 | 다음 구축부터 브랜드 구분과 target·intent를 라벨링 전에 확정 (이번에 늦게 고쳐 147건 변경, 사람 라벨 35건 재저장) | [구축 절차 - S5](https://docs.google.com/document/d/1VTIpuMH3Ua_rbfGTyJ7v35SfcRYUkVDKTgu4Y-RwPtI/edit) |
| 2026-11-09 | 기준 | S2 Span 8개 라벨(0 유관, 1~7 무관 유형). 무관 유형 우선순위 3 > 4 > 5 > 6 > 2 > 1 > 7 | [criteria/s2.md](../dataset/criteria/s2.md) |
| 2026-11-09 | 데이터 | S2 표본은 S5 파티션을 그대로 쓰고, 무관 유형마다 칸의 10%, 유관은 나머지 | [구축 방안 - S2](https://docs.google.com/document/d/1EMPjiprPBBQ2rpbEJXpNTbm9c-aciOtWhWXm8Gr-rgQ/edit) |
| 2026-11-10 | 데이터 | S4 표본은 S5 마스터 키워드·파티션을 그대로 쓰고, 정답은 S5 검수에서 고친 최종 구분 | [구축 방안 - S4](https://docs.google.com/document/d/16m2NJAPJelQtV4s6gA0YsqphzEy0kckqGBmSuEVPbwE/edit) |
| 2026-11-12 | 기준 | S4는 target에 고유 이름이 하나라도 있으면 브랜드. 의도어는 보지 않음 | [criteria/s4.md](../dataset/criteria/s4.md) |
| 2026-11-16 | 학습 | 실험 설계: B0(무학습) → 1단계 질문 형식 선택(S-*) → 2단계 태스크·언어 축(J+R-*) | [실험 설계서](https://docs.google.com/document/d/1CQsWkZvSeysl-i-sbdY7u5qJDOJopjaNgnokIdAgZEc/edit), [train/configs/groups.yaml](../train/configs/groups.yaml) |
| 2026-11-25 | 운영 | 기준서 원본을 저장소 `dataset/criteria/`로 옮김. Google Docs는 공유용 사본 | 이 저장소 |
| 2026-11-25 | 운영 | 라벨은 Git(`dataset/labels/`), 입력·SERP는 S3에 두고 manifest에 경로·sha256 기록 | 이 저장소 |
| 2026-11-25 | 운영 | 학습데이터 구축(기준서·라벨·도구·회차 기록)은 `dataset/` 아래로 묶음 | 이 저장소 |
| 2026-11-25 | 운영 | 학습·평가 코드를 `train/` 아래로 옮김. SageMaker는 `train/`만 올리고, dirty 판정도 `train/`만 봄 | [PR #9](https://github.com/Yu-Jin-lee/ds-decision-model/pull/9) |
| 2026-11-25 | 학습 | 실험 모델은 kev-4b·laya-multilingual·chon-1.0-e4b·chon-1.0-e2b 4개. clef-flash·kev-9b·gemma-4-26b-a4b는 코드는 두고 문서에 미사용으로 표시 | [train/README.md](../train/README.md) |
