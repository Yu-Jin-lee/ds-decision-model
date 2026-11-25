# ds-decision-model

Journey Finder의 판단 단계(S2·S4·S5)를 운영 LLM 대신 작은 결정모델로 바꾸는 프로젝트입니다.

| 폴더 | 내용 |
|---|---|
| [dataset/](dataset/README.md) | 학습데이터 구축: 라벨링 기준서(`criteria/`), 라벨(`labels/`), 구축 도구 설계(`tools/`, 코드는 아직 없음), 작업 회차 기록(`reports/`) |
| [train/](train/README.md) | 학습·평가 파이프라인: 설정(`configs/`), 코드(`src/`), 실행·제출 스크립트, SageMaker 진입점, 테스트 |
| [experiments/](experiments/README.md) | 학습 실험 기록 |
| [docs/decisions.md](docs/decisions.md) | 결정 로그 |
| [jf_samples/](jf_samples/) | 운영 Journey Finder 결과 샘플 (산업군별) |

학습·평가 명령은 `train/` 안에서 실행합니다. 자세한 사용법은 [train/README.md](train/README.md)를 봅니다.
프로젝트 전체 맥락(태스크, 흐름, 지켜야 할 것)은 [CLAUDE.md](CLAUDE.md)에 있습니다.
