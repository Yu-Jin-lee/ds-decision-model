# experiments - 학습 실험 기록

실험 한 묶음(같은 질문을 위해 돌린 실행들)마다 파일 하나를 씁니다. 실행 결과 원본은 S3 `runs/`에 있고, 여기에는 무엇을 왜 돌렸고 무엇을 결론 냈는지 남깁니다.

```
experiments/YYYY-MM-DD-<주제>.md       예) 2026-11-19-kev-4b-s5-question-format.md
```

## 틀

```markdown
# <주제> (YYYY-MM-DD)

**결론:** 한두 문장. 이 결과로 무엇을 정했는지.

## 질문
이 실험이 답하려는 것. 실험 설계서의 어느 단계인지 (B0·P0 / 1단계 S-* / 2단계 L-*·J-best·J+R-*).

## 실행
| 모델 | 실험군 | seed | 데이터 버전 | 커밋 | S3 경로 |
(summary.json의 git commit·data sha256을 그대로 옮김)

## 결과
development 기준 칸별 정확도·Brier·ECE·자동 처리 비율. test는 쓰지 않음.

## 해석과 다음 단계
결정한 것은 docs/decisions.md에도 한 줄 남김 (예: train/configs/groups.yaml의 selected_levels).
```

## 이전 기록 (Drive)

저장소로 옮기기 전 결과는 Drive `03 평가` 폴더에 있습니다.

| 문서 | 링크 |
|---|---|
| 실험 결과 - Laya multilingual S5 질문 유형 비교 (2026.11.18) | [Google Docs](https://docs.google.com/document/d/1Hr5hhUd44GlESpQvkAEYiZroSQ2kI0j3X1CfAz8OtOo/edit) |
| 실험 결과 - Kev-4b multilingual S5 질문 유형 비교 (2026.11.19) | [Google Docs](https://docs.google.com/document/d/1j-V88KY4N61sMiEGu5cWvHhP03sP9lPGfPQD7qXWMbk/edit) |
| 실험 결과 정리 - 결정모델 모델별 결과 (2026.11.19) | [Google Docs](https://docs.google.com/document/d/1r1hwTqmKPpUL4wnoe4EavQB5FAH28P4sKZ3Lmsbj4Rg/edit) |
