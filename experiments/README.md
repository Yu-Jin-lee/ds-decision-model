# experiments - 학습 실험 기록

실험 한 묶음(같은 질문을 위해 돌린 실행들)마다 파일 하나를 씁니다. 실행 결과 원본은 S3 `runs/`에 있고, 여기에는 무엇을 왜 돌렸고 무엇을 결론 냈는지 남깁니다.

```
experiments/YYYY-MM-DD-<주제>.md       예) 2026-11-19-kev-4b-s5-question-format.md
```

## 결과 대시보드

지금까지의 실험 결과를 한 페이지에 모았습니다: [결정모델 파인튜닝 실험 결과](https://claude.ai/artifact/4UTsxocRnZ9zwaXz3HHNhT) (Claude 아티팩트)

| 묶음 | 내용 |
|---|---|
| 효과 | B0 대비 칸별·태스크별 정확도, 운영 Gemma 정확도 |
| 질문 유형 | 형식별(choice·score·동시) B0와 단독 학습 결과 |
| 태스크 | J-best(세 태스크 한 모델) vs 단독 학습 |
| 언어 | 혼합 학습 vs 언어 전용, 학습×평가 언어 변화 |
| S2 2지 | 유관(0) vs 무관(1~7)으로 다시 잰 정확도 |
| 속도 | 모델·태스크별 지연·처리량, S5 정확도 vs 지연, 동시 요청 부하 테스트 |
| A/B | chon-1.0-e4b J-best vs 운영 Gemma 일치율 (JF 뷰티 kr·jp·us) |

정확도는 development 기준, seed 0 한 번입니다(test는 쓰지 않음). A/B는 운영 JF 키워드에서 Gemma와의 일치율이라 정확도가 아닙니다. 수치의 출처(결과 정리 문서 날짜, S3 경로, 커밋)는 대시보드의 Sources에 있습니다.

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

저장소로 옮기기 전 결과는 Drive [03 평가](https://drive.google.com/drive/folders/12go7zGKMhisnkx1rVzpPmp1G-vVe9pVp) 폴더에 있습니다.

| 문서 | 링크 |
|---|---|
| 실험 결과 - Laya multilingual S5 질문 유형 비교 (2026.11.18) | [Google Docs](https://docs.google.com/document/d/1Hr5hhUd44GlESpQvkAEYiZroSQ2kI0j3X1CfAz8OtOo/edit) |
| 실험 결과 - Kev-4b multilingual S5 질문 유형 비교 (2026.11.19) | [Google Docs](https://docs.google.com/document/d/1j-V88KY4N61sMiEGu5cWvHhP03sP9lPGfPQD7qXWMbk/edit) |
| 실험 결과 정리 - 결정모델 모델별 결과 (2026.11.19) | [Google Docs](https://docs.google.com/document/d/1r1hwTqmKPpUL4wnoe4EavQB5FAH28P4sKZ3Lmsbj4Rg/edit) |
