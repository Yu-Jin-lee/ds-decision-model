#!/usr/bin/env python3
"""Compare load tests across models and methods, against the Gemma baseline.

    aws s3 sync s3://BUCKET/ds-jf-decision-model/runs/bench/ runs/bench/
    python bench/compare.py runs/bench --out runs/bench/_compare
    python bench/compare.py runs/bench/job --out runs/bench/_compare_job      # one method only

Reads every summary.json under the given folders (bench/job/run.py and bench/endpoint/run.py write the same
format) and writes to --out:

* comparison.md / comparison.csv: one row per result. Throughput (records/s) at its best step and within the
  p95 target, latency with one user, and the ratio to the baseline measured the same way (same method and
  task(s)).
* steps_all.csv: every step of every result (long format, for your own charts).
* throughput_vs_users.png, p95_vs_users.png, p95_vs_throughput.png (needs matplotlib).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

BASELINE = "gemma-4-26b-a4b"
COLUMNS = [
    "method", "model", "group", "seed", "label", "tasks", "mode", "precision", "questions_per_record",
    "max_records_per_s", "max_at_users", "max_p95_ms", "slo_records_per_s", "slo_at_users", "slo_p95_ms",
    "single_p50_ms", "single_p95_ms", "throughput_x_baseline", "slo_throughput_x_baseline",
    "single_p50_x_baseline", "baseline", "failed_requests", "records_reused",
]


def find_summaries(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for path in map(Path, paths):
        found += [path] if path.is_file() else sorted(p for p in path.rglob("summary.json") if "_compare" not in p.parts)
    return found


def name(summary: dict[str, Any]) -> str:
    label = f" [{summary['label']}]" if summary.get("label") else ""
    return f"{summary.get('method', 'job')}: {summary['model']} {summary['group']} s{summary['seed']}{label}"


def _ratio(value: float | None, base: float | None) -> float | None:
    return round(value / base, 2) if value and base else None


def row_of(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary["best"]
    top, slo, single = best.get("max_throughput") or {}, best.get("within_slo") or {}, best.get("single_user") or {}
    steps = summary["steps"]
    qpr = [s["questions_per_record"] for s in steps if s.get("questions_per_record")]
    return {
        "method": summary.get("method", "job"), "model": summary["model"], "group": summary["group"], "seed": summary["seed"], "label": summary.get("label") or "",
        "tasks": "+".join(summary["tasks"]), "mode": summary["endpoint"].get("mode"),
        "precision": summary["endpoint"].get("precision"),
        "questions_per_record": qpr[0] if qpr else None,
        "max_records_per_s": top.get("records_per_s"), "max_at_users": top.get("users"), "max_p95_ms": top.get("p95_ms"),
        "slo_records_per_s": slo.get("records_per_s"), "slo_at_users": slo.get("users"), "slo_p95_ms": best.get("slo_p95_ms"),
        "single_p50_ms": single.get("p50_ms"), "single_p95_ms": single.get("p95_ms"),
        "failed_requests": sum(s.get("failed") or 0 for s in steps),
        "records_reused": summary["conditions"].get("records_reused"),
    }


def add_baseline(rows: list[dict[str, Any]], baseline: str) -> None:
    """Ratios to the baseline result with the same method, label (measurement round) and tasks
    (e.g. S-s5-choice against P0-s5)."""
    bases = {(r["method"], r["label"], r["tasks"]): r for r in rows if r["model"] == baseline}
    for row in rows:
        base = bases.get((row["method"], row["label"], row["tasks"]))
        if base is None:
            continue
        row["baseline"] = f"{base['model']} {base['group']}"
        row["throughput_x_baseline"] = _ratio(row["max_records_per_s"], base["max_records_per_s"])
        row["slo_throughput_x_baseline"] = _ratio(row["slo_records_per_s"], base["slo_records_per_s"])
        # Below 1 is faster than the baseline.
        row["single_p50_x_baseline"] = _ratio(row["single_p50_ms"], base["single_p50_ms"])


def _fmt(value: Any) -> str:
    return "" if value is None else str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]], baseline: str) -> None:
    head = ["방식", "모델", "실험군", "태스크", "처리 방식", "최대 처리량 (records/s)", "그때 users", "그때 p95 ms",
            "목표 안 처리량", "1명 p50 ms", "1명 p95 ms", f"처리량 배수 ({baseline} 대비)", "1명 p50 배수", "실패"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in sorted(rows, key=lambda r: (r["method"], r["tasks"], -(r["max_records_per_s"] or 0))):
        cells = [r["method"], r["model"] + (f" [{r['label']}]" if r["label"] else ""), r["group"], r["tasks"], r["mode"],
                 r["max_records_per_s"], r["max_at_users"], r["max_p95_ms"], r["slo_records_per_s"],
                 r["single_p50_ms"], r["single_p95_ms"], r.get("throughput_x_baseline"),
                 r.get("single_p50_x_baseline"), r["failed_requests"]]
        lines.append("| " + " | ".join(_fmt(c) for c in cells) + " |")
    slo = next((r["slo_p95_ms"] for r in rows if r["slo_p95_ms"] is not None), None)
    lines += ["", f"- 처리량은 records/s(레코드 1건 = 키워드 판정 1건, 그 레코드의 질문 전부)입니다. "
                  f"목표 안 처리량은 p95 ≤ {slo} ms인 단계 중 가장 높은 값입니다.",
              f"- 배수는 방식과 태스크가 같은 {baseline} 결과와 비교합니다. 처리량 배수는 클수록, p50 배수는 작을수록 좋습니다.",
              "- 방식 job은 같은 장비 안 localhost, endpoint는 SageMaker 실시간 엔드포인트를 밖에서 호출한 값입니다."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot(out: Path, summaries: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: no charts")
        return []
    charts = [
        ("throughput_vs_users.png", "users", "records_per_s", "concurrent users", "records / s", True),
        ("p95_vs_users.png", "users", "p95_ms", "concurrent users", "p95 latency (ms)", True),
        ("p95_vs_throughput.png", "records_per_s", "p95_ms", "records / s", "p95 latency (ms)", False),
    ]
    written = []
    for file, x_key, y_key, x_label, y_label, log_x in charts:
        figure, axis = plt.subplots(figsize=(8, 5))
        for summary in summaries:
            points = [(s[x_key], s[y_key]) for s in summary["steps"] if s.get(x_key) and s.get(y_key)]
            if points:
                xs, ys = zip(*points)
                axis.plot(xs, ys, marker="o", label=name(summary),
                          linestyle="--" if summary["model"] == BASELINE else "-")
        if log_x:
            axis.set_xscale("log", base=2)
        if y_key == "p95_ms":
            axis.set_yscale("log")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(out / file, dpi=150)
        plt.close(figure)
        written.append(file)
    return written


def main() -> None:
    global BASELINE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="result folders or summary.json files")
    parser.add_argument("--out", default="runs/bench/_compare")
    parser.add_argument("--baseline", default=BASELINE, help="model the ratios are against")
    args = parser.parse_args()
    BASELINE = args.baseline

    summaries = [json.loads(p.read_text()) for p in find_summaries(args.paths)]
    summaries = [s for s in summaries if "steps" in s and "best" in s]   # only load-test results
    if not summaries:
        raise SystemExit("no load-test summary.json found")
    rows = [row_of(s) for s in summaries]
    add_baseline(rows, args.baseline)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "comparison.csv", rows, COLUMNS)
    write_markdown(out / "comparison.md", rows, args.baseline)
    step_rows = [{"result": name(s), "method": s.get("method", "job"), "model": s["model"], "group": s["group"], "seed": s["seed"],
                  "label": s.get("label") or "", "tasks": "+".join(s["tasks"]), **{k: v for k, v in step.items() if k != "by_task"}}
                 for s in summaries for step in s["steps"]]
    write_csv(out / "steps_all.csv", step_rows, list(step_rows[0]))
    charts = plot(out, summaries)
    print((out / "comparison.md").read_text(encoding="utf-8"))
    print(f"wrote {out}: comparison.md, comparison.csv, steps_all.csv{''.join(', ' + c for c in charts)}")


if __name__ == "__main__":
    main()
