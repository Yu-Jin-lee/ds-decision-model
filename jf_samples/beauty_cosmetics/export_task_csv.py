"""JF run → S2(Span)·S4·S5 태스크 CSV: 프롬프트 입력 컬럼 + 결과 컬럼.

사용 (jf 서버에서):
  python export_task_csv.py <출력 폴더> --jf-backend <ds-journey-finder-async/backend> \
      kr=<job_id> jp=<job_id> us=<job_id>

SERP는 run이 끝나면 단계 파일에서 지워지므로 hbase에서 다시 연다.
  1. S4-B5가 남긴 serp_key(AI overview가 있던 키워드)가 있으면 그 시점 SERP 그대로  → serp_source=serp_key
  2. 없으면 run 시작 시각 이전의 가장 최근 hbase 수집본                           → serp_source=hbase_before_run
     (S1은 hbase 최신본을 쓰고, 비었을 때만 수집 API로 폴백한다. 1번 키로 이 규칙을 검증한다)
  3. 그것도 없으면 빈 칸                                                            → serp_source=none
"""
import argparse
import csv
import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path("/data/prefect-results/jf")
HBASE = "https://mongttang-data-api.ascentlab.io/serp"
LOCALE = {"kr": ("ko", "kr"), "jp": ("ja", "jp"), "us": ("en", "us")}
TYPE_LABEL = {0: "related", 1: "other_category", 2: "same_name_other_category", 3: "hiring",
              4: "stock_finance", 5: "news_pr", 6: "person", 7: "other_unrelated"}


def key_time(serp_key: str) -> str:
    return serp_key.rsplit("timestamp=", 1)[1]  # YYYYmmddHHMMSSfff (UTC)


def list_keys(session, q, country):
    hl, gl = LOCALE[country]
    r = session.get(f"{HBASE}/list?q={quote(q)}&hl={hl}&gl={gl}&engine=GG&device=M&reserve=true&limit=30",
                    timeout=30)
    r.raise_for_status()
    return r.json() or []


def fetch_row(session, serp_key):
    r = session.get(f"{HBASE}/?row_key={serp_key}", timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def resolve_serp(q, country, stored_key, cutoff, session):
    """→ (serp_key, source, raw, picked_key_for_check)."""
    picked = None
    try:
        keys = list_keys(session, q, country)
        picked = next((k for k in keys if key_time(k) <= cutoff), None)
    except requests.RequestException:
        keys = []
    key, source = (stored_key, "serp_key") if stored_key else (picked, "hbase_before_run")
    if not key:
        return None, "none", None, picked
    raw = fetch_row(session, key)
    return (key, source, raw, picked) if raw else (None, "none", None, picked)


def join(values):
    return ", ".join(str(v).strip() for v in values or [] if str(v).strip())


def export(country, job_id, serp_mod):
    d = json.load(gzip.open(ROOT / job_id / "result.json.gz"))
    s1 = json.load(open(ROOT / job_id / "1.meta.json"))["stats"]
    aio = (json.load(open(ROOT / job_id / "brand_groups.json")).get("ai_overview")) or {}
    km, cdj = d["keyword_meta"], d["cdj_results"]
    started = datetime.fromisoformat(d["meta"]["started_at"])
    cutoff = f"{started:%Y%m%d%H%M%S}{started.microsecond // 1000:03d}"

    span_kws = [k for k, v in km.items() if v.get("tn") is not None]
    need = list(dict.fromkeys(span_kws + [r["keyword"] for r in cdj]))
    session = requests.Session()
    with ThreadPoolExecutor(16) as pool:
        got = dict(zip(need, pool.map(
            lambda q: resolve_serp(q, country, (aio.get(q) or {}).get("serp_key"), cutoff, session), need)))

    serp = {}
    check = {"stored_keys": 0, "rule_matches_stored": 0}
    for q, (key, source, raw, picked) in got.items():
        if source == "serp_key":
            check["stored_keys"] += 1
            check["rule_matches_stored"] += picked == key
        record = serp_mod.serp_record(raw) if raw else None
        overview = serp_mod.extract_ai_overview(raw) if raw else None
        top3 = serp_mod.format_serp_for_prompt(record, empty="") if record else ""
        requested = ((raw or {}).get("search_parameters") or {}).get("requested_time", "")
        serp[q] = {"serp_top3": top3,
                   "ai_overview": serp_mod.format_ai_overview(overview) if overview else "",
                   "serp_source": source, "serp_requested_at": requested}

    industry = s1.get("category_1depth") or ""
    category = "/".join(s1.get("category_2depth") or [])
    seeds = join(d["meta"]["queries"])
    base = {"country": country, "job_id": job_id}

    s2 = [{**base, "keyword": k, "industry": industry, "category": category,
           "market": f"{industry or '(없음)'} > {category or '(없음)'}", "seeds": seeds, **serp[k],
           "type_num": km[k]["tn"], "type_label": TYPE_LABEL[km[k]["tn"]], "relevant": km[k]["tn"] == 0}
          for k in span_kws]
    s4 = [{**base, "keyword": r["keyword"], "target": join(t.lower() for t in r["target"]), **serp[r["keyword"]],
           "llm_called": bool(r["target"]), "brand": r["brand"], "brand_part": r["brand_part"]}
          for r in cdj]
    s5 = [{**base, "keyword": r["keyword"], "industry": industry, "category": category, "brand": r["brand"],
           "target": join(r["target"]), "intent": join(r["intent"]), "site": join(r["site"]), **serp[r["keyword"]],
           "cdj_stage": r["result"], "stage_num": r["stage_num"]}
          for r in cdj]
    sources = {}
    for row in serp.values():
        sources[row["serp_source"]] = sources.get(row["serp_source"], 0) + 1
    return {"s2": s2, "s4": s4, "s5": s5}, {"serp_source": sources, **check}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--jf-backend", required=True)
    ap.add_argument("jobs", nargs="+", help="country=job_id")
    a = ap.parse_args()
    sys.path.insert(0, a.jf_backend)
    from app import serp as serp_mod

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {"s2": [], "s4": [], "s5": []}
    report = {}
    for arg in a.jobs:
        country, job_id = arg.split("=", 1)
        t, report[country] = export(country, job_id, serp_mod)
        for name in tables:
            tables[name] += t[name]
    for name, rows in tables.items():
        with open(out / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        report[f"{name}_rows"] = len(rows)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
