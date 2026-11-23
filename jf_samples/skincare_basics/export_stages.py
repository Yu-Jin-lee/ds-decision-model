"""JF 배치 result.json.gz → 단계별(S1~S6) 키워드 단위 JSONL.gz.

사용: python export_stages.py <출력 폴더>  (jf 서버에서, /data/prefect-results 읽기 권한 필요)
"""
import gzip, json, sys
from pathlib import Path

JOBS = {
    "kr": "254afd99-68f5-4983-9725-8b7495732fa6",
    "jp": "ef8f8a56-d45e-4776-92d1-b7dfcc3f9f6f",
    "us": "a397ec90-6692-423a-a02f-5cc703a9ecbb",
}
ROOT = Path("/data/prefect-results/jf")
out_root = Path(sys.argv[1])


def topic_index(data):
    idx = {}
    def walk(stage, node, path):
        here = [*path, node.get("label", "")]
        for e in node.get("keywords") or []:
            if e.get("keyword"):
                idx[(stage, e["keyword"])] = (here, node.get("id"))
        for ch in node.get("children") or []:
            walk(stage, ch, here)
    for stage, p in (data or {}).items():
        for r in ((p or {}).get("tree") or {}).get("roots") or []:
            walk(stage, r, [])
    return idx


def write(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


summary = {}
for country, job_id in JOBS.items():
    d = json.load(gzip.open(ROOT / job_id / "result.json.gz"))
    step_meta = {n: json.load(open(ROOT / job_id / f"{n}.meta.json")) for n in range(1, 7)}
    out = out_root / country
    out.mkdir(parents=True, exist_ok=True)
    km = d["keyword_meta"]
    cdj = d["cdj_results"]
    method = {(st, r["keyword"]): r.get("method") for st, p in d["data"].items() for r in p.get("keyword_results") or []}
    topics = topic_index(d["data"])

    counts = {}
    counts["s1_keyword_expansion"] = write(out / "s1_keyword_expansion.jsonl.gz", [
        {"keyword": k, "volume": v.get("v"), "seeds": v.get("sd", []), "sources": v.get("src", []),
         "graph_hops": v.get("hp")} for k, v in km.items()])
    counts["s2_relevance"] = write(out / "s2_relevance.jsonl.gz", [
        {"keyword": k, "relevant": v.get("rel"), **({"type_num": v["tn"]} if "tn" in v else {})}
        for k, v in km.items()])
    counts["s3_attributes"] = write(out / "s3_attributes.jsonl.gz", [
        {"keyword": r["keyword"], "target": r["target"], "intent": r["intent"], "site": r["site"]} for r in cdj])
    counts["s4_brand"] = write(out / "s4_brand.jsonl.gz", [
        {"keyword": r["keyword"], "brand": r["brand"], "brand_part": r["brand_part"]} for r in cdj])
    counts["s5_cdj"] = write(out / "s5_cdj.jsonl.gz", [
        {"keyword": r["keyword"], "brand": r["brand"], "cdj_stage": r["result"], "stage_num": r["stage_num"],
         "error": r["error"]} for r in cdj])
    s6 = []
    for r in cdj:
        st = r["result"]
        path, cid = topics.get((st, r["keyword"]), (None, None))
        s6.append({"keyword": r["keyword"], "cdj_stage": st, "topic_path": path, "cluster_id": cid,
                   "method": method.get((st, r["keyword"]))})
    counts["s6_topics"] = write(out / "s6_topics.jsonl.gz", s6)

    meta = {**d["meta"], "industry": step_meta[1]["stats"].get("category_1depth"),
            "category_2depth": step_meta[1]["stats"].get("category_2depth"),
            "steps": {n: {"name": m["name"], "finished_at": m["finished_at"], "stats": m["stats"]}
                      for n, m in step_meta.items()},
            "files": counts}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    summary[country] = counts
print(json.dumps(summary, indent=1))
