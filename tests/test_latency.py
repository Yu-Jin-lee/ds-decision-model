import itertools

from decision_model.latency import measure, sample_records


def _records(task, lang, n, questions=1):
    return [{"id": f"{task}:{lang}:{i}", "task": task, "lang": lang, "questions": {f"q{j}": {} for j in range(questions)}}
            for i in range(n)]


def test_sample_is_fixed_per_cell_and_independent_of_input_order():
    records = _records("s2", "ko", 50) + _records("s4", "ja", 10)
    a = sample_records(records, per_cell=30)
    b = sample_records(list(reversed(records)), per_cell=30)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert sum(r["task"] == "s2" for r in a) == 30 and sum(r["task"] == "s4" for r in a) == 10


def test_measure_counts_only_timed_records_after_warmup():
    records = _records("s5", "en", 40, questions=2)
    calls = []
    ticks = itertools.count(0.0, 0.004)  # each clock read advances 4 ms
    result = measure(lambda r: calls.append(r["id"]), records, per_cell=20, warmup=5, clock=lambda: next(ticks))
    assert len(calls) == 25  # 5 warm-up + 20 timed
    assert result["records"] == 20
    assert result["p50_ms"] == 4.0 and result["records_per_s"] == 250.0
    assert result["questions_per_record"] == 2
    assert result["conditions"]["batch_size"] == 1
