"""Endpoint load test: the endpoint every model is served behind, which records each step sends, and how
the answers become latency and throughput."""

import importlib.util
import os
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest

from decision_model.backends.base import Backend
from decision_model.bench import best_steps, check_partitions, plan_steps, record_pool, summarize_step, write_results
from decision_model.config import CONFIG_DIR, load_model_config
from decision_model.endpoint import Endpoint, MicroBatcher, make_server, open_endpoint

REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = _load("entry", "sagemaker/entry.py")
launch = _load("launch", "scripts/launch_sagemaker.py")
compare = _load("bench_compare", "bench/compare.py")
deploy = _load("deploy", "bench/endpoint/deploy.py")
serve = _load("serve", "bench/endpoint/serve.py")


def _record(i, task="s5"):
    return {"id": f"{task}:en:{i}", "task": task, "lang": "en", "level": "choice", "variant": None, "gold": 2,
            "state": {"keyword": f"kw{i}"}, "questions": {"s5_choice": {"type": "choice", "label": 2, "criteria": {}}}}


# -- records and steps ----------------------------------------------------------------------------------------------

def test_pool_drops_labels_and_does_not_depend_on_input_order():
    records = [_record(i) for i in range(20)]
    a, b = record_pool(records), record_pool(list(reversed(records)))
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert all("gold" not in r and "label" not in r["questions"]["s5_choice"] for r in a)
    assert "label" in records[0]["questions"]["s5_choice"]   # the source records are untouched


def test_each_step_sends_its_own_records_and_warmup_comes_from_the_end():
    pool = record_pool([_record(i) for i in range(100)])
    plan = plan_steps(pool, [1, 2, 4], requests_per_step=10, min_requests_per_user=4, warmup=5)
    assert [len(s["records"]) for s in plan["steps"]] == [10, 10, 16]
    sent = [r["id"] for s in plan["steps"] for r in s["records"]]
    assert len(sent) == len(set(sent)) and plan["records_reused"] == 0
    assert [r["id"] for r in plan["warmup"]] == [r["id"] for r in pool[-5:]]
    assert not set(sent) & {r["id"] for r in plan["warmup"]}


def test_a_small_pool_wraps_around_and_says_how_many_records_repeat():
    pool = record_pool([_record(i) for i in range(12)])
    plan = plan_steps(pool, [1, 1], requests_per_step=8, min_requests_per_user=1, warmup=2)
    assert plan["records_reused"] == 6   # 10 timed records, 16 sent


def test_the_test_partition_is_never_load_tested():
    with pytest.raises(ValueError, match="test partition"):
        check_partitions(["development", "test"])
    assert check_partitions(["train", "development"]) == ["train", "development"]


def _rows(n, users, every_s, ms):
    """n answers arriving every ``every_s`` seconds, each taking ``ms``."""
    return [{"id": str(i), "task": "s5", "lang": "en", "questions": 1, "ok": True, "ms": ms,
             "start_s": i * every_s, "end_s": i * every_s + ms / 1000, "server_ms": ms - 1, "batch": users}
            for i in range(n)]


def test_steady_throughput_leaves_out_ramp_and_drain():
    rows = _rows(100, users=4, every_s=0.01, ms=40.0)
    rows[0]["ms"] = rows[-1]["ms"] = 5000.0          # slow first/last answers are not counted
    stats = summarize_step(rows, users=4)
    assert stats["steady_records"] == 92
    assert stats["records_per_s"] == pytest.approx(100.0)
    assert stats["p95_ms"] == 40.0 and stats["max_ms"] == 40.0
    assert stats["questions_per_s"] == pytest.approx(100.0) and stats["mean_batch"] == 4


def test_failed_requests_are_counted_and_left_out_of_latency():
    rows = _rows(30, users=1, every_s=0.1, ms=50.0)
    rows[10] = {**rows[10], "ok": False, "ms": 1.0}
    stats = summarize_step(rows, users=1)
    assert stats["failed"] == 1 and stats["error_rate"] == pytest.approx(1 / 30, abs=1e-4)
    assert stats["p50_ms"] == 50.0


def test_best_steps_pick_max_throughput_and_the_best_within_the_target():
    steps = [{"users": 1, "records_per_s": 10, "questions_per_s": 10, "p50_ms": 90, "p95_ms": 100, "failed": 0},
             {"users": 8, "records_per_s": 60, "questions_per_s": 60, "p50_ms": 120, "p95_ms": 300, "failed": 0},
             {"users": 32, "records_per_s": 80, "questions_per_s": 80, "p50_ms": 400, "p95_ms": 900, "failed": 0},
             {"users": 64, "records_per_s": 95, "questions_per_s": 95, "p50_ms": 600, "p95_ms": 1500, "failed": 2}]
    best = best_steps(steps, slo_p95_ms=500)
    assert best["max_throughput"]["users"] == 32       # the 64-user step failed requests
    assert best["within_slo"]["users"] == 8
    assert best["single_user"]["p50_ms"] == 90


def test_results_are_written_with_summary_last(tmp_path):
    summary = {"steps": [summarize_step(_rows(20, 1, 0.1, 10.0), 1)]}
    write_results(tmp_path, summary, _rows(20, 1, 0.1, 10.0))
    assert {p.name for p in tmp_path.iterdir()} == {"summary.json", "steps.csv", "requests.jsonl.gz"}
    assert (tmp_path / "steps.csv").read_text().splitlines()[0].startswith("users,requests,ok")


# -- endpoint --------------------------------------------------------------------------------------------------------

def test_micro_batcher_runs_waiting_requests_together():
    sizes = []
    release = threading.Event()

    def predict_many(records):
        sizes.append(len(records))
        if len(sizes) == 1:
            release.wait(5)   # hold the first batch so the others queue up behind it
        return [{"id": r["id"], "answers": {}} for r in records]

    batcher = MicroBatcher(predict_many, max_batch=4)
    results = {}
    threads = [threading.Thread(target=lambda i=i: results.__setitem__(i, batcher.predict({"id": str(i)})))
               for i in range(9)]
    threads[0].start()
    time.sleep(0.05)
    for t in threads[1:]:
        t.start()
    time.sleep(0.05)
    release.set()
    for t in threads:
        t.join(5)
    batcher.close()
    assert sizes[0] == 1 and sizes[1:] == [4, 4]
    assert all(results[i][0]["id"] == str(i) for i in range(9))


def test_micro_batcher_gives_every_waiting_request_the_error():
    def broken(records):
        raise RuntimeError("cuda out of memory")

    batcher = MicroBatcher(broken, max_batch=4)
    with pytest.raises(RuntimeError, match="out of memory"):
        batcher.predict({"id": "x"})
    batcher.close()


class _Fake(Backend):
    def __init__(self, concurrent=False, batch=True, config=None):
        super().__init__(config or {"name": "fake", "backend": "fake", "checkpoint": "fake/ckpt"})
        self.concurrent_predictor = concurrent
        self.batch = batch

    def train(self, *args, **kwargs):
        raise AssertionError

    def predict(self, checkpoint, records):
        return [{"id": r["id"], "answers": {}} for r in records]

    @contextmanager
    def predictor(self, checkpoint):
        yield lambda record: {"id": record["id"], "answers": {}}

    def has_batch_predictor(self):
        return self.batch

    @contextmanager
    def batch_predictor(self, checkpoint):
        yield lambda records: self.predict(checkpoint, records)


@pytest.mark.parametrize("concurrent, batch, max_batch, mode", [
    (True, False, None, "concurrent"),   # kev.serve / chon.serve / vLLM batch it themselves
    (False, True, 8, "batch"),           # Laya in this process
    (False, True, 1, "serial"),          # batching turned off
    (False, False, 8, "serial"),         # no batch predictor
])
def test_the_endpoint_mode_follows_the_backend(concurrent, batch, max_batch, mode):
    with open_endpoint(_Fake(concurrent, batch), None, max_batch=max_batch) as endpoint:
        assert endpoint.mode == mode and endpoint.info["mode"] == mode
        assert endpoint.predict({"id": "a"})[0]["id"] == "a"


def test_server_backends_run_concurrently_and_laya_batches_by_its_config():
    for name in ("kev-4b", "chon-1.0-e4b", "gemma-4-26b-a4b"):
        config = load_model_config(name)
        from decision_model.backends import _make

        assert _make(config).concurrent_predictor, name
    laya = load_model_config("laya-multilingual")
    assert laya["serving"]["endpoint"]["max_batch"] > 1
    with open_endpoint(_Fake(config=laya), None) as endpoint:
        assert endpoint.mode == "batch" and endpoint.info["max_batch"] == laya["serving"]["endpoint"]["max_batch"]


def _post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def test_http_endpoint_answers_predict_info_and_errors():
    def predict_one(record):
        if record["id"] == "boom":
            raise ValueError("bad record")
        return {"id": record["id"], "answers": {"q": {"a": -0.1}}}

    server = make_server(Endpoint("concurrent", predict_one=predict_one, info={"model": "fake", "mode": "concurrent"}),
                         port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        answer = _post(f"{url}/predict", {"id": "r1", "questions": {}})
        assert answer["prediction"]["id"] == "r1" and answer["batch"] == 1 and answer["server_ms"] >= 0
        with urllib.request.urlopen(f"{url}/info", timeout=5) as response:
            assert json.loads(response.read())["model"] == "fake"
        with pytest.raises(urllib.error.HTTPError) as error:
            _post(f"{url}/predict", {"id": "boom"})
        assert error.value.code == 500 and "bad record" in error.value.read().decode()
    finally:
        server.shutdown()
        server.server_close()


# -- comparison ------------------------------------------------------------------------------------------------------

def _summary(model, group, tasks, steps, method="job"):
    return {"method": method, "model": model, "group": group, "seed": 0, "label": None, "tasks": tasks,
            "endpoint": {"mode": "concurrent", "precision": "bf16"}, "conditions": {"records_reused": 0},
            "steps": steps, "best": best_steps(steps, 1000)}


def _step(users, rps, p50, p95):
    return {"users": users, "records_per_s": rps, "questions_per_s": rps, "p50_ms": p50, "p95_ms": p95,
            "failed": 0, "questions_per_record": 1.0}


def test_comparison_puts_each_result_against_the_baseline_with_the_same_task():
    gemma = _summary("gemma-4-26b-a4b", "P0-s5", ["s5"], [_step(1, 2.0, 500, 600), _step(16, 20.0, 800, 950)])
    kev = _summary("kev-4b", "S-s5-choice", ["s5"], [_step(1, 25.0, 40, 50), _step(16, 200.0, 80, 100)])
    other = _summary("kev-4b", "S-s4-noul", ["s4"], [_step(1, 30.0, 30, 40)])
    rows = [compare.row_of(s) for s in (gemma, kev, other)]
    compare.add_baseline(rows, "gemma-4-26b-a4b")
    assert rows[1]["throughput_x_baseline"] == 10.0 and rows[1]["single_p50_x_baseline"] == 0.08
    assert rows[1]["baseline"] == "gemma-4-26b-a4b P0-s5"
    assert rows[2].get("throughput_x_baseline") is None      # no Gemma s4 result


def test_compare_script_writes_tables_and_long_steps(tmp_path):
    for summary in (_summary("gemma-4-26b-a4b", "P0-s5", ["s5"], [_step(1, 2.0, 500, 600)]),
                    _summary("kev-4b", "S-s5-choice", ["s5"], [_step(1, 25.0, 40, 50)])):
        folder = tmp_path / "bench" / "job" / summary["model"] / summary["group"] / "s0"
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps(summary))
    out = tmp_path / "compare"
    subprocess.run([sys.executable, str(REPO / "bench" / "compare.py"), str(tmp_path / "bench"),
                    "--out", str(out)], check=True, capture_output=True)
    assert {"comparison.md", "comparison.csv", "steps_all.csv"} <= {p.name for p in out.iterdir()}
    assert "12.5" in (out / "comparison.csv").read_text()     # 25 / 2 records/s


# -- SageMaker -------------------------------------------------------------------------------------------------------

def test_bench_jobs_go_under_bench_and_run_the_load_test(monkeypatch, tmp_path):
    assert launch.output_id("kev-4b/S-s5-choice/s0", False, bench=True) == "bench/job/kev-4b/S-s5-choice/s0"
    assert entry.take_flag(["--model", "m", "--bench", "1"], "--bench") == (["--model", "m"], True)
    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
    calls, uploads = [], []
    monkeypatch.setattr(entry, "sh", lambda *command, **_: calls.append(command))
    monkeypatch.setattr(entry, "upload_outputs", lambda run_id=None: uploads.append(run_id))
    monkeypatch.setattr(entry, "serving_checkpoint_args",
                        lambda model, group, seed: [] if group.startswith("B0") else ["--checkpoint", f"/ck/{group}"])
    entry.run_batch("kev-4b", [("B0-s5-choice", 0), ("S-s5-choice", 0)], [], None, serving=True, bench=True)
    assert all(c[1].endswith("bench/job/run.py") and "--serving" not in c for c in calls)
    assert calls[1][calls[1].index("--checkpoint") + 1] == "/ck/S-s5-choice"
    assert uploads == ["bench/job/kev-4b/B0-s5-choice/s0", "bench/job/kev-4b/S-s5-choice/s0"]


def test_bench_config_has_the_steps_and_no_test_partition():
    from decision_model.bench import load_bench_config

    config = load_bench_config(CONFIG_DIR)
    assert config["users"] == sorted(config["users"]) and config["users"][0] == 1
    check_partitions(config["partitions"])


def test_the_baseline_is_matched_by_method_too():
    rows = [compare.row_of(s) for s in (
        _summary("gemma-4-26b-a4b", "P0-s5", ["s5"], [_step(1, 2.0, 500, 600)], "job"),
        _summary("gemma-4-26b-a4b", "P0-s5", ["s5"], [_step(1, 1.0, 600, 700)], "endpoint"),
        _summary("kev-4b", "S-s5-choice", ["s5"], [_step(1, 20.0, 50, 60)], "endpoint"))]
    compare.add_baseline(rows, "gemma-4-26b-a4b")
    assert rows[2]["throughput_x_baseline"] == 20.0      # against the endpoint Gemma, not the job one


def test_results_of_the_two_methods_go_to_separate_folders(tmp_path):
    from decision_model.bench import result_dir

    assert result_dir(tmp_path, "job", "kev-4b/B0-s5-choice/s0") == tmp_path / "bench/job/kev-4b/B0-s5-choice/s0"
    assert result_dir(tmp_path, "endpoint", "m/g/s0", "b1") == tmp_path / "bench/endpoint/m/g/s0/b1"
    with pytest.raises(ValueError):
        result_dir(tmp_path, "other", "m/g/s0")


def test_job_launcher_is_the_run_launcher_in_bench_mode(capsys):
    launch.main(["--model", "kev-4b", "--group", "B0-s5-choice", "--seed", "0", "--data", "s3://b/d/",
                 "--output", "s3://b/r/", "--role", "r", "--image-uri", "i", "--dry-run"], bench=True)
    out = capsys.readouterr().out
    assert "kev-4b-bench-B0-s5-choice-s0" in out and "ml.g6e.xlarge" in out
    with pytest.raises(SystemExit):
        launch.main(["--model", "kev-4b", "--serving", "--data", "d", "--output", "o", "--role", "r",
                     "--image-uri", "i"], bench=True)          # --serving is not an option of the bench launcher


# -- method endpoint -------------------------------------------------------------------------------------------------

def test_sagemaker_routes_and_info_through_invocations():
    server = make_server(Endpoint("concurrent", predict_one=lambda r: {"id": r["id"], "answers": {}},
                                  info={"model": "fake", "mode": "concurrent"}), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{url}/ping", timeout=5) as response:
            assert response.status == 200
        assert _post(f"{url}/invocations", {"id": "r1", "questions": {}})["prediction"]["id"] == "r1"
        assert _post(f"{url}/invocations", {"__info__": True})["model"] == "fake"
    finally:
        server.shutdown()
        server.server_close()


def test_endpoint_names_fit_sagemaker():
    name = deploy.endpoint_name("chon-1.0-e4b", "J+R-all", 0, now=0)
    assert name.startswith("dm-bench-chon-1-0-e4b-j-r-all-s0-") and len(name) <= 63
    long = deploy.endpoint_name("laya-multilingual", "S-s5-choice-with-a-very-long-suffix-for-testing", 12, now=0)
    assert len(long) <= 63 and long.endswith("-0101000000") and "--" not in long


class _FakeSageMaker:
    def __init__(self, fail_on=None, statuses=("Creating", "InService")):
        self.calls, self.fail_on, self.statuses = [], fail_on, list(statuses)

    def __getattr__(self, name):
        def call(**kwargs):
            self.calls.append(name)
            if name == self.fail_on:
                raise RuntimeError(f"{name} failed")
            if name == "describe_endpoint":
                status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
                return {"EndpointStatus": status}
            return {}
        return call


def test_deploy_creates_model_config_endpoint_and_waits(monkeypatch):
    monkeypatch.setattr(deploy.time, "sleep", lambda s: None)
    sagemaker = _FakeSageMaker()
    deploy.deploy(sagemaker, "n", "img", "role", {"A": "1"}, "ml.g6e.xlarge", 3600, deploy.TAGS, log=lambda m: None)
    assert sagemaker.calls[:3] == ["create_model", "create_endpoint_config", "create_endpoint"]
    assert "delete_endpoint" not in sagemaker.calls


def test_a_failed_deploy_deletes_what_it_created(monkeypatch):
    monkeypatch.setattr(deploy.time, "sleep", lambda s: None)
    sagemaker = _FakeSageMaker(statuses=("Creating", "Failed"))
    with pytest.raises(RuntimeError, match="Failed"):
        deploy.deploy(sagemaker, "n", "img", "role", {}, "ml.g6e.xlarge", 3600, deploy.TAGS, log=lambda m: None)
    assert {"delete_endpoint", "delete_endpoint_config", "delete_model"} <= set(sagemaker.calls)


def test_container_gets_the_run_and_the_runs_prefix_for_its_checkpoint():
    env = deploy.container_environment("kev-4b", "S-s5-choice", 0, "s3://b/runs/")
    assert env == {"DECISION_MODEL_NAME": "kev-4b", "BENCH_GROUP": "S-s5-choice", "BENCH_SEED": "0",
                   "SM_OUTPUT_PREFIX": "s3://b/runs/"}
    command = serve.server_command("kev-4b", ["--checkpoint", "/ck"])
    assert command[1].endswith("bench/serve_endpoint.py")
    assert command[command.index("--host") + 1] == "0.0.0.0" and command[command.index("--port") + 1] == "8080"


def test_image_build_only_installs(monkeypatch):
    installs, weights = [], []
    monkeypatch.setattr(entry, "install", lambda config, serving, bench=False: installs.append((config["name"], serving)))
    monkeypatch.setattr(entry, "prepare_weights", lambda config: weights.append(config["name"]))
    monkeypatch.setattr(entry.sys, "argv", ["entry.py", "--model", "gemma-4-26b-a4b", "--serving", "1", "--install-only"])
    entry.main()
    assert installs == [("gemma-4-26b-a4b", True)] and weights == []   # weights are fetched when the container starts
    dockerfile = (REPO / "bench" / "endpoint" / "Dockerfile").read_text()
    assert "--install-only" in dockerfile and "bench/endpoint/serve.py" in dockerfile


# -- equal batching for every model ----------------------------------------------------------------------------------

def _fake_chon(monkeypatch):
    """Stand-ins for chon.infer and chon.serve with the parts chon_serve patches."""
    import threading as real_threading
    import types

    infer = types.ModuleType("chon.infer")

    class DecisionModel:
        def __init__(self):
            self._req_layout = None
            self._req_asked = None
            self._deadline = None
            self._shared = None

        def predict(self, layout, gate=None, seen=None):
            self._req_layout = layout          # set at the start of a request ...
            if gate:
                gate.wait(5)                   # ... another request finishes meanwhile ...
            seen.append(self._req_layout)      # ... and this one must still see its own layout
            self._req_layout = None

    infer.DecisionModel = DecisionModel
    serve = types.ModuleType("chon.serve")
    serve.threading = real_threading
    serve.create_app = lambda *a, **k: types.SimpleNamespace(on_event=lambda name: (lambda fn: fn))
    serve.main = lambda argv=None: None
    chon = types.ModuleType("chon")
    chon.infer, chon.serve = infer, serve
    for name, module in (("chon", chon), ("chon.infer", infer), ("chon.serve", serve)):
        monkeypatch.setitem(sys.modules, name, module)
    return infer, serve


def test_chon_serves_concurrent_requests_without_mixing_their_prompt_layout(monkeypatch):
    from decision_model.backends import chon_serve

    infer, serve = _fake_chon(monkeypatch)
    chon_serve.patch()
    model, seen, gate = infer.DecisionModel(), [], threading.Event()
    slow = threading.Thread(target=model.predict, args=("header",), kwargs={"gate": gate, "seen": seen})
    slow.start()
    model.predict("state_first", seen=seen)    # finishes while "header" is in flight and clears its own field
    gate.set()
    slow.join(5)
    assert seen == ["state_first", "header"]
    with serve.threading.Lock():               # the one-request lock is gone
        pass
    assert serve.threading.Lock is chon_serve._NoLock


def test_serving_chon_starts_the_concurrent_server_and_evaluation_keeps_chon_serve():
    from decision_model.backends import get_backend

    config = load_model_config("chon-1.0-e4b")
    assert "decision_model.backends.chon_serve" in get_backend(config, serving=True).serve_command(8000)
    assert "chon.serve" in get_backend(config).serve_command(8000)
    env = get_backend(config, serving=True).serve_env()
    assert str(REPO / "src") in env["PYTHONPATH"].split(os.pathsep)


def test_every_decision_model_can_batch_every_concurrent_request():
    from decision_model.bench import load_bench_config

    most = max(load_bench_config(CONFIG_DIR)["users"])
    assert load_model_config("laya-multilingual")["serving"]["endpoint"]["max_batch"] >= most


def test_answers_carry_each_questions_top_option():
    # In its own process: importing Locust patches the standard library for gevent.
    pytest.importorskip("locust")
    code = ("import json, sys; sys.path.insert(0, 'src'); from decision_model.load_test import _check_answer; row = {}; "
            "body = json.dumps({'prediction': {'id': 'r1', 'answers': {'q': {'a': -2.0, 'b': -0.1}}}}).encode(); "
            "assert _check_answer(row, {'id': 'r1'}, 200, body) and row['top'] == {'q': 'b'}, row")
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)


# -- measurement rounds ----------------------------------------------------------------------------------------------

def test_a_round_label_goes_one_folder_deeper_everywhere(monkeypatch, tmp_path, capsys):
    assert launch.output_id("kev-4b/B0-s5-choice/s0", False, True, "r2") == "bench/job/kev-4b/B0-s5-choice/s0/r2"
    launch.main(["--model", "kev-4b", "--group", "B0-s5-choice", "--seed", "0", "--data", "d", "--output", "o",
                 "--role", "r", "--image-uri", "i", "--label", "r2", "--dry-run"], bench=True)
    assert "kev-4b-bench" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        launch.main(["--model", "kev-4b", "--data", "d", "--output", "o", "--role", "r", "--image-uri", "i",
                     "--label", "bad/label", "--dry-run"], bench=True)
    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(entry, "sh", lambda *command, **_: calls.append(command))
    monkeypatch.setattr(entry, "upload_outputs", lambda run_id=None: None)
    monkeypatch.setattr(entry, "serving_checkpoint_args", lambda *a: [])
    entry.run_batch("kev-4b", [("B0-s5-choice", 0)], [], None, serving=True, bench=True, label="r2")
    assert calls[0][calls[0].index("--label") + 1] == "r2"


def test_ratios_compare_within_one_round():
    rows = [compare.row_of(s) for s in (
        {**_summary("gemma-4-26b-a4b", "P0-s5", ["s5"], [_step(1, 2.0, 500, 600)]), "label": None},
        {**_summary("gemma-4-26b-a4b", "P0-s5", ["s5"], [_step(1, 4.0, 500, 600)]), "label": "r2"},
        {**_summary("kev-4b", "B0-s5-choice", ["s5"], [_step(1, 8.0, 50, 60)]), "label": "r2"})]
    compare.add_baseline(rows, "gemma-4-26b-a4b")
    assert rows[2]["throughput_x_baseline"] == 2.0
