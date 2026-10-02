"""Tests for rentml.tracking (MLflow on a temporary SQLite store and the JSONL fallback)."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rentml import tracking
from rentml.tracking import (
    BASE_COLUMNS,
    Tracker,
    ablation_table,
    to_latex,
    to_markdown,
)


@pytest.fixture(scope="module")
def mlflow_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One SQLite store per module: creating the MLflow schema takes a few seconds."""
    return tmp_path_factory.mktemp("mlruns")


def _log_example(tracker: Tracker) -> None:
    with tracker.run(
        "lgbm", stage="tabular", params={"lgbm": {"num_leaves": 31}, "seed": 42}, tags={"n": 3}
    ):
        tracker.log_metrics({"MAE": 250.5, "MAPE (%)": 12.0, "R²": np.float64(0.8)})
        tracker.log_metrics({"MAE": 240.0}, step=1)
    with tracker.run("gpboost", stage="tabular+geo"):
        tracker.log_metrics({"MAE": 230.0})


def _log_epochs(tracker: Tracker) -> None:
    """Per-epoch values, a final value without step, and an explicit step going backwards."""
    with tracker.run("fusion", stage="+text"):
        for epoch in range(3):
            tracker.log_metrics({"val_MAE": 300.0 - epoch, "loss": 1.0 / (epoch + 1)}, step=epoch)
        tracker.log_metrics({"val_MAE": 150.0})  # restored best / final value
        tracker.log_metrics({"lr": 0.1}, step=5)
        tracker.log_metrics({"lr": 0.2}, step=3)  # lower step: MLflow keeps step 5


def _runs(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["start_time"] = pd.to_datetime(frame["start_time"], unit="s", utc=True)
    return frame


def test_mlflow_tracker_logs_run(mlflow_dir: Path) -> None:
    tracker = Tracker(mlflow_dir, experiment="exp-basic")
    assert tracker.backend == "mlflow"
    assert tracker.tracking_uri == f"sqlite:///{(mlflow_dir / 'mlflow.db').as_posix()}"
    _log_example(tracker)
    runs = tracker.runs_frame()
    assert list(runs.columns[: len(BASE_COLUMNS)]) == list(BASE_COLUMNS)
    assert runs["run_name"].tolist() == ["lgbm", "gpboost"]
    first = runs.iloc[0]
    assert first["stage"] == "tabular" and first["status"] == "FINISHED"
    assert first["metrics.MAE"] == 240.0 and first["metrics.MAPE_pct"] == 12.0
    assert first["metrics.R2"] == pytest.approx(0.8)
    assert first["params.lgbm.num_leaves"] == "31" and first["tags.n"] == "3"
    assert isinstance(first["start_time"], pd.Timestamp) and first["start_time"].tz is not None


def test_mlflow_artifacts_go_to_tracking_dir(mlflow_dir: Path, tmp_path: Path) -> None:
    tracker = Tracker(mlflow_dir, experiment="exp-artifacts")
    extra = tmp_path / "note.txt"
    extra.write_text("hello", encoding="utf-8")
    with tracker.run("tables", stage="report") as run_id:
        mlflow = tracking._import_mlflow()
        assert mlflow.active_run().info.run_id == run_id  # fluent API usable inside the block
        tracker.log_table(pd.DataFrame({"a": [1, 2]}), "segments")
        tracker.log_artifact(extra, artifact_path="notes")
    assert tracker.active_run_id is None
    found = {p.name for p in (mlflow_dir / "artifacts").rglob("*") if run_id in p.as_posix()}
    assert {"segments.csv", "note.txt"} <= found


def test_backends_produce_same_frame(mlflow_dir: Path, tmp_path: Path) -> None:
    mlflow_tracker = Tracker(mlflow_dir, experiment="exp-compare")
    jsonl_tracker = Tracker(tmp_path, experiment="exp-compare", enabled=False)
    for tracker in (mlflow_tracker, jsonl_tracker):
        _log_example(tracker)
        _log_epochs(tracker)
    a, b = mlflow_tracker.runs_frame(), jsonl_tracker.runs_frame()
    assert list(a.columns) == list(b.columns)
    stable = [c for c in a.columns if c not in ("run_id", "start_time", "end_time")]
    pd.testing.assert_frame_equal(a[stable], b[stable])


@pytest.mark.parametrize("enabled", [True, False])
def test_final_value_without_step_is_reported(
    mlflow_dir: Path, tmp_path: Path, enabled: bool
) -> None:
    tracker = Tracker(mlflow_dir if enabled else tmp_path, experiment="exp-steps", enabled=enabled)
    _log_epochs(tracker)
    run = tracker.runs_frame().iloc[-1]
    assert run["metrics.val_MAE"] == 150.0 and run["metrics.loss"] == pytest.approx(1 / 3)
    assert run["metrics.lr"] == 0.1


def test_jsonl_lines_without_steps_keep_last_value(tmp_path: Path) -> None:
    tracker = Tracker(tmp_path, experiment="old", enabled=False)
    events = [
        {"event": "start", "run_id": "r1", "run_name": "old", "stage": "s", "time_ms": 1},
        {"event": "metrics", "run_id": "r1", "metrics": {"MAE": 2.0}},
        {"event": "metrics", "run_id": "r1", "metrics": {"MAE": 1.0}},
        {"event": "metrics", "run_id": "unknown", "metrics": {"MAE": 9.0}},
    ]
    lines = [json.dumps({**event, "experiment": "old"}) for event in events]
    tracker.jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    runs = tracker.runs_frame()
    assert runs["metrics.MAE"].tolist() == [1.0] and runs["status"].tolist() == ["RUNNING"]


@pytest.mark.parametrize("enabled", [True, False])
def test_failed_run_is_recorded(mlflow_dir: Path, tmp_path: Path, enabled: bool) -> None:
    tracker = Tracker(mlflow_dir if enabled else tmp_path, experiment="exp-fail", enabled=enabled)
    with pytest.raises(ZeroDivisionError), tracker.run("broken", stage="tabular"):
        tracker.log_metrics({"MAE": 1.0})
        _ = 1 / 0
    runs = tracker.runs_frame()
    assert runs["status"].tolist() == ["FAILED"]
    assert tracker.active_run_id is None


def test_jsonl_fallback_when_disabled(tmp_path: Path) -> None:
    tracker = Tracker(tmp_path, enabled=False)
    assert tracker.backend == "jsonl" and tracker.tracking_uri is None
    with tracker.run("baseline", stage="naive") as run_id:
        tracker.log_metrics({"MAE": math.inf, "RMSE": 400})
        tracker.log_table(pd.DataFrame({"a": [1]}), "coverage.csv")
    assert not (tmp_path / "mlflow.db").exists()
    assert (tmp_path / "artifacts" / run_id / "tables" / "coverage.csv").is_file()
    events = [json.loads(line) for line in tracker.jsonl_path.read_text().splitlines()]
    assert [e["event"] for e in events] == ["start", "metrics", "artifact", "end"]
    runs = tracker.runs_frame()
    assert np.isnan(runs.loc[0, "metrics.MAE"]) and runs.loc[0, "metrics.RMSE"] == 400.0


def test_falls_back_when_mlflow_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> None:
        raise ImportError("no mlflow")

    monkeypatch.setattr(tracking, "_import_mlflow", missing)
    assert Tracker(tmp_path).backend == "jsonl"


def test_runs_frame_empty_and_malformed_lines(tmp_path: Path) -> None:
    tracker = Tracker(tmp_path, enabled=False)
    assert list(tracker.runs_frame().columns) == list(BASE_COLUMNS)
    with tracker.run("ok", stage="s"):
        tracker.log_params({"alpha": 0.1})
    with tracker.jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    other = Tracker(tmp_path, experiment="other", enabled=False)
    assert len(tracker.runs_frame()) == 1 and other.runs_frame().empty


def test_logging_requires_active_run_and_no_nesting(tmp_path: Path) -> None:
    tracker = Tracker(tmp_path, enabled=False)
    with pytest.raises(RuntimeError):
        tracker.log_metrics({"MAE": 1.0})
    with (
        tracker.run("outer", stage="s"),
        pytest.raises(RuntimeError),
        tracker.run("inner", stage="s"),
    ):
        pass
    with pytest.raises(ValueError):
        Tracker(tmp_path, experiment=" ", enabled=False)
    with pytest.raises(ValueError), tracker.run("", stage="s"):
        pass


def test_colliding_keys_are_rejected(tmp_path: Path) -> None:
    tracker = Tracker(tmp_path, enabled=False)
    with tracker.run("r", stage="s"):
        with pytest.raises(ValueError, match="MAPE_pct"):
            tracker.log_metrics({"MAPE (%)": 1.0, "MAPE_pct": 2.0})
        with pytest.raises(ValueError, match="a_b"):
            tracker.log_params({"a b": 1, "a_b": 2})
        with pytest.raises(ValueError, match="a.b"):
            tracker.log_params({"a.b": 1, "a": {"b": 2}})
        tracker.log_params({"a b": 1})
        with pytest.raises(ValueError, match="cannot change"):
            tracker.log_params({"a_b": 2})  # same key as "a b" -> immutable
    with (
        pytest.raises(ValueError, match="x_y"),
        tracker.run("t", stage="s", tags={"x y": 1, "x_y": 2}),
    ):
        pass
    runs = tracker.runs_frame()
    assert runs["params.a_b"].tolist() == ["1"] and "metrics.MAPE_pct" not in runs


def test_foreign_mlflow_run_is_reported(mlflow_dir: Path) -> None:
    tracker = Tracker(mlflow_dir, experiment="exp-stale")
    mlflow = tracking._import_mlflow()
    mlflow.set_tracking_uri(tracker.tracking_uri)
    mlflow.start_run(experiment_id=tracker._experiment_id, run_name="crashed-cell")
    try:
        with pytest.raises(RuntimeError, match="end_run"), tracker.run("next", stage="s"):
            pass
    finally:
        mlflow.end_run()
    assert tracker.active_run_id is None
    with tracker.run("next", stage="s"):
        tracker.log_metrics({"MAE": 1.0})
    assert tracker.runs_frame()["run_name"].tolist() == ["crashed-cell", "next"]


def test_git_tags_mark_dirty_trees(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    outputs = {"rev-parse HEAD": "abc123", "status --porcelain": " M src/rentml/data.py"}
    monkeypatch.setattr(tracking, "_git", lambda args: outputs[args])
    assert tracking._git_tags() == {"git_commit": "abc123", "git_dirty": "true"}
    outputs["status --porcelain"] = ""
    tracker = Tracker(tmp_path, enabled=False)
    with tracker.run("r", stage="s", tags={"git_commit": "manual"}):
        pass
    tags = tracker.runs_frame().iloc[0]
    assert tags["tags.git_commit"] == "manual" and tags["tags.git_dirty"] == "false"


def test_git_tags_absent_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(args: str) -> str:
        raise FileNotFoundError("git")

    monkeypatch.setattr(tracking, "_git", no_git)
    assert tracking._git_tags() == {}


def test_params_are_immutable_and_metrics_numeric(tmp_path: Path) -> None:
    tracker = Tracker(tmp_path, enabled=False)
    with tracker.run("r", stage="s"):
        tracker.log_params({"a": 1})
        tracker.log_params({"a": 1})  # identical value: no-op
        with pytest.raises(ValueError):
            tracker.log_params({"a": 2})
        with pytest.raises(TypeError):
            tracker.log_metrics({"MAE": "high"})  # type: ignore[dict-item]
        with pytest.raises(FileNotFoundError):
            tracker.log_artifact(tmp_path / "missing.png")
        with pytest.raises(ValueError):
            tracker.log_table(pd.DataFrame(), "a/b")
    assert tracker.runs_frame().loc[0, "params.a"] == "1"


def test_ablation_table_orders_stages_and_picks_runs() -> None:
    runs = _runs(
        [
            {"run_name": "a1", "stage": "tabular", "status": "FINISHED", "start_time": 1,
             "metrics.MAE": 300.0, "metrics.RMSE": 420.0},
            {"run_name": "a2", "stage": "tabular", "status": "FINISHED", "start_time": 2,
             "metrics.MAE": 310.0, "metrics.RMSE": 430.0},
            {"run_name": "b1", "stage": "+geo", "status": "FAILED", "start_time": 3,
             "metrics.MAE": 100.0, "metrics.RMSE": 150.0},
            {"run_name": "b2", "stage": "+geo", "status": "FINISHED", "start_time": 4,
             "metrics.MAE": 280.0, "metrics.RMSE": np.nan},
        ]
    )  # fmt: skip
    last = ablation_table(runs, stage_order=["tabular", "+geo", "+text"], metrics=["MAE", "RMSE"])
    assert last.index.tolist() == ["tabular", "+geo"]
    assert last["run_name"].tolist() == ["a2", "b2"]
    best = ablation_table(runs, stage_order=["tabular"], metrics=["metrics.MAE"], pick="best")
    assert best.loc["tabular", "run_name"] == "a1" and best.loc["tabular", "MAE"] == 300.0


def test_ablation_table_rejects_bad_input() -> None:
    runs = _runs([{"run_name": "a", "stage": "s", "status": "FINISHED", "start_time": 1}])
    with pytest.raises(KeyError, match="MAE"):
        ablation_table(runs, stage_order=["s"], metrics=["MAE"])
    with pytest.raises(ValueError):
        ablation_table(runs, stage_order=["s", "s"], metrics=["MAE"])
    with pytest.raises(ValueError):
        ablation_table(runs, stage_order=["s"], metrics=[])
    runs["metrics.MAE"] = 1.0
    with pytest.raises(ValueError):
        ablation_table(runs, stage_order=["s"], metrics=["MAE"], pick="median")


def test_to_markdown_formats_numbers_and_index() -> None:
    df = pd.DataFrame(
        {"MAE": [250.456, np.nan], "n": [10, 20], "R2": [0.81234, 0.7]},
        index=pd.Index(["tabular", "geo|text"], name="stage"),
    )
    md = to_markdown(df, floatfmt={"MAE": ".1f", "R2": ".3f"})
    lines = md.splitlines()
    assert lines[0] == "| stage | MAE | n | R2 |"
    assert lines[1] == "|:---|---:|---:|---:|"
    assert lines[2] == "| tabular | 250.5 | 10 | 0.812 |"
    assert lines[3] == "| geo\\|text | – | 20 | 0.700 |"
    assert md.endswith("\n")


def test_to_markdown_default_index_hidden_and_empty() -> None:
    assert to_markdown(pd.DataFrame({"a": ["x"]})).splitlines()[0] == "| a |"
    empty = pd.DataFrame({"a": pd.Series([], dtype=object), "b": pd.Series([], dtype=float)})
    assert to_markdown(empty).splitlines() == ["| a | b |", "|:---|---:|"]


def test_to_latex_booktabs_and_escaping() -> None:
    df = pd.DataFrame({"MAE_chf": [250.46], "share %": [np.nan]}, index=pd.Index(["a&b"]))
    tex = to_latex(df, caption="Ablation ($R^2$, 95% CI)", label="tab:ablation")
    assert "\\begin{tabular}{lrr}" in tex and "\\toprule" in tex and "\\bottomrule" in tex
    assert "\\caption{Ablation ($R^2$, 95\\% CI)}" in tex
    assert " & MAE\\_chf & share \\% \\\\" in tex
    assert "a\\&b & 250.5 & -- \\\\" in tex
    assert tex.rstrip().endswith("\\end{table}")


def test_to_latex_handles_backslash_and_na_rep() -> None:
    df = pd.DataFrame({"name": ["C:\\tmp~x^2"], "v": [None]})
    tex = to_latex(df, caption="c", label="l", na_rep="n/a")
    assert "C:\\textbackslash{}tmp\\textasciitilde{}x\\textasciicircum{}2 & n/a \\\\" in tex
