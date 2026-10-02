"""Experiment tracking (MLflow on SQLite, JSONL fallback) and report tables.

:class:`Tracker` logs runs with their ablation ``stage`` to ``sqlite:///<dir>/mlflow.db``
(artifacts in ``<dir>/artifacts``) or, without MLflow / with ``enabled=False``, to
``<dir>/runs.jsonl``; both backends return the same :meth:`Tracker.runs_frame`.
"""

import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pandas as pd
from sqlalchemy.exc import SADeprecationWarning, SQLAlchemyError

from rentml.plotting import ablation_table, to_latex, to_markdown

if TYPE_CHECKING:
    from mlflow import MlflowClient
    from mlflow.entities import Run

logger = logging.getLogger(__name__)

# ablation_table/to_markdown/to_latex live in rentml.plotting and are re-exported here.
__all__ = ["BASE_COLUMNS", "RunRecord", "Tracker", "ablation_table", "to_latex", "to_markdown"]

JSONL_NAME = "runs.jsonl"
BASE_COLUMNS: tuple[str, ...] = ("run_id", "run_name", "stage", "status", "start_time", "end_time")
_MAX_PARAM_LENGTH = 6000  # MLflow limit for parameter values
_KEY_REPLACEMENTS = {"%": " pct ", "²": "2", "³": "3", "Δ": "delta_", "±": "_pm_"}
_INVALID_KEY_CHARS = re.compile(r"[^A-Za-z0-9_\-./:]+")


@dataclass
class RunRecord:
    """One tracked run, independent of the backend (times in epoch milliseconds)."""

    run_id: str
    run_name: str
    stage: str
    status: str
    start_ms: int
    end_ms: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class _ActiveRun:
    run_id: str
    name: str
    params: dict[str, str] = field(default_factory=dict)
    max_step: dict[str, int] = field(default_factory=dict)  # highest step logged per metric
    last_ms: int = 0  # timestamp of the last metric batch (kept strictly increasing)


class Tracker:
    """Log runs, metrics, parameters and artifacts to MLflow or a JSONL file.

    Args:
        tracking_dir: Directory for ``mlflow.db``, ``artifacts/`` and ``runs.jsonl``.
        experiment: Experiment name.
        enabled: ``False`` forces the JSONL fallback (tests, CI).
        tracking_uri: Optional MLflow URI instead of the local SQLite file (e.g. PostgreSQL).

    Attributes:
        backend: ``"mlflow"`` or ``"jsonl"``.
        tracking_uri: MLflow tracking URI (``None`` for JSONL).
    """

    def __init__(
        self,
        tracking_dir: Path,
        experiment: str = "dspro2-rent",
        *,
        enabled: bool = True,
        tracking_uri: str | None = None,
    ) -> None:
        if not experiment.strip():
            raise ValueError("experiment name must not be empty")
        self.tracking_dir = Path(tracking_dir).resolve()
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self.backend = "jsonl"
        self.tracking_uri: str | None = None
        self._client: MlflowClient | None = None
        self._experiment_id: str | None = None
        self._active: _ActiveRun | None = None
        if enabled:
            self._init_mlflow(tracking_uri)
        logger.info("Tracker for %r uses the %s backend", experiment, self.backend)

    @property
    def jsonl_path(self) -> Path:
        """Path of the JSONL fallback file."""
        return self.tracking_dir / JSONL_NAME

    @property
    def active_run_id(self) -> str | None:
        """Id of the currently open run, if any."""
        return self._active.run_id if self._active else None

    @contextmanager
    def run(
        self,
        name: str,
        *,
        stage: str,
        params: Mapping[str, object] | None = None,
        tags: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        """Open a run; it ends as FINISHED, or FAILED/KILLED if the block raises.

        With MLflow it is also the active fluent run, so ``mlflow.<flavor>.log_model`` works
        inside the block.

        Args:
            name: Run name, e.g. ``"lgbm_monotone"``.
            stage: Ablation stage, e.g. ``"tabular+geo"`` (stored as tag ``stage``).
            params: Parameters logged at the start (nested dicts flattened with dots).
            tags: Extra tags; ``git_commit`` and ``git_dirty`` (``"true"`` if the working tree
                has uncommitted or untracked files) are added when available.

        Yields:
            The run id.

        Raises:
            RuntimeError: If a run of this tracker or a foreign MLflow run is still active.
            ValueError: If ``name`` or ``stage`` is empty or two tag keys collide.
        """
        if self._active is not None:
            raise RuntimeError(f"Run {self._active.name!r} is still active (no nesting)")
        if not name.strip() or not stage.strip():
            raise ValueError("Run name and stage must not be empty")
        tag_keys = _sanitize_keys(tags or {}, kind="Tag")
        user_tags = dict(zip(tag_keys, map(str, (tags or {}).values()), strict=True))
        run_id = self._start_run(name, stage, {**_git_tags(), **user_tags, "stage": stage})
        self._active = _ActiveRun(run_id=run_id, name=name)
        status = "FINISHED"
        try:
            self.log_params(params or {})
            yield run_id
        except BaseException as exc:  # only records the outcome; re-raised unchanged
            status = "KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
            raise
        finally:
            self._active = None
            self._end_run(run_id, status)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """Log numeric metrics to the active run.

        Both backends report per key the value at the highest step, newest on ties (MLflow).
        ``step=None`` logs at the key's highest step so far, so a final value logged after
        per-epoch values is the reported one. Keys are sanitised, infinite values become NaN.

        Args:
            metrics: Metric name to value.
            step: Optional step, e.g. the epoch.

        Raises:
            RuntimeError: If no run is active.
            TypeError: If a value is not numeric.
            ValueError: If two keys collide after sanitising (``"MAPE (%)"``, ``"MAPE_pct"``).
        """
        active = self._require_active()
        clean = _clean_metrics(metrics)
        if not clean:
            return
        steps = {k: active.max_step.get(k, 0) if step is None else int(step) for k in clean}
        active.max_step.update({k: max(s, active.max_step.get(k, s)) for k, s in steps.items()})
        # Strictly increasing: MLflow breaks (step, timestamp) ties by the larger value.
        active.last_ms = max(_now_ms(), active.last_ms + 1)
        if self._client is None:
            event = {"event": "metrics", "run_id": active.run_id, "metrics": clean}
            self._append({**event, "steps": steps})
            return
        from mlflow.entities import Metric  # lazy: MLflow is optional

        entries = [Metric(key, val, active.last_ms, steps[key]) for key, val in clean.items()]
        self._client.log_batch(active.run_id, metrics=entries)  # MLflow chunks large batches

    def log_params(self, params: Mapping[str, object]) -> None:
        """Log parameters as strings; nested mappings become dotted keys.

        Args:
            params: Parameter name to value. Re-logging an identical value is a no-op.

        Raises:
            RuntimeError: If no run is active.
            ValueError: If a parameter was already logged with a different value, or two keys
                collide after flattening/sanitising (``"a b"`` and ``"a_b"``).
        """
        active = self._require_active()
        pairs = _flatten(params)
        keys = _sanitize_keys([name for name, _ in pairs], kind="Parameter")
        clean = {key: str(value) for key, (_, value) in zip(keys, pairs, strict=True)}
        if any(len(value) > _MAX_PARAM_LENGTH for value in clean.values()):
            logger.warning("Parameter values truncated to %d characters", _MAX_PARAM_LENGTH)
            clean = {key: value[:_MAX_PARAM_LENGTH] for key, value in clean.items()}
        changed = [k for k, v in clean.items() if active.params.get(k, v) != v]
        if changed:
            raise ValueError(f"Parameters cannot change within a run: {changed}")
        new = {key: value for key, value in clean.items() if key not in active.params}
        if not new:
            return
        active.params.update(new)
        if self._client is None:
            self._append({"event": "params", "run_id": active.run_id, "params": new})
            return
        from mlflow.entities import Param  # lazy: MLflow is optional

        self._client.log_batch(active.run_id, params=[Param(k, v) for k, v in new.items()])

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        """Store a file (or a directory's contents) with the active run.

        Args:
            path: File or directory.
            artifact_path: Optional sub-directory inside the run's artifact folder.

        Raises:
            RuntimeError: If no run is active.
            FileNotFoundError: If ``path`` does not exist.
        """
        active = self._require_active()
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Artifact not found: {source}")
        if self._client is not None:
            log = self._client.log_artifacts if source.is_dir() else self._client.log_artifact
            log(active.run_id, str(source), artifact_path)
            return
        # Same layout as MLflow: a directory's contents go into artifact_path.
        target_dir = self.tracking_dir / "artifacts" / active.run_id / (artifact_path or "")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir if source.is_dir() else target_dir / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
        relative = target.relative_to(self.tracking_dir).as_posix()
        self._append({"event": "artifact", "run_id": active.run_id, "path": relative})

    def log_table(
        self, df: pd.DataFrame, name: str, *, index: bool = True, artifact_path: str = "tables"
    ) -> None:
        """Store a DataFrame as ``<artifact_path>/<name>.csv`` with the active run.

        Args:
            df: Table to store.
            name: File stem (``.csv`` suffix optional).
            index: Write the index as first column.
            artifact_path: Sub-directory inside the run's artifact folder.

        Raises:
            RuntimeError: If no run is active.
            ValueError: If ``name`` is empty or contains a path separator.
        """
        self._require_active()
        stem = name.strip().removesuffix(".csv")
        if not stem or "/" in stem or "\\" in stem:
            raise ValueError(f"Invalid table name: {name!r}")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / f"{stem}.csv"
            df.to_csv(target, index=index)
            self.log_artifact(target, artifact_path=artifact_path)

    def runs_frame(self) -> pd.DataFrame:
        """Return all runs of the experiment, oldest first.

        Returns:
            One row per run: :data:`BASE_COLUMNS` (times as UTC timestamps), then sorted
            ``metrics.*`` (float), ``params.*`` (str) and ``tags.*`` (str) columns.
        """
        records = self._mlflow_records() if self._client is not None else self._jsonl_records()
        return _records_to_frame(records)

    def _init_mlflow(self, tracking_uri: str | None) -> None:
        try:
            mlflow = _import_mlflow()
        except ImportError:
            logger.warning("MLflow is not installed; using the JSONL fallback")
            return
        from mlflow.exceptions import MlflowException  # lazy: MLflow is optional

        uri = tracking_uri or f"sqlite:///{(self.tracking_dir / 'mlflow.db').as_posix()}"
        try:
            client = mlflow.MlflowClient(tracking_uri=uri)
            experiment = client.get_experiment_by_name(self.experiment)
            if experiment is None:
                artifacts = (self.tracking_dir / "artifacts").as_uri()
                experiment_id = client.create_experiment(self.experiment, artifacts)
            else:
                experiment_id = experiment.experiment_id
                if experiment.lifecycle_stage == "deleted":
                    client.restore_experiment(experiment_id)
        except (MlflowException, SQLAlchemyError, OSError) as exc:
            logger.warning("MLflow unavailable (%s); using the JSONL fallback", exc)
            return
        self._client, self._experiment_id = client, experiment_id
        self.tracking_uri, self.backend = uri, "mlflow"

    def _start_run(self, name: str, stage: str, tags: dict[str, str]) -> str:
        if self._client is None:
            run_id = uuid.uuid4().hex
            event = {"event": "start", "run_id": run_id, "run_name": name, "stage": stage}
            self._append({**event, "tags": tags})
            return run_id
        mlflow = _import_mlflow()
        if (stale := mlflow.active_run()) is not None:  # e.g. a crashed cell, Optuna callback
            hint = "call mlflow.end_run() first"
            raise RuntimeError(f"MLflow run {stale.info.run_id} is active outside Tracker; {hint}")
        mlflow.set_tracking_uri(self.tracking_uri)
        active = mlflow.start_run(experiment_id=self._experiment_id, run_name=name, tags=tags)
        return str(active.info.run_id)

    def _end_run(self, run_id: str, status: str) -> None:
        if self._client is None:
            self._append({"event": "end", "run_id": run_id, "status": status})
            return
        mlflow = _import_mlflow()
        if (current := mlflow.active_run()) is not None and current.info.run_id == run_id:
            mlflow.end_run(status=status)
        else:
            self._client.set_terminated(run_id, status)

    def _require_active(self) -> _ActiveRun:
        if self._active is None:
            raise RuntimeError("No active run; use `with tracker.run(...):` first")
        return self._active

    def _append(self, event: dict[str, object]) -> None:
        payload = {**event, "experiment": self.experiment, "time_ms": _now_ms()}
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _jsonl_records(self) -> list[RunRecord]:
        if not self.jsonl_path.is_file():
            return []
        records: dict[str, RunRecord] = {}
        steps: dict[tuple[str, str], int] = {}  # (run_id, metric) -> step of the reported value
        lines = self.jsonl_path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d in %s", number, self.jsonl_path)
                continue
            if event.get("experiment") == self.experiment:
                _apply_event(records, event, steps)
        return list(records.values())

    def _mlflow_records(self) -> list[RunRecord]:
        runs = self._client.search_runs(
            [self._experiment_id], max_results=50_000, order_by=["attributes.start_time ASC"]
        )  # 50'000 is MLflow's per-call maximum; far above the expected number of runs
        return [_record_from_mlflow(run) for run in runs]


def _import_mlflow() -> ModuleType:
    # Lazy import: MLflow is heavy (~1 s) and optional. MLFLOW_DISABLE_AGENT_HINT mutes its import
    # banner; its INFO/alembic logs and SQLAlchemy warnings from MLflow's own ORM are silenced.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    warnings.filterwarnings("ignore", category=SADeprecationWarning, module=r"mlflow(\.|$)")
    import mlflow

    for name in ("mlflow", "alembic"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return mlflow


def _record_from_mlflow(run: "Run") -> RunRecord:
    info, tags = run.info, dict(run.data.tags)
    return RunRecord(
        run_id=info.run_id,
        run_name=info.run_name or tags.get("mlflow.runName", ""),
        stage=tags.get("stage", ""),
        status=info.status,
        start_ms=int(info.start_time),
        end_ms=int(info.end_time) if info.end_time else None,
        metrics=dict(run.data.metrics),
        params=dict(run.data.params),
        tags={k: v for k, v in tags.items() if k != "stage" and not k.startswith("mlflow.")},
    )


def _apply_event(
    records: dict[str, RunRecord], event: dict[str, Any], steps: dict[tuple[str, str], int]
) -> None:
    # `event` comes from json.loads, whose values are untyped -- hence Any. Metrics follow
    # MLflow: the highest step wins, ties go to the later line (old lines without steps: 0).
    run_id, kind = str(event.get("run_id", "")), event.get("event")
    if kind == "start":
        tags = {k: v for k, v in (event.get("tags") or {}).items() if k != "stage"}
        name, stage = str(event.get("run_name", "")), str(event.get("stage", ""))
        start_ms = int(event.get("time_ms", 0))
        records[run_id] = RunRecord(run_id, name, stage, "RUNNING", start_ms, tags=tags)
        return
    record = records.get(run_id)
    if record is None:
        return
    if kind == "metrics":
        logged_steps = event.get("steps") or {}
        for key, value in (event.get("metrics") or {}).items():
            step = int(logged_steps.get(key, 0))
            if step >= steps.get((run_id, key), step):
                record.metrics[key], steps[(run_id, key)] = value, step
    elif kind == "params":
        record.params.update(event.get("params") or {})
    elif kind == "end":
        record.status = str(event.get("status", "FINISHED"))
        record.end_ms = int(event.get("time_ms", 0))


def _records_to_frame(records: Sequence[RunRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=list(BASE_COLUMNS))
    # Nested dicts become "metrics.<key>" etc.; dots inside keys are kept as they are.
    frame = pd.json_normalize([asdict(record) for record in records])
    frame = frame.rename(columns={"start_ms": "start_time", "end_ms": "end_time"})
    for col in ("start_time", "end_time"):
        frame[col] = pd.to_datetime(frame[col], unit="ms", utc=True)
    prefixes = ("metrics.", "params.", "tags.")
    dynamic = [col for prefix in prefixes for col in sorted(frame) if col.startswith(prefix)]
    metric_cols = [col for col in dynamic if col.startswith("metrics.")]
    frame[metric_cols] = frame[metric_cols].astype(float)
    frame = frame.sort_values("start_time", kind="stable").reset_index(drop=True)
    return frame[[*BASE_COLUMNS, *dynamic]]


def _clean_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    keys = _sanitize_keys(metrics, kind="Metric")
    clean: dict[str, float] = {}
    for key, (name, value) in zip(keys, metrics.items(), strict=True):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Metric {name!r} must be numeric, got {value!r}") from exc
        clean[key] = math.nan if math.isinf(number) else number
    return clean


def _flatten(params: Mapping[str, object], prefix: str = "") -> list[tuple[str, object]]:
    pairs: list[tuple[str, object]] = []
    for key, value in params.items():
        if isinstance(value, Mapping):
            pairs.extend(_flatten(value, prefix=f"{prefix}{key}."))
        else:
            pairs.append((f"{prefix}{key}", value))
    return pairs


def _sanitize_keys(keys: Iterable[object], *, kind: str) -> list[str]:
    """Map keys to MLflow's character set; reject keys that collide (one would be lost)."""
    seen: dict[str, object] = {}
    for key in keys:
        text = str(key)
        for old, new in _KEY_REPLACEMENTS.items():
            text = text.replace(old, new)
        clean = re.sub(r"_+", "_", _INVALID_KEY_CHARS.sub("_", text.strip())).strip("_")
        if not clean:
            raise ValueError(f"{kind} key {key!r} has no valid characters")
        if clean in seen:
            raise ValueError(f"{kind} keys {seen[clean]!r} and {key!r} both become {clean!r}")
        seen[clean] = key
    return list(seen)


def _git_tags() -> dict[str, str]:
    """``git_commit`` plus ``git_dirty``: HEAD alone does not identify uncommitted code."""
    try:
        commit, status = (_git(args) for args in ("rev-parse HEAD", "status --porcelain"))
    except (OSError, subprocess.SubprocessError):
        return {}
    return {"git_commit": commit, "git_dirty": str(bool(status)).lower()} if commit else {}


def _git(args: str) -> str:
    cmd = ["git", *args.split()]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True).stdout.strip()


def _now_ms() -> int:
    return time.time_ns() // 1_000_000
