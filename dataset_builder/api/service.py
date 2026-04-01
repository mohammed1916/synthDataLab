"""API service layer for synthDataLab pipeline operations."""
from __future__ import annotations

import datetime
import json
import logging
import shutil
import threading
import traceback
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset_builder.config import Config
from dataset_builder.ingestion.ingestor import Ingestor, IngestionResult
from dataset_builder.main import (
    _save_jsonl,
    _setup_file_logging,
    step_analyze,
    step_evaluate,
    step_filter,
    step_generate,
    step_ingest,
    step_validate,
)
from dataset_builder.schema.dataset_schema import DatasetSample
from dataset_builder.validation.annotation import AnnotatedSample

try:
    from generation.orchestrator import MultiAgentOrchestrator, OrchestratorConfig, SteeringMode
except ImportError:
    MultiAgentOrchestrator = None  # type: ignore
    OrchestratorConfig = None  # type: ignore
    SteeringMode = None  # type: ignore

logger = logging.getLogger(__name__)


class RunStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class RunStatus:
    run_id: str
    created_at: str
    updated_at: str
    status: RunStatusEnum
    input_path: Optional[str] = None
    input_text: Optional[str] = None
    mock: bool = False
    workers: int = 1
    agent: bool = False
    steering: str = "auto"
    threshold: float = 0.7
    force: bool = False
    reset_fingerprints: bool = False
    cancel_requested: bool = False
    error: Optional[str] = None
    run_dir: Optional[str] = None
    outputs: Dict[str, str] = None
    pipeline_stage: Optional[str] = "pending"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class PipelineController:
    """Runs and tracks multiple pipeline executions."""

    def __init__(self):
        self._lock = threading.Lock()
        self._runs: Dict[str, RunStatus] = {}
        self._load_existing_runs()

    def _run_base_path(self) -> Path:
        return Path(Config().storage.data_dir)

    def _run_path(self, run_id: str) -> Path:
        return self._run_base_path() / "runs" / run_id

    def _status_path(self, run_id: str) -> Path:
        return self._run_path(run_id) / "status.json"

    def _save_status(self, run_status: RunStatus) -> None:
        path = self._status_path(run_status.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run_status.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_existing_runs(self) -> None:
        root = self._run_base_path() / "runs"
        if not root.exists():
            return
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            status_file = run_dir / "status.json"
            if not status_file.exists():
                continue
            try:
                raw = json.loads(status_file.read_text(encoding="utf-8"))
                run_status = RunStatus(
                    run_id=raw.get("run_id", run_dir.name),
                    created_at=raw.get("created_at", ""),
                    updated_at=raw.get("updated_at", ""),
                    status=RunStatusEnum(raw.get("status", "failed")),
                    input_path=raw.get("input_path"),
                    input_text=raw.get("input_text"),
                    mock=raw.get("mock", False),
                    workers=raw.get("workers", 1),
                    agent=raw.get("agent", False),
                    steering=raw.get("steering", "auto"),
                    threshold=float(raw.get("threshold", 0.7)),
                    force=raw.get("force", False),
                    reset_fingerprints=raw.get("reset_fingerprints", False),
                    cancel_requested=raw.get("cancel_requested", False),
                    error=raw.get("error"),
                    run_dir=str(run_dir),
                    outputs=raw.get("outputs", {}),
                    pipeline_stage=raw.get("pipeline_stage", "pending"),
                )
                self._runs[run_status.run_id] = run_status
            except Exception:
                logger.warning("Ignoring corrupt run status in %s", status_file)

    def list_runs(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in sorted(self._runs.values(), key=lambda run: run.created_at, reverse=True)]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        run_status = self._runs.get(run_id)
        if not run_status:
            return None
        return run_status.to_dict()

    def get_run_logs(self, run_id: str) -> Optional[str]:
        log_path = self._run_base_path() / "logs" / f"pipeline_{run_id}.log"
        if not log_path.exists():
            return None
        return log_path.read_text(encoding="utf-8", errors="replace")

    def create_run(
        self,
        input_path: Optional[str] = None,
        input_text: Optional[str] = None,
        mock: bool = False,
        workers: int = 1,
        agent: bool = False,
        steering: str = "auto",
        threshold: float = 0.7,
        force: bool = False,
        reset_fingerprints: bool = False,
    ) -> RunStatus:

        cfg = Config()
        cfg.llm.provider = "mock" if mock else cfg.llm.provider
        cfg.generation.max_workers = workers

        run_status = RunStatus(
            run_id=cfg.run_id,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            status=RunStatusEnum.PENDING,
            input_path=input_path,
            input_text=input_text,
            mock=mock,
            workers=workers,
            agent=agent,
            steering=steering,
            threshold=threshold,
            force=force,
            reset_fingerprints=reset_fingerprints,
            run_dir=str(cfg.run_dir()),
            outputs={},
            pipeline_stage="pending",
        )

        self._runs[cfg.run_id] = run_status
        self._save_status(run_status)

        thread = threading.Thread(
            target=self._execute_run,
            args=(cfg, run_status,),
            daemon=True,
        )
        thread.start()

        return run_status

    def _execute_run(self, cfg: Config, run_status: RunStatus) -> None:
        run_id = run_status.run_id
        try:
            self._update_status(run_id, RunStatusEnum.RUNNING, pipeline_stage="ingest")

            _setup_file_logging(cfg.storage.data_dir / "logs", run_id)
            cfg.ensure_dirs()
            cfg.validate()

            # Persist manifest
            run_dir = cfg.run_dir()
            (run_dir / "manifest.json").write_text(
                json.dumps(cfg.run_manifest(), indent=2, ensure_ascii=False), encoding="utf-8"
            )

            # 1. Ingest
            if run_status.input_text:
                ingestor = Ingestor()
                ingestion_results = ingestor.ingest_text(run_status.input_text, source_name="api_text")
            else:
                ingestion_results = step_ingest(cfg, run_status.input_path)

            if self._is_cancel_requested(run_id):
                self._update_status(run_id, RunStatusEnum.CANCELED, error="Run canceled during ingest", pipeline_stage="canceled")
                return

            # 2. Generate
            self._update_status(run_id, RunStatusEnum.RUNNING, pipeline_stage="generate")

            # 2. Generate
            if run_status.agent and MultiAgentOrchestrator is not None:
                orch_cfg = OrchestratorConfig(
                    steering_mode=SteeringMode(run_status.steering),
                    critic_pass_threshold=run_status.threshold,
                    critic_review_threshold=max(0.0, run_status.threshold - 0.25),
                    show_dashboard=False,
                )
                orch = MultiAgentOrchestrator(config=cfg, orch_config=orch_cfg)
                orch_result = orch.run(ingestion_results)
                raw_samples = [DatasetSample.from_dict(d) for d in (orch_result.accepted + orch_result.fix_required)]
                if getattr(orch_result, 'critic_scores', None):
                    _save_jsonl(
                        [cs.to_dict() for cs in orch_result.critic_scores],
                        run_dir / "critic_scores.jsonl",
                    )
            else:
                raw_samples = step_generate(cfg, ingestion_results)

            _save_jsonl([s.to_dict() for s in raw_samples], cfg.storage.raw_path())

            if self._is_cancel_requested(run_id):
                self._update_status(run_id, RunStatusEnum.CANCELED, error="Run canceled after generation", pipeline_stage="canceled")
                return

            # 3. Validate
            self._update_status(run_id, RunStatusEnum.RUNNING, pipeline_stage="validate")
            annotated = step_validate(cfg, raw_samples)

            # 4. Filter
            self._update_status(run_id, RunStatusEnum.RUNNING, pipeline_stage="filter")
            filtered, filter_report = step_filter(cfg, annotated)

            if self._is_cancel_requested(run_id):
                self._update_status(run_id, RunStatusEnum.CANCELED, error="Run canceled after filtering", pipeline_stage="canceled")
                return

            # 5. Evaluate
            self._update_status(run_id, RunStatusEnum.RUNNING, pipeline_stage="evaluate")
            step_evaluate(cfg, raw_samples, filtered, filter_report)

            # 6. Analyze
            self._update_status(run_id, RunStatusEnum.RUNNING, pipeline_stage="analyze")
            step_analyze(cfg, annotated)

            # Copy artifacts to run folder
            for src in [
                cfg.storage.raw_path(),
                cfg.storage.annotated_path(),
                cfg.storage.filtered_path(),
                cfg.storage.metrics_path(),
                cfg.storage.error_path(),
            ]:
                if src.exists():
                    shutil.copy2(src, run_dir / src.name)

            # Update symlink
            latest = cfg.storage.data_dir / "latest"
            try:
                if latest.is_symlink() or latest.exists():
                    latest.unlink()
                latest.symlink_to(Path("runs") / run_id)
            except OSError:
                pass

            outputs = {
                "raw": str(cfg.storage.raw_path()),
                "annotated": str(cfg.storage.annotated_path()),
                "filtered": str(cfg.storage.filtered_path()),
                "metrics": str(cfg.storage.metrics_path()),
                "error_analysis": str(cfg.storage.error_path()),
            }
            self._update_status(run_id, RunStatusEnum.SUCCEEDED, outputs=outputs, pipeline_stage="complete")

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Run %s failed: %s", run_id, tb)
            if self._is_cancel_requested(run_id):
                self._update_status(run_id, RunStatusEnum.CANCELED, error="Run canceled by user", pipeline_stage="canceled")
            else:
                self._update_status(run_id, RunStatusEnum.FAILED, error=str(exc), pipeline_stage="failed")

    def _is_cancel_requested(self, run_id: str) -> bool:
        with self._lock:
            run_status = self._runs.get(run_id)
            return bool(run_status and run_status.cancel_requested)

    def cancel_run(self, run_id: str) -> Optional[RunStatus]:
        with self._lock:
            run_status = self._runs.get(run_id)
            if not run_status:
                return None
            run_status.cancel_requested = True
            if run_status.status in {RunStatusEnum.PENDING, RunStatusEnum.RUNNING}:
                run_status.status = RunStatusEnum.CANCELED
                run_status.pipeline_stage = "canceled"
                run_status.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                run_status.error = "Run canceled by user"
            self._save_status(run_status)
            return run_status

    def _update_status(
        self,
        run_id: str,
        status: RunStatusEnum,
        error: Optional[str] = None,
        outputs: Optional[Dict[str, str]] = None,
        pipeline_stage: Optional[str] = None,
    ) -> None:
        with self._lock:
            run_status = self._runs.get(run_id)
            if not run_status:
                return
            run_status.status = status
            run_status.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            if pipeline_stage is not None:
                run_status.pipeline_stage = pipeline_stage
            if error is not None:
                run_status.error = error
            if outputs is not None:
                run_status.outputs = outputs
            self._save_status(run_status)


controller = PipelineController()
