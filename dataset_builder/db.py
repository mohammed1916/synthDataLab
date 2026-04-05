from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from dataset_builder.config import Config

logger = logging.getLogger(__name__)
Base = declarative_base()


class RunStatusModel(Base):
    __tablename__ = "run_status"

    run_id = Column(String(64), primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(32), nullable=False)
    input_path = Column(Text, nullable=True)
    input_text = Column(Text, nullable=True)
    mock = Column(Boolean, nullable=False, default=False)
    workers = Column(Integer, nullable=False, default=1)
    agent = Column(Boolean, nullable=False, default=False)
    steering = Column(String(32), nullable=False, default="auto")
    threshold = Column(Float, nullable=False, default=0.7)
    force = Column(Boolean, nullable=False, default=False)
    reset_fingerprints = Column(Boolean, nullable=False, default=False)
    cancel_requested = Column(Boolean, nullable=False, default=False)
    error = Column(Text, nullable=True)
    run_dir = Column(Text, nullable=True)
    pipeline_stage = Column(String(32), nullable=True)
    outputs = Column(JSON, nullable=True)


class DataSampleModel(Base):
    __tablename__ = "data_samples"

    id = Column(String(64), primary_key=True)
    run_id = Column(String(64), ForeignKey("run_status.run_id"), nullable=False, index=True)
    stage = Column(String(32), nullable=False, index=True)
    sample_id = Column(String(64), nullable=False, index=True)
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class DatabaseManager:
    """Simple SQLAlchemy-backed persistence for run metadata and sample payloads."""

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or ""
        self.enabled = bool(self.database_url)
        self._engine = None
        self._session_factory = None

        if self.enabled:
            self._engine = create_engine(self.database_url, future=True)
            self._session_factory = sessionmaker(bind=self._engine, future=True)

    @classmethod
    def from_config(cls, config: Config) -> "DatabaseManager":
        return cls(config.database.url)

    def create_tables(self) -> None:
        if not self.enabled:
            return
        Base.metadata.create_all(self._engine)

    def session(self) -> Session:
        if not self.enabled or self._session_factory is None:
            raise RuntimeError("Database is not enabled")
        return self._session_factory()

    def health(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        try:
            with self.session() as session:
                session.execute(select(func.count()).select_from(RunStatusModel))
            return {"enabled": True, "status": "connected"}
        except SQLAlchemyError as exc:
            logger.warning("Database health check failed: %s", exc)
            return {"enabled": True, "status": "error", "error": str(exc)}

    def save_run(self, run_status: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            with self.session() as session:
                existing = session.get(RunStatusModel, run_status["run_id"])
                if existing is None:
                    existing = RunStatusModel(run_id=run_status["run_id"])
                existing.created_at = datetime.fromisoformat(run_status["created_at"])
                existing.updated_at = datetime.fromisoformat(run_status["updated_at"])
                existing.status = run_status["status"]
                existing.input_path = run_status.get("input_path")
                existing.input_text = run_status.get("input_text")
                existing.mock = run_status.get("mock", False)
                existing.workers = run_status.get("workers", 1)
                existing.agent = run_status.get("agent", False)
                existing.steering = run_status.get("steering", "auto")
                existing.threshold = float(run_status.get("threshold", 0.7))
                existing.force = run_status.get("force", False)
                existing.reset_fingerprints = run_status.get("reset_fingerprints", False)
                existing.cancel_requested = run_status.get("cancel_requested", False)
                existing.error = run_status.get("error")
                existing.run_dir = run_status.get("run_dir")
                existing.pipeline_stage = run_status.get("pipeline_stage")
                existing.outputs = run_status.get("outputs")
                session.add(existing)
                session.commit()
        except SQLAlchemyError as exc:
            logger.warning("Failed to save run status to DB: %s", exc)

    def load_runs(self) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            with self.session() as session:
                rows = session.execute(select(RunStatusModel)).scalars().all()
                return [self._model_to_dict(row) for row in rows]
        except SQLAlchemyError as exc:
            logger.warning("Failed to load runs from DB: %s", exc)
            return []

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        try:
            with self.session() as session:
                row = session.get(RunStatusModel, run_id)
                return self._model_to_dict(row) if row else None
        except SQLAlchemyError as exc:
            logger.warning("Failed to load run from DB: %s", exc)
            return None

    def save_samples(self, run_id: str, stage: str, samples: Iterable[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        try:
            batch = []
            for sample in samples:
                batch.append(
                    DataSampleModel(
                        id=uuid.uuid4().hex,
                        run_id=run_id,
                        stage=stage,
                        sample_id=str(sample.get("id", "")),
                        payload=sample,
                        created_at=datetime.now(timezone.utc),
                    )
                )
            with self.session() as session:
                session.add_all(batch)
                session.commit()
        except SQLAlchemyError as exc:
            logger.warning("Failed to persist %s samples to DB: %s", stage, exc)

    def sample_counts(self, run_id: str) -> dict[str, int]:
        if not self.enabled:
            return {"raw": 0, "annotated": 0, "filtered": 0}
        try:
            with self.session() as session:
                counts = session.execute(
                    select(DataSampleModel.stage, func.count())
                    .where(DataSampleModel.run_id == run_id)
                    .group_by(DataSampleModel.stage)
                ).all()
            result = {"raw": 0, "annotated": 0, "filtered": 0}
            for stage, count in counts:
                result[stage] = count
            return result
        except SQLAlchemyError as exc:
            logger.warning("Failed to count samples in DB: %s", exc)
            return {"raw": 0, "annotated": 0, "filtered": 0}

    @staticmethod
    def _model_to_dict(model: RunStatusModel | None) -> dict[str, Any] | None:
        if model is None:
            return None
        return {
            "run_id": model.run_id,
            "created_at": model.created_at.isoformat(),
            "updated_at": model.updated_at.isoformat(),
            "status": model.status,
            "input_path": model.input_path,
            "input_text": model.input_text,
            "mock": model.mock,
            "workers": model.workers,
            "agent": model.agent,
            "steering": model.steering,
            "threshold": model.threshold,
            "force": model.force,
            "reset_fingerprints": model.reset_fingerprints,
            "cancel_requested": model.cancel_requested,
            "error": model.error,
            "run_dir": model.run_dir,
            "pipeline_stage": model.pipeline_stage,
            "outputs": model.outputs,
        }


db_manager = DatabaseManager.from_config(Config())
