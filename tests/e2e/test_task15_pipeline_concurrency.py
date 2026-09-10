from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from test_pipeline_resume import CountingAdapter, RuleProvider, _config, _processor, _record

from college_builder.config import ConcurrencyConfig
from college_builder.pipeline.runner import PipelineInterrupted, PipelineRunner
from college_builder.providers.base import ModelClassificationRequest, ModelDecision
from college_builder.storage.state import RunStage


class ConcurrentProbeProvider(RuleProvider):
    def __init__(self, *, provider: str, model: str) -> None:
        super().__init__(provider=provider, model=model)
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def classify(self, request: ModelClassificationRequest) -> ModelDecision:
        if request.task == "gate_1_university_stem":
            with self._lock:
                self._active += 1
                self.max_active = max(self.max_active, self._active)
            try:
                time.sleep(0.12)
                return super().classify(request)
            finally:
                with self._lock:
                    self._active -= 1
        return super().classify(request)


def test_classification_stage_honors_configured_concurrency(tmp_path: Path) -> None:
    config = _config().model_copy(
        update={
            "concurrency": ConcurrencyConfig(
                acquisition=1,
                normalization=1,
                rule_gate=1,
                classifier=2,
                verifier=1,
                embedding_batch_workers=1,
            )
        }
    )
    primary = ConcurrentProbeProvider(provider="fake", model="fixture-primary")
    verifier = RuleProvider(provider="fake", model="fixture-verifier")
    workspace = tmp_path / "workspace"
    runner = PipelineRunner(
        config=config,
        workspace=workspace,
        processor=_processor(
            config=config,
            workspace=workspace,
            primary=primary,
            verifier=verifier,
        ),
    )
    adapter = CountingAdapter(
        (_record(1, "VALID"), _record(2, "VALID")),
        revision="sha256:" + "1" * 64,
    )

    with pytest.raises(PipelineInterrupted) as interrupted:
        runner.run(
            adapter=adapter,
            source_config={},
            output_dir=tmp_path / "output",
            interrupt_after=RunStage.CLASSIFIED,
        )

    assert interrupted.value.stage is RunStage.CLASSIFIED
    assert primary.max_active >= 2
