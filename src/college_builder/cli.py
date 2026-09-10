"""CLI entrypoints for Task 14 pipeline execution, resume, report, and config validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import typer

from college_builder.config import PipelineConfig, ProviderConfig
from college_builder.domain.source import RawSourceRecord
from college_builder.pipeline.runner import GatePipelineProcessor, PipelineRunner
from college_builder.providers.base import StructuredModelProvider
from college_builder.providers.openai_compatible import OpenAICompatibleStructuredModelProvider
from college_builder.reporting.pilot_report import load_pilot_report
from college_builder.source.base import AdapterCheckpoint, SourceDescriptor

app = typer.Typer(no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True)
app.add_typer(config_app, name="config")


class _LocalJsonlAdapter:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise ValueError(f"input JSONL does not exist: {self._path}")
        revision = hashlib.sha256(self._path.read_bytes()).hexdigest()
        self._descriptor = SourceDescriptor(
            source_type="dataset",
            source_dataset=self._path.stem,
            source_url=self._path.resolve().as_uri(),
            locator=str(self._path.resolve()),
            source_revision=revision,
        )
        self._next_row_offset = 0

    def discover(self, config: object) -> Iterable[SourceDescriptor]:
        return (self._descriptor,)

    def acquire(self, descriptor: SourceDescriptor) -> Iterable[RawSourceRecord]:
        if descriptor != self._descriptor:
            raise ValueError("local adapter descriptor mismatch")
        for index, line in enumerate(self._path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            record = RawSourceRecord.model_validate_json(line)
            self._next_row_offset = index + 1
            yield record

    def checkpoint(self) -> AdapterCheckpoint:
        return AdapterCheckpoint(
            descriptor=self._descriptor,
            next_row_offset=self._next_row_offset,
            next_shard=None,
            source_revision=self._descriptor.source_revision,
        )


def _provider(config: ProviderConfig) -> StructuredModelProvider:
    if config.name == "openai_compatible":
        return OpenAICompatibleStructuredModelProvider(model=config.model)
    raise ValueError("the fake provider is test-only; CLI run/resume requires openai_compatible")


def _runner(config: PipelineConfig, workspace: Path) -> PipelineRunner:
    primary = _provider(config.providers.classifier)
    verifier = _provider(config.providers.verifier)
    processor = GatePipelineProcessor(
        config=config,
        cache_path=workspace / "cache" / "calls.sqlite3",
        primary_provider=primary,
        verifier_provider=verifier,
        project_root=Path("."),
    )
    return PipelineRunner(config=config, workspace=workspace, processor=processor)


@app.command("run")
def run_command(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, dir_okay=False),
    ],
    config_path: Annotated[Path, typer.Option("--config")] = Path("config/pilot.yaml"),
    workspace: Annotated[Path, typer.Option("--workspace")] = Path("artifacts/workspace"),
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/output"),
) -> None:
    """Run a local RawSourceRecord JSONL snapshot through the complete pipeline."""
    config = PipelineConfig.load(config_path)
    result = _runner(config, workspace).run(
        adapter=_LocalJsonlAdapter(input_path),
        source_config={},
        output_dir=output,
    )
    typer.echo(f"run_id={result.run_id}")
    typer.echo(f"questions={result.output_path}")
    typer.echo(f"report={result.report_path}")


@app.command("resume")
def resume_command(
    run_id: Annotated[str, typer.Option("--run-id")],
    config_path: Annotated[Path, typer.Option("--config")] = Path("config/pilot.yaml"),
    workspace: Annotated[Path, typer.Option("--workspace")] = Path("artifacts/workspace"),
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/output"),
) -> None:
    """Resume an interrupted run from its persisted stage state and artifacts."""
    config = PipelineConfig.load(config_path)
    result = _runner(config, workspace).resume(run_id=run_id, output_dir=output)
    typer.echo(f"run_id={result.run_id}")
    typer.echo(f"questions={result.output_path}")
    typer.echo(f"report={result.report_path}")


@app.command("report")
def report_command(
    run_id: Annotated[str, typer.Option("--run-id")],
    workspace: Annotated[Path, typer.Option("--workspace")] = Path("artifacts/workspace"),
) -> None:
    """Print the deterministic JSON report for one run."""
    path = workspace / "runs" / run_id / "run_report.json"
    report = load_pilot_report(path)
    typer.echo(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


@config_app.command("validate")
def validate_config(config_path: Path) -> None:
    """Validate a configuration before any source acquisition."""
    config = PipelineConfig.load(config_path)
    typer.echo(config.config_version)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
