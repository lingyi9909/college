from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

import pytest

from college_builder.domain.source import RawSourceRecord
from college_builder.source.base import AdapterCheckpoint, SourceAdapter, SourceDescriptor


class FakeAdapter:
    def __init__(self) -> None:
        self._checkpoint: AdapterCheckpoint | None = None

    def discover(self, config: object) -> Iterable[SourceDescriptor]:
        descriptor = SourceDescriptor(
            source_type="fixture",
            source_dataset="fake-university-source",
            source_url="https://example.test/datasets/fake",
            locator="fixture://fake.jsonl",
            shard="fake.jsonl",
            source_revision="fake-rev-1",
        )
        self._checkpoint = AdapterCheckpoint(
            descriptor=descriptor,
            next_row_offset=0,
            next_shard=descriptor.shard,
            source_revision=descriptor.source_revision,
        )
        return (descriptor,)

    def acquire(self, descriptor: SourceDescriptor) -> Iterable[RawSourceRecord]:
        raw_payload = {
            "id": "42",
            "question": "Original question",
            "answer": "Original answer",
            "analysis": "Original analysis",
            "nested": {"tags": ["linear-algebra"]},
        }
        acquisition = {
            "adapter": "fake",
            "source_revision": descriptor.source_revision,
            "row_offset": 0,
        }
        license_metadata = {"declared": "CC-BY-4.0", "status": "reviewed"}
        raw_sha256 = hashlib.sha256(
            json.dumps(
                raw_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        record = RawSourceRecord(
            record_id="raw_fake_42",
            source_type="dataset",
            source_dataset="fake-university-source",
            source_id="42",
            source_url="https://example.test/questions/42",
            raw_question="Original question",
            raw_answer="Original answer",
            raw_analysis="Original analysis",
            raw_payload=raw_payload,
            metadata={"acquisition": acquisition},
            license_metadata=license_metadata,
            raw_sha256=raw_sha256,
        )

        raw_payload["nested"]["tags"].append("mutated-after-record")
        acquisition["row_offset"] = 999
        license_metadata["declared"] = "mutated-after-record"
        self._checkpoint = AdapterCheckpoint(
            descriptor=descriptor,
            next_row_offset=1,
            next_shard=descriptor.shard,
            source_revision=descriptor.source_revision,
        )
        return (record,)

    def checkpoint(self) -> AdapterCheckpoint:
        if self._checkpoint is None:
            raise RuntimeError("adapter has not discovered a source")
        return self._checkpoint


def _one_record(adapter: FakeAdapter) -> tuple[SourceDescriptor, RawSourceRecord]:
    descriptor = tuple(adapter.discover({}))[0]
    record = tuple(adapter.acquire(descriptor))[0]
    return descriptor, record


def test_source_adapter_contract_is_runtime_checkable_and_stable() -> None:
    first_adapter = FakeAdapter()
    second_adapter = FakeAdapter()

    assert isinstance(first_adapter, SourceAdapter)

    first_descriptor, first = _one_record(first_adapter)
    second_descriptor, second = _one_record(second_adapter)

    assert first_descriptor == second_descriptor
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.record_id == second.record_id
    assert first.raw_sha256 == second.raw_sha256


def test_adapter_boundary_preserves_immutable_source_faithful_provenance() -> None:
    adapter = FakeAdapter()
    descriptor, record = _one_record(adapter)
    dumped = record.model_dump(mode="json")

    assert record.source_dataset == "fake-university-source"
    assert record.source_id == "42"
    assert record.source_url == "https://example.test/questions/42"
    assert dumped["raw_payload"]["nested"]["tags"] == ["linear-algebra"]
    assert dumped["license_metadata"]["declared"] == "CC-BY-4.0"
    assert dumped["metadata"]["acquisition"] == {
        "adapter": "fake",
        "row_offset": 0,
        "source_revision": "fake-rev-1",
    }
    assert len(record.raw_sha256) == 64

    with pytest.raises(TypeError):
        record.raw_payload["new"] = "forbidden"  # type: ignore[index]

    checkpoint = adapter.checkpoint()
    assert checkpoint.descriptor == descriptor
    assert checkpoint.next_row_offset == 1
    assert checkpoint.next_shard == "fake.jsonl"
    assert checkpoint.source_revision == "fake-rev-1"
