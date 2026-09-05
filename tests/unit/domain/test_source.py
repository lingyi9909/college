import json
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from college_builder.domain.source import NormalizedQA, RawSourceRecord


def _raw_record() -> RawSourceRecord:
    return RawSourceRecord(
        record_id="raw_stackmathqa_42",
        source_type="dataset",
        source_dataset="stackmathqa",
        source_id="42",
        source_url="https://math.stackexchange.com/questions/42",
        raw_question="Q",
        raw_answer="A",
        raw_analysis="Because A",
        raw_payload={"id": 42, "nested": {"tags": ["linear-algebra"]}},
        metadata={"tags": ["linear-algebra"]},
        license_metadata={"declared": "UNREVIEWED"},
        raw_sha256="a" * 64,
    )


def test_raw_source_requires_stable_identity_and_raw_hash() -> None:
    record = _raw_record()

    assert record.source_id == "42"
    assert len(record.raw_sha256) == 64


def test_raw_source_rejects_invalid_sha256() -> None:
    with pytest.raises(ValidationError):
        RawSourceRecord(
            record_id="raw_stackmathqa_42",
            source_type="dataset",
            source_dataset="stackmathqa",
            source_id="42",
            source_url="https://math.stackexchange.com/questions/42",
            raw_question="Q",
            raw_answer="A",
            raw_analysis="Because A",
            raw_payload={"id": 42},
            metadata={},
            license_metadata={"declared": "UNREVIEWED"},
            raw_sha256="not-a-hash",
        )


def test_raw_source_is_deeply_immutable_after_construction() -> None:
    record = _raw_record()

    with pytest.raises(ValidationError):
        record.raw_question = "rewritten"  # type: ignore[misc]

    assert isinstance(record.raw_payload, MappingProxyType)
    with pytest.raises(TypeError):
        record.raw_payload["id"] = 43  # type: ignore[index]

    nested = record.raw_payload["nested"]
    assert isinstance(nested, MappingProxyType)
    with pytest.raises(TypeError):
        nested["tags"] = ("calculus",)  # type: ignore[index]


def test_raw_source_defensively_copies_mutable_input() -> None:
    raw_payload = {"id": 42, "nested": {"tags": ["linear-algebra"]}}
    record = RawSourceRecord(
        record_id="raw_stackmathqa_42",
        source_type="dataset",
        source_dataset="stackmathqa",
        source_id="42",
        source_url="https://math.stackexchange.com/questions/42",
        raw_question="Q",
        raw_answer="A",
        raw_analysis="Because A",
        raw_payload=raw_payload,
        metadata={},
        license_metadata={"declared": "UNREVIEWED"},
        raw_sha256="a" * 64,
    )

    raw_payload["id"] = 99
    raw_payload["nested"]["tags"].append("calculus")  # type: ignore[union-attr]

    assert record.raw_payload["id"] == 42
    assert record.raw_payload["nested"]["tags"] == ("linear-algebra",)  # type: ignore[index]


def test_normalized_qa_keeps_normalized_content_linked_to_raw_source() -> None:
    normalized = NormalizedQA(
        record_id="norm_stackmathqa_42",
        source_record_id="raw_stackmathqa_42",
        question="Normalized Q",
        answer="Normalized A",
        analysis="Normalized analysis",
        subject_candidates=("linear-algebra",),
        images=(),
        metadata={"language": "en"},
        normalization_evidence={"line_endings": "normalized"},
    )

    assert normalized.source_record_id == "raw_stackmathqa_42"
    assert normalized.question == "Normalized Q"
    assert "raw_question" not in type(normalized).model_fields


def _complex_provenance(seed: str) -> dict[str, object]:
    return {
        "root": [
            (
                {
                    "branch": [
                        {"seed": seed},
                        {"leaf": {"values": [1, 2, {"final": seed}]}},
                    ]
                },
            )
        ]
    }


def test_raw_source_deep_freezes_mapping_and_list_tuple_nesting() -> None:
    nested_mapping = MappingProxyType(
        {
            "root": [
                (
                    {
                        "branch": [
                            {"seed": "raw"},
                            {"leaf": MappingProxyType({"values": [1, {"final": "raw"}]})},
                        ]
                    },
                )
            ]
        }
    )
    record = RawSourceRecord(
        record_id="raw_nested_1",
        source_type="dataset",
        source_dataset="fixture",
        source_id="1",
        source_url="https://example.test/1",
        raw_question="Q",
        raw_answer="A",
        raw_analysis="Analysis",
        raw_payload=nested_mapping,
        metadata={},
        license_metadata={"declared": "UNREVIEWED"},
        raw_sha256="b" * 64,
    )

    root = record.raw_payload["root"]
    assert isinstance(root, tuple)
    level2 = root[0]
    assert isinstance(level2, tuple)
    level3 = level2[0]
    assert isinstance(level3, MappingProxyType)
    branch = level3["branch"]
    assert isinstance(branch, tuple)
    assert isinstance(branch[0], MappingProxyType)
    leaf = branch[1]["leaf"]  # type: ignore[index]
    assert isinstance(leaf, MappingProxyType)
    values = leaf["values"]
    assert isinstance(values, tuple)
    assert isinstance(values[1], MappingProxyType)

    with pytest.raises(TypeError):
        branch[0]["seed"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        values[1]["final"] = "changed"  # type: ignore[index]


def test_all_raw_provenance_containers_are_defensive_deep_copies() -> None:
    raw_payload = _complex_provenance("raw")
    metadata = _complex_provenance("metadata")
    license_metadata = _complex_provenance("license")

    record = RawSourceRecord(
        record_id="raw_nested_2",
        source_type="dataset",
        source_dataset="fixture",
        source_id="2",
        source_url="https://example.test/2",
        raw_question="Q",
        raw_answer="A",
        raw_analysis="Analysis",
        raw_payload=raw_payload,
        metadata=metadata,
        license_metadata=license_metadata,
        raw_sha256="c" * 64,
    )

    for source in (raw_payload, metadata, license_metadata):
        source["root"].append("external-change")  # type: ignore[union-attr]
        source["root"][0][0]["branch"][0]["seed"] = "mutated"  # type: ignore[index]

    containers = {
        "raw": record.raw_payload,
        "metadata": record.metadata,
        "license": record.license_metadata,
    }
    for expected_seed, container in containers.items():
        root = container["root"]
        assert isinstance(root, tuple)
        branch = root[0][0]["branch"]  # type: ignore[index]
        assert isinstance(branch, tuple)
        assert branch[0]["seed"] == expected_seed  # type: ignore[index]
        assert len(root) == 1
        with pytest.raises(TypeError):
            branch[0]["seed"] = "record-mutation"  # type: ignore[index]


def test_model_dump_json_restores_plain_json_objects_and_lists() -> None:
    record = RawSourceRecord(
        record_id="raw_nested_3",
        source_type="dataset",
        source_dataset="fixture",
        source_id="3",
        source_url="https://example.test/3",
        raw_question="Q",
        raw_answer="A",
        raw_analysis="Analysis",
        raw_payload=_complex_provenance("raw"),
        metadata=_complex_provenance("metadata"),
        license_metadata=_complex_provenance("license"),
        raw_sha256="d" * 64,
    )

    dumped = record.model_dump(mode="json")

    for key in ("raw_payload", "metadata", "license_metadata"):
        container = dumped[key]
        assert type(container) is dict
        assert type(container["root"]) is list
        assert type(container["root"][0]) is list
        assert type(container["root"][0][0]) is dict
        assert type(container["root"][0][0]["branch"]) is list
        assert type(container["root"][0][0]["branch"][1]["leaf"]) is dict
        assert type(container["root"][0][0]["branch"][1]["leaf"]["values"]) is list

    serialized = json.dumps(dumped)
    assert json.loads(serialized) == dumped
