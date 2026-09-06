"""OpenStax/open-textbook XML acquisition with deterministic problem-solution pairing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import httpx
from pydantic import BaseModel, ConfigDict, Field

from college_builder.domain.source import JsonLike, RawSourceRecord
from college_builder.source.base import AdapterCheckpoint, SourceDescriptor
from college_builder.source.gold import GoldRole, gold_role_metadata

NonEmptyStr = Annotated[str, Field(min_length=1)]


class OpenTextbookConfig(BaseModel):
    """Source-only configuration for an OpenStax or compatible CNXML textbook."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_file: NonEmptyStr
    source_url: NonEmptyStr
    gold_role: GoldRole
    dataset_name: NonEmptyStr = "openstax"
    revision: NonEmptyStr | None = None
    license_metadata: dict[str, JsonLike] = Field(default_factory=dict)


class OpenTextbookAdapter:
    """Acquire source-native textbook problem/solution pairs without rewriting."""

    def __init__(
        self,
        *,
        resume_from: AdapterCheckpoint | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._resume_from = resume_from
        self._timeout_seconds = timeout_seconds
        self._config: OpenTextbookConfig | None = None
        self._descriptor: SourceDescriptor | None = None
        self._checkpoint: AdapterCheckpoint | None = resume_from

    def discover(self, config: object) -> Iterable[SourceDescriptor]:
        validated = OpenTextbookConfig.model_validate(config).model_copy(deep=True)
        descriptor = SourceDescriptor(
            source_type="textbook",
            source_dataset=validated.dataset_name,
            source_url=validated.source_url,
            locator=validated.data_file,
            shard=validated.data_file,
            source_revision=validated.revision,
        )
        self._config = validated
        self._descriptor = descriptor
        self._validate_resume(descriptor)
        if self._resume_from is None:
            self._checkpoint = AdapterCheckpoint(
                descriptor=descriptor,
                next_row_offset=0,
                next_shard=descriptor.shard,
                source_revision=descriptor.source_revision,
            )
        return (descriptor,)

    def acquire(self, descriptor: SourceDescriptor) -> Iterable[RawSourceRecord]:
        config = self._require_config()
        if descriptor != self._descriptor:
            raise ValueError("descriptor was not produced by the current discover call")

        root = self._load_root(descriptor.locator)
        document_id = self._required_attribute(root, "id", "textbook document")
        textbook_title = self._textbook_title(root)
        pairs = self._exercise_pairs(root)
        start_offset = self._resume_from.next_row_offset if self._resume_from else 0
        if start_offset > len(pairs):
            raise ValueError("checkpoint row offset exceeds current OpenStax source length")

        for row_offset, (exercise_id, exercise, problem, solution) in enumerate(pairs):
            if row_offset < start_offset:
                continue
            record = self._map_pair(
                config,
                descriptor,
                row_offset,
                document_id,
                textbook_title,
                exercise_id,
                exercise,
                problem,
                solution,
            )
            self._checkpoint = AdapterCheckpoint(
                descriptor=descriptor,
                next_row_offset=row_offset + 1,
                next_shard=descriptor.shard,
                source_revision=descriptor.source_revision,
            )
            yield record

        self._checkpoint = AdapterCheckpoint(
            descriptor=descriptor,
            next_row_offset=len(pairs),
            next_shard=None,
            source_revision=descriptor.source_revision,
        )

    def checkpoint(self) -> AdapterCheckpoint:
        if self._checkpoint is None:
            raise RuntimeError("adapter has not discovered a source")
        return self._checkpoint

    def _map_pair(
        self,
        config: OpenTextbookConfig,
        descriptor: SourceDescriptor,
        row_offset: int,
        document_id: str,
        textbook_title: str,
        exercise_id: str,
        exercise: ET.Element,
        problem: ET.Element,
        solution: ET.Element,
    ) -> RawSourceRecord:
        problem_xml = ET.tostring(problem, encoding="unicode")
        solution_xml = ET.tostring(solution, encoding="unicode")
        exercise_xml = ET.tostring(exercise, encoding="unicode")
        source_id = f"{document_id}:{exercise_id}"
        raw_payload: dict[str, JsonLike] = {
            "document_id": document_id,
            "exercise_id": exercise_id,
            "textbook_title": textbook_title,
            "problem_xml": problem_xml,
            "solution_xml": solution_xml,
            "exercise_xml": exercise_xml,
        }
        metadata = gold_role_metadata(config.gold_role)
        metadata.update(
            {
                "document_id": document_id,
                "exercise_id": exercise_id,
                "textbook_title": textbook_title,
                "acquisition": {
                    "adapter": "open_textbook",
                    "row_offset": row_offset,
                    "shard": descriptor.shard,
                    "source_revision": descriptor.source_revision,
                    "source_descriptor": descriptor.model_dump(mode="json"),
                },
            }
        )

        return RawSourceRecord.model_validate(
            {
                "record_id": self._record_id(descriptor.source_dataset, source_id),
                "source_type": "textbook",
                "source_dataset": descriptor.source_dataset,
                "source_id": source_id,
                "source_url": descriptor.source_url,
                "raw_question": problem_xml,
                "raw_answer": "",
                "raw_analysis": solution_xml,
                "raw_payload": raw_payload,
                "metadata": metadata,
                "license_metadata": config.license_metadata,
                "raw_sha256": self._raw_sha256(raw_payload),
            }
        )

    def _load_root(self, locator: str) -> ET.Element:
        text = self._load_text(locator)
        try:
            return ET.fromstring(text)
        except ET.ParseError as exc:
            raise ValueError("invalid OpenStax/OpenTextbook XML") from exc

    def _load_text(self, locator: str) -> str:
        parsed = urlparse(locator)
        if parsed.scheme in {"http", "https"}:
            response = httpx.get(
                locator,
                timeout=self._timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.text
        if parsed.scheme:
            raise ValueError(f"unsupported OpenTextbook locator scheme: {parsed.scheme}")
        return Path(locator).read_text(encoding="utf-8")

    def _exercise_pairs(
        self,
        root: ET.Element,
    ) -> list[tuple[str, ET.Element, ET.Element, ET.Element]]:
        pairs: list[tuple[str, ET.Element, ET.Element, ET.Element]] = []
        seen_ids: set[str] = set()
        for exercise in root.iter():
            if self._local_name(exercise.tag) != "exercise":
                continue
            exercise_id = self._required_attribute(exercise, "id", "exercise")
            if exercise_id in seen_ids:
                raise ValueError(f"duplicate OpenTextbook exercise id: {exercise_id}")
            seen_ids.add(exercise_id)
            problems = [
                element
                for element in exercise
                if self._local_name(element.tag) == "problem"
            ]
            solutions = [
                element
                for element in exercise
                if self._local_name(element.tag) == "solution"
            ]
            if len(problems) != 1:
                raise ValueError(
                    f"exercise {exercise_id} must contain exactly one problem container"
                )
            if len(solutions) != 1:
                raise ValueError(
                    f"exercise {exercise_id} must contain exactly one solution container"
                )
            pairs.append((exercise_id, exercise, problems[0], solutions[0]))
        return pairs

    @classmethod
    def _textbook_title(cls, root: ET.Element) -> str:
        for child in root:
            if cls._local_name(child.tag) != "title":
                continue
            title = "".join(child.itertext()).strip()
            if title:
                return title
            break
        raise ValueError("OpenTextbook document must contain a non-empty title")

    @staticmethod
    def _required_attribute(element: ET.Element, name: str, label: str) -> str:
        value = element.attrib.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must contain a non-empty {name} attribute")
        return value

    @staticmethod
    def _local_name(tag: str) -> str:
        return tag.rsplit("}", maxsplit=1)[-1]

    def _validate_resume(self, descriptor: SourceDescriptor) -> None:
        if self._resume_from is None:
            return
        checkpoint = self._resume_from
        if checkpoint.descriptor != descriptor:
            raise ValueError("checkpoint descriptor does not match OpenTextbook source")
        if checkpoint.source_revision != descriptor.source_revision:
            raise ValueError("checkpoint source revision does not match OpenTextbook source")
        if checkpoint.next_shard not in {descriptor.shard, None}:
            raise ValueError("checkpoint shard does not match OpenTextbook source")

    @staticmethod
    def _record_id(dataset_name: str, source_id: str) -> str:
        identity = f"open_textbook\0{dataset_name}\0{source_id}".encode()
        return f"raw_open_textbook_{hashlib.sha256(identity).hexdigest()}"

    @staticmethod
    def _raw_sha256(payload: dict[str, JsonLike]) -> str:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _require_config(self) -> OpenTextbookConfig:
        if self._config is None:
            raise RuntimeError("discover must be called before acquire")
        return self._config
