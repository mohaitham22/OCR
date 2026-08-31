"""Field-level accuracy harness -- the script that decides which engine we ship.

Point it at a folder of documents, each sitting next to a same-stem `.json`
file holding the fields a human checked, and it runs every requested engine
through `app.pipeline.process` and reports, per engine: field-level accuracy,
median latency, how many documents produced no extraction at all, and the five
fields it is weakest on.

Field accuracy, not character error rate, because CER answers a question
nobody filing a document is asking. An engine can be 98% correct character by
character and still get 40% of totals wrong: the characters a recogniser
misses are not spread evenly across a page, they cluster on digits, and a
single misread digit turns a correct-looking transcription into a wrong total.
Character accuracy would reward that engine anyway. Field accuracy asks the
only question that matters here -- did this cell end up right -- and it is
computed after `normalise` removes the formatting differences that are not
extraction errors, so an engine is not marked wrong for writing "12.50" where
the label says "12.5".

This harness is only as good as the labels it is run against. If `samples/`
is empty or has documents with no matching `.json`, `main` says so and exits
without running anything: inventing ground truth, or falling back to a
synthetic corpus, would produce a report that looks like it means something
and does not. A run under about 100 documents proves the harness works, not
that either engine does; `main` says that too.
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from dataclasses import dataclass, field
from json import JSONDecodeError
from json import loads as json_loads
from pathlib import Path
from typing import Any

from app import db
from app.engines import ENGINE_KEYS
from app.pipeline import process
from app.schemas import DOC_TYPES

logger = logging.getLogger(__name__)

DOCUMENT_SUFFIXES: tuple[str, ...] = (
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)


@dataclass(slots=True)
class Sample:
    """One document paired with the fields a human confirmed for it."""

    path: Path
    expected: dict[str, Any]


@dataclass(slots=True)
class FieldResult:
    """One cell of the per-document per-field CSV."""

    document: str
    engine: str
    field: str
    expected: Any
    extracted: Any
    correct: bool


@dataclass(slots=True)
class EngineSummary:
    """What one engine did across the whole folder."""

    engine: str
    document_count: int = 0
    failed_count: int = 0
    field_count: int = 0
    correct_count: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    # field key -> [correct, total], keyed by _field_key so "items.2.total" and
    # "items.7.total" from two different documents count as the same field.
    field_stats: dict[str, list[int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float | None:
        return self.correct_count / self.field_count if self.field_count else None

    @property
    def median_latency_ms(self) -> float | None:
        return statistics.median(self.latencies_ms) if self.latencies_ms else None

    def record(self, path: str, correct: bool) -> None:
        self.field_count += 1
        self.correct_count += int(correct)
        counts = self.field_stats.setdefault(_field_key(path), [0, 0])
        counts[1] += 1
        counts[0] += int(correct)

    def weakest_fields(self, n: int) -> list[tuple[str, float, int]]:
        """The `n` lowest-accuracy fields, as (field, accuracy, sample count).

        Ranked by accuracy alone, not by accuracy weighted by volume: a field
        that is wrong every time it appears is exactly what someone deciding
        between engines needs to see, even if it only appears a handful of
        times in this folder.
        """
        scored = [(key, correct / total, total) for key, (correct, total) in self.field_stats.items()]
        scored.sort(key=lambda item: (item[1], -item[2]))
        return scored[:n]


def normalise(value: Any) -> Any:
    """Reduce a value to the form a person compares by eye, not by string equality.

    Money rounds to two places so 12.5 and "12.50" agree. Text is case-folded
    and has its whitespace collapsed so "Total  Mart" and "total mart" agree. A
    numeric-looking string coerces to a number so "12.50" and 12.5 agree even
    though one came from a human's JSON and the other from a Pydantic model.
    None of these are extraction errors, and counting them as errors would make
    every engine look worse than it is by an amount that has nothing to do with
    whether the number on the page was read correctly.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if isinstance(value, str):
        text = " ".join(value.strip().split())
        if not text:
            return None
        try:
            return round(float(text), 2)
        except ValueError:
            return text.casefold()
    return value


def _field_key(path: str) -> str:
    """Collapse a dotted path to a field name comparable across documents.

    `items.0.unit_price` on one document and `items.3.unit_price` on another
    are the same field; the line number is an artefact of that one document's
    layout, not something the next document will share. Stripping digit
    segments is what lets "weakest fields" mean something across a folder
    instead of naming a line index that happens to appear once.
    """
    return ".".join(part for part in path.split(".") if not part.isdigit())


def discover_samples(folder: Path) -> list[Sample]:
    """Documents in `folder` that have a same-stem `.json` label sitting beside them.

    A document with no label is logged and skipped rather than silently
    dropped, since a folder that is mostly unlabelled is the folder `main`
    refuses to run against.
    """
    samples: list[Sample] = []
    unlabelled: list[str] = []
    for doc_path in sorted(folder.iterdir()):
        if not doc_path.is_file() or doc_path.suffix.lower() not in DOCUMENT_SUFFIXES:
            continue
        label_path = doc_path.with_suffix(".json")
        if not label_path.exists():
            unlabelled.append(doc_path.name)
            continue
        try:
            expected = json_loads(label_path.read_text(encoding="utf-8"))
        except (OSError, JSONDecodeError) as exc:
            logger.warning("could not read %s: %s", label_path, exc)
            unlabelled.append(doc_path.name)
            continue
        if not isinstance(expected, dict):
            logger.warning("%s does not contain a JSON object; skipping", label_path)
            unlabelled.append(doc_path.name)
            continue
        samples.append(Sample(path=doc_path, expected=expected))

    if unlabelled:
        logger.warning(
            "%d document(s) with no usable .json label, skipped: %s",
            len(unlabelled),
            ", ".join(unlabelled),
        )
    return samples


def evaluate(
    samples: list[Sample],
    engine_keys: list[str],
    doc_type: str,
    *,
    clean_images: bool,
) -> tuple[list[FieldResult], dict[str, EngineSummary]]:
    """Run every engine over every sample and score the fields.

    Each (sample, engine) pair is one call to `app.pipeline.process` with
    `persist=False` -- an eval run is not a record and must not write rows a
    reviewer would later have to account for.
    """
    rows: list[FieldResult] = []
    summaries = {key: EngineSummary(engine=key) for key in engine_keys}

    for sample in samples:
        data = sample.path.read_bytes()
        expected_flat = db.flatten_fields(sample.expected)

        for engine_key in engine_keys:
            summary = summaries[engine_key]
            summary.document_count += 1

            result = process(
                data,
                sample.path.name,
                doc_type,
                engine_key,
                clean_images=clean_images,
                persist=False,
            )
            if result.duration_ms is not None:
                summary.latencies_ms.append(result.duration_ms)

            document = result.extraction.document if result.extraction is not None else None
            if result.status == "failed" or document is None:
                summary.failed_count += 1
            extracted_flat = db.flatten_fields(document) if document is not None else {}

            # `db._path_key` rather than a plain string sort: it is the same
            # ordering `app.pipeline.field_diff` uses, so "items.2" comes
            # before "items.10" here the same way it does in the compare-mode
            # table this report sits beside.
            for path in sorted(set(expected_flat) | set(extracted_flat), key=db._path_key):
                expected_value = expected_flat.get(path)
                extracted_value = extracted_flat.get(path)
                correct = normalise(expected_value) == normalise(extracted_value)
                summary.record(path, correct)
                rows.append(
                    FieldResult(
                        document=sample.path.name,
                        engine=engine_key,
                        field=path,
                        expected=expected_value,
                        extracted=extracted_value,
                        correct=correct,
                    )
                )
    return rows, summaries


def write_csv(path: Path, rows: list[FieldResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["document", "engine", "field", "expected", "extracted", "correct"])
        for row in rows:
            writer.writerow([row.document, row.engine, row.field, row.expected, row.extracted, row.correct])


def _log_summary(summary: EngineSummary) -> None:
    accuracy = summary.accuracy
    latency = summary.median_latency_ms
    logger.info(
        "%-20s accuracy=%-7s (%d/%d fields)  median_latency=%-8s  failed=%d/%d documents",
        summary.engine,
        f"{accuracy:.1%}" if accuracy is not None else "n/a",
        summary.correct_count,
        summary.field_count,
        f"{latency:.0f} ms" if latency is not None else "n/a",
        summary.failed_count,
        summary.document_count,
    )
    weakest = summary.weakest_fields(5)
    if weakest:
        logger.info(
            "%-20s weakest fields: %s",
            "",
            ", ".join(f"{key} ({acc:.0%}, n={n})" for key, acc, n in weakest),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more extraction engines against a folder of labelled documents "
            "and report field-level accuracy, latency, and failures per engine."
        )
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder of documents, each next to a same-stem .json file of correct fields.",
    )
    parser.add_argument(
        "--engines",
        default=",".join(ENGINE_KEYS),
        help=f"Comma-separated engine keys to run. Default: all of {', '.join(ENGINE_KEYS)}.",
    )
    parser.add_argument(
        "--doc-type",
        required=True,
        choices=sorted(DOC_TYPES),
        help="Schema every document in the folder is scored against.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip the crop/deskew/lighting chain before extraction.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval_results.csv"),
        help="Where to write the per-document per-field CSV. Default: eval_results.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)

    if not args.folder.exists():
        logger.error("%s does not exist", args.folder)
        return 1
    if not args.folder.is_dir():
        logger.error("%s is not a folder", args.folder)
        return 1

    samples = discover_samples(args.folder)
    if not samples:
        logger.error(
            "%s has no labelled documents -- a document needs a same-stem .json file of "
            "human-checked fields beside it before it counts. Not running: a harness "
            "validated against invented ground truth is worse than no harness. Add "
            "labelled samples and try again.",
            args.folder,
        )
        return 1

    engine_keys = [key.strip() for key in args.engines.split(",") if key.strip()]
    unknown = [key for key in engine_keys if key not in ENGINE_KEYS]
    if unknown:
        logger.error("unknown engine(s) %s; expected one of %s", unknown, ", ".join(ENGINE_KEYS))
        return 1

    rows, summaries = evaluate(samples, engine_keys, args.doc_type, clean_images=not args.no_clean)
    write_csv(args.out, rows)

    logger.info("%d document(s) evaluated from %s", len(samples), args.folder)
    if len(samples) < 100:
        logger.info("fewer than 100 documents: this run proves the harness, not the engines.")
    for key in engine_keys:
        _log_summary(summaries[key])
    logger.info("per-document per-field detail written to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
