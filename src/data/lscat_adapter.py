"""Import the official LS-CAT runtime archive into the selector schema."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import zipfile

from data.pilot_adapter import NLP_THREAD_COUNTS


def _pair(value: str, *, field: str) -> tuple[int, int]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if (
        not isinstance(parsed, tuple)
        or len(parsed) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in parsed)
        or min(parsed) <= 0
    ):
        raise ValueError(f"invalid {field}: {value!r}")
    return parsed


def _source_identity(path: str, function: str, source_root: Path) -> tuple[str, str, Path]:
    relative = Path(path)
    parts = relative.parts
    if parts and parts[0] == "step2":
        relative = Path(*parts[1:])
    if relative.name != "main.cu" or len(relative.parts) < 3:
        raise ValueError(f"unexpected LS-CAT harness path: {path!r}")
    source = source_root / relative.parent / f"{function}.cu"
    if not source.is_file() or not source.resolve().is_relative_to(source_root.resolve()):
        raise ValueError(f"missing LS-CAT kernel source: {source}")
    kernel_id = (relative.parent / function).as_posix()
    split_group = relative.parts[0]
    return kernel_id, split_group, source


def import_lscat_1d(
    repository: str | Path,
    output_directory: str | Path,
    *,
    archive: str | Path | None = None,
) -> dict:
    """Create a complete 16-action T4 subset without modifying the source archive."""

    repository = Path(repository).resolve()
    output = Path(output_directory).resolve()
    source_root = repository / "data" / "kernels"
    archive_path = Path(archive).resolve() if archive else (
        repository / "results" / "archive" / "T4_18-6-3_sm52.zip"
    )
    if not source_root.is_dir():
        raise FileNotFoundError(f"LS-CAT kernel directory not found: {source_root}")
    if not archive_path.is_file():
        raise FileNotFoundError(f"LS-CAT T4 archive not found: {archive_path}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"choose an empty output directory: {output}")

    output.mkdir(parents=True, exist_ok=True)
    runtime_path = output / "runtimes.csv"
    group_path = output / "split_groups.json"
    source_path = output / "sources.json"
    metadata_path = output / "dataset.json"

    groups: dict[str, str] = {}
    sources: dict[str, str] = {}
    archive_rows = one_d_rows = rejected_rows = 0

    try:
        with tempfile.TemporaryDirectory(prefix="lscat-import-") as temporary:
            database_path = Path(temporary) / "measurements.sqlite3"
            connection = sqlite3.connect(database_path)
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE measurements ("
                "kernel_id TEXT, m INTEGER, n INTEGER, threads INTEGER, runtime REAL)"
            )
            pending = []
            with zipfile.ZipFile(archive_path) as zipped:
                csv_names = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
                if len(csv_names) != 1:
                    raise ValueError(f"expected one CSV in {archive_path}, found {csv_names}")
                with zipped.open(csv_names[0]) as raw:
                    reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
                    required = {"path", "function", "blocks", "matrix", "time"}
                    missing = required.difference(reader.fieldnames or ())
                    if missing:
                        raise ValueError(f"LS-CAT archive is missing columns: {sorted(missing)}")
                    for row_number, row in enumerate(reader, start=2):
                        archive_rows += 1
                        try:
                            block_a, block_b = _pair(row["blocks"], field="blocks")
                        except (SyntaxError, TypeError, ValueError):
                            # The official archive uses zero-filled placeholder rows for
                            # kernels that did not produce measurements.
                            rejected_rows += 1
                            continue
                        if block_a != 1 or block_b not in NLP_THREAD_COUNTS:
                            continue
                        try:
                            m, n = _pair(row["matrix"], field="matrix")
                            runtime = float(row["time"])
                            if not math.isfinite(runtime) or runtime <= 0:
                                rejected_rows += 1
                                continue
                            kernel_id, split_group, source = _source_identity(
                                row["path"], row["function"], source_root
                            )
                        except (TypeError, ValueError) as exc:
                            raise ValueError(f"invalid LS-CAT row {row_number}: {exc}") from exc
                        previous_group = groups.setdefault(kernel_id, split_group)
                        previous_source = sources.setdefault(
                            kernel_id, source.relative_to(repository).as_posix()
                        )
                        if previous_group != split_group or previous_source != source.relative_to(repository).as_posix():
                            raise ValueError(f"inconsistent identity for kernel {kernel_id}")
                        pending.append((kernel_id, m, n, block_b, runtime))
                        one_d_rows += 1
                        if len(pending) == 10_000:
                            connection.executemany(
                                "INSERT INTO measurements VALUES (?, ?, ?, ?, ?)", pending
                            )
                            pending.clear()
                    if pending:
                        connection.executemany(
                            "INSERT INTO measurements VALUES (?, ?, ?, ?, ?)", pending
                        )
            connection.commit()
            connection.execute(
                "CREATE TABLE averaged AS "
                "SELECT kernel_id, m, n, threads, AVG(runtime) AS runtime "
                "FROM measurements GROUP BY kernel_id, m, n, threads"
            )
            connection.execute(
                "CREATE INDEX averaged_context ON averaged(kernel_id, m, n)"
            )
            connection.execute(
                "CREATE TABLE complete AS "
                "SELECT kernel_id, m, n FROM averaged "
                "GROUP BY kernel_id, m, n HAVING COUNT(*) = ?",
                (len(NLP_THREAD_COUNTS),),
            )
            complete_contexts = int(connection.execute("SELECT COUNT(*) FROM complete").fetchone()[0])
            fields = (
                "kernel_id", "m", "n", "block_x", "block_y", "block_z",
                "runtime_us", "split_group",
            )
            runtime_rows = 0
            used_kernels = set()
            with runtime_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                query = (
                    "SELECT a.kernel_id, a.m, a.n, a.threads, a.runtime "
                    "FROM averaged a JOIN complete c USING(kernel_id, m, n) "
                    "ORDER BY a.kernel_id, a.m, a.n, a.threads"
                )
                for kernel_id, m, n, threads, runtime in connection.execute(query):
                    writer.writerow(
                        (kernel_id, m, n, threads, 1, 1, runtime, groups[kernel_id])
                    )
                    used_kernels.add(kernel_id)
                    runtime_rows += 1
            connection.close()

        used_groups = {kernel: groups[kernel] for kernel in sorted(used_kernels)}
        used_sources = {kernel: sources[kernel] for kernel in sorted(used_kernels)}
        group_path.write_text(json.dumps(used_groups, indent=2) + "\n", encoding="utf-8")
        source_path.write_text(json.dumps(used_sources, indent=2) + "\n", encoding="utf-8")
        summary = {
            "name": "LS-CAT T4 1D",
            "source_repository": "https://github.com/NTNU-HPC-Lab/LS-CAT",
            "source_repository_path": str(repository),
            "source_archive": str(archive_path),
            "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "problem_columns": ["m", "n"],
            "actions": list(NLP_THREAD_COUNTS),
            "aggregation": "arithmetic mean across duplicate archived measurements",
            "archive_rows": archive_rows,
            "one_dimensional_rows": one_d_rows,
            "rejected_nonpositive_or_nonfinite_rows": rejected_rows,
            "kernels": len(used_kernels),
            "split_groups": len(set(used_groups.values())),
            "complete_contexts": complete_contexts,
            "runtime_rows": runtime_rows,
            "runtime_column_note": (
                "Preserves the archive's `time` values. The LS-CAT harness labels these "
                "microseconds and times a loop of kernel launches."
            ),
        }
        metadata_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    except Exception:
        for path in (runtime_path, group_path, source_path, metadata_path):
            path.unlink(missing_ok=True)
        try:
            output.rmdir()
        except OSError:
            pass
        raise
