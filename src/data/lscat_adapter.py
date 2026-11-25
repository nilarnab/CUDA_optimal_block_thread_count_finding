import ast
import csv
from contextlib import closing
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile
import zipfile

NLP_THREAD_COUNTS = tuple(range(64, 1025, 64))


def _pair(value, *, field):
    parsed = ast.literal_eval(value)
    return parsed


def _source_identity(path, function, source_root):
    relative = Path(path)
    parts = relative.parts
    if parts and parts[0] == "step2":
        relative = Path(*parts[1:])
    source = source_root / relative.parent / f"{function}.cu"
    kernel_id = (relative.parent / function).as_posix()
    split_group = relative.parts[0]
    return kernel_id, split_group, source


def import_lscat_1d(repository, output_directory, *, archive=None):
    repository = Path(repository).resolve()
    output = Path(output_directory).resolve()
    source_root = repository / "data" / "kernels"
    archive_path = Path(archive).resolve() if archive else (
        repository / "results" / "archive" / "T4_18-6-3_sm52.zip"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    groups = {}
    sources = {}
    archive_rows = one_d_rows = rejected_rows = 0
    block_pair = re.compile(r"\s*([+-]?[0-9]+)\s*,\s*([+-]?[0-9]+)\s*,?\s*")

    with tempfile.TemporaryDirectory(prefix="lscat-import-", dir=output.parent) as temporary:
        staged = Path(temporary) / "dataset"
        staged.mkdir()
        with closing(sqlite3.connect(Path(temporary) / "measurements.sqlite3")) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE measurements ("
                "kernel_id TEXT, m INTEGER, n INTEGER, threads INTEGER, runtime REAL)"
            )
            pending = []
            with zipfile.ZipFile(archive_path) as zipped:
                csv_names = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
                with zipped.open(csv_names[0]) as raw:
                    reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
                    for row in reader:
                        archive_rows += 1
                        block_text = (row["blocks"] or "").strip()
                        if block_text.startswith("(") and block_text.endswith(")"):
                            block_text = block_text[1:-1]
                        match = block_pair.fullmatch(block_text)
                        blocks = tuple(map(int, match.groups())) if match else ()
                        if not blocks or min(blocks) <= 0:
                            rejected_rows += 1
                            continue
                        block_a, block_b = blocks
                        if block_a != 1 or block_b not in NLP_THREAD_COUNTS:
                            continue
                        m, n = _pair(row["matrix"], field="matrix")
                        runtime = float(row["time"])
                        if not math.isfinite(runtime) or runtime <= 0:
                            rejected_rows += 1
                            continue
                        kernel_id, split_group, source = _source_identity(
                            row["path"], row["function"], source_root
                        )
                        previous_group = groups.setdefault(kernel_id, split_group)
                        previous_source = sources.setdefault(
                            kernel_id, source.relative_to(repository).as_posix()
                        )
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
            with (staged / "runtimes.csv").open("w", newline="", encoding="utf-8") as handle:
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
        used_groups = {kernel: groups[kernel] for kernel in sorted(used_kernels)}
        used_sources = {kernel: sources[kernel] for kernel in sorted(used_kernels)}
        (staged / "split_groups.json").write_text(json.dumps(used_groups, indent=2) + "\n", encoding="utf-8")
        (staged / "sources.json").write_text(json.dumps(used_sources, indent=2) + "\n", encoding="utf-8")
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
        (staged / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        staged.replace(output)
    return summary
