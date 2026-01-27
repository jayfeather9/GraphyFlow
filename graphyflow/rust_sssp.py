from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

Indexing = Literal["auto", "zero-based", "one-based"]
Weight = Literal["unit", "third-column-u32"]
OutFormat = Literal["full-u32-le", "updates-u32x2-le"]


@dataclass(frozen=True)
class SsspPlan:
    dataset: str
    source: int = 0
    indexing: Indexing = "auto"
    weight: Weight = "unit"
    num_nodes: Optional[int] = None
    detect_sample_bytes: Optional[int] = None
    out: Optional[str] = None
    out_format: Optional[OutFormat] = None

    def to_json_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "dataset": self.dataset,
            "source": self.source,
            "indexing": self.indexing,
            "weight": self.weight,
        }
        if self.num_nodes is not None:
            d["num_nodes"] = self.num_nodes
        if self.detect_sample_bytes is not None:
            d["detect_sample_bytes"] = self.detect_sample_bytes
        if self.out is not None:
            d["out"] = self.out
        if self.out_format is not None:
            d["out_format"] = self.out_format
        return d


def default_runner_path(repo_root: Path) -> Path:
    return repo_root / "rust_sim" / "graphyflow_sssp" / "target" / "release" / "graphyflow_sssp"


def ensure_runner_built(repo_root: Path) -> Path:
    runner = default_runner_path(repo_root)
    if runner.exists():
        return runner

    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=str(repo_root / "rust_sim" / "graphyflow_sssp"),
        check=True,
    )
    if not runner.exists():
        raise RuntimeError(f"Rust runner build finished, but binary not found at {runner}")
    return runner


def write_plan(plan: SsspPlan, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def run_plan(
    plan_path: Path,
    *,
    repo_root: Optional[Path] = None,
    runner_path: Optional[Path] = None,
    build_if_missing: bool = True,
) -> Dict[str, Any]:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    if runner_path is None:
        runner_path = default_runner_path(repo_root)
        if build_if_missing and not runner_path.exists():
            runner_path = ensure_runner_built(repo_root)

    proc = subprocess.run(
        [str(runner_path), "--plan", str(plan_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError("Rust runner produced no output on stdout")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse runner JSON output: {e}\nstdout:\n{stdout}\nstderr:\n{proc.stderr}")


def run_sssp(
    dataset: Path,
    *,
    source: int = 0,
    indexing: Indexing = "auto",
    weight: Weight = "unit",
    num_nodes: Optional[int] = None,
    out: Optional[Path] = None,
    out_format: Optional[OutFormat] = None,
    repo_root: Optional[Path] = None,
    runner_path: Optional[Path] = None,
    build_if_missing: bool = True,
) -> Dict[str, Any]:
    plan = SsspPlan(
        dataset=str(dataset),
        source=source,
        indexing=indexing,
        weight=weight,
        num_nodes=num_nodes,
        out=str(out) if out is not None else None,
        out_format=out_format,
    )
    with tempfile.TemporaryDirectory(prefix="graphyflow_sssp_") as td:
        plan_path = Path(td) / "plan.json"
        write_plan(plan, plan_path)
        return run_plan(
            plan_path,
            repo_root=repo_root,
            runner_path=runner_path,
            build_if_missing=build_if_missing,
        )

