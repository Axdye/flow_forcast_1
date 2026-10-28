#!/usr/bin/env python
"""Witness for the two classes of bug that make an experiment table untrustworthy.

Runs against the current code, with a throw-away output root (synthetic data, no
GPU, no training), and checks:

A. ``_find_run`` returns the Base checkpoint of the requested seed only.
B. ``_find_run`` separates adapter methods (structured vs an ablation).
C. ``_router_checkpoint_for`` returns the Router of the requested seed only.
D. ``aggregate_experiments`` never turns one seed's val+test rows into
   ``seeds=2, mean=..., std=...``, and rejects duplicate / multi-seed rows.

Usage::

    PYTHONPATH=src python scripts/verify_selection.py [--keep]

Exit code is 0 only if every check passes.  This is a readable companion to
``tests/unit/test_run_selection.py`` and ``tests/unit/test_artifacts.py``; it is
not a substitute for real training or GPU results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from sbona.cli import _find_run, _router_checkpoint_for
from sbona.config import Config
from sbona.data.dataset import prepare_dataset
from sbona.evaluation.artifacts import aggregate_experiments
from sbona.training.trainer import make_hashes

BASE_CFG = {
    "schema_version": 1,
    "protocol_id": "selection_check_v1",
    "method_id": "sbona",
    "seed": 11,
    "data": {
        "dataset": "synthetic",
        "input_steps": 12,
        "output_steps": 12,
        "split_ratios": [0.6, 0.2, 0.2],
        "steps_per_day": 24,
        "synthetic": True,
        "synthetic_num_nodes": 5,
        "synthetic_num_steps": 240,
        "synthetic_missing_fraction": 0.0,
        "day_of_week": False,
    },
    "graph": {"topology": "published_binary_undirected"},
    "backbone": {
        "name": "staeformer",
        "mode": "frozen",
        "input_embedding_dim": 8,
        "tod_embedding_dim": 8,
        "adaptive_embedding_dim": 8,
        "feed_forward_dim": 16,
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.0,
    },
    "adapter": {"num_candidates": 8, "hidden_dim": 8, "residual_dim": 4, "graph_hops": 1},
    "training": {
        "precision": "fp32",
        "initial_batch_size": 4,
        "eval_batch_size": 4,
        "max_epochs": {"base": 1, "single": 1, "candidates": 1, "router": 1, "probe": 1},
        "patience": {"base": 1, "single": 1, "candidates": 1, "router": 1, "probe": 1},
        "num_workers": 0,
        "device": "cpu",
    },
    "evaluation": {"horizons": [3, 6, 12], "bootstrap_replicates": 20, "block_origins": 12},
}

failures: list[str] = []


def check(label: str, ok: bool, detail: object) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        failures.append(label)


def cfg_with(**overrides: object) -> Config:
    raw = json.loads(json.dumps(BASE_CFG))
    raw.update(overrides)
    return Config.from_dict(raw)


def make_run(
    root: Path, prepared, name: str, *, seed: int, method_id: str, stage: str, k: int
) -> Path:
    run = root / "outputs" / name
    run.mkdir(parents=True)
    cfg = cfg_with(seed=seed, method_id=method_id)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "stage": stage,
        "method_id": method_id,
        "adapter_kind": method_id,
        "seed": seed,
        "config_hash": cfg.hash,
        "hashes": make_hashes(prepared, cfg, k),
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run / "checkpoint_best.pt").write_bytes(f"weights-of-{name}".encode())
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep the temporary output root")
    args = parser.parse_args()

    root = Path(tempfile.mkdtemp(prefix="sbona-selection-check-"))
    future = time.time() + 100

    def age(run: Path) -> None:
        # The pre-fix rule preferred the newest manifest; make the wrong run newest.
        os.utime(run / "manifest.json", (future, future))

    try:
        prepared, _ = prepare_dataset(cfg_with(), root / "data", root / "outputs")
        outputs = root / "outputs"

        print("A. Stage A Base: seed 11 vs seed 22 (seed 22 looks newer)")
        a11 = make_run(root, prepared, "A-seed11", seed=11, method_id="sbona_base", stage="A", k=1)
        a22 = make_run(root, prepared, "A-seed22", seed=22, method_id="sbona_base", stage="A", k=1)
        age(a22)
        for seed, want in ((11, a11), (22, a22)):
            got = _find_run(
                outputs,
                stage="A",
                prepared=prepared,
                cfg=cfg_with(seed=seed, method_id="sbona_base"),
                num_candidates=1,
                label="base",
            )
            check(f"seed {seed} Base", got == want / "checkpoint_best.pt", got.parent.name)
        try:
            _find_run(
                outputs,
                stage="A",
                prepared=prepared,
                cfg=cfg_with(seed=33, method_id="sbona_base"),
                num_candidates=1,
                label="base",
            )
            check("missing seed is refused", False, "returned a checkpoint for seed 33")
        except FileNotFoundError:
            check("missing seed is refused", True, "FileNotFoundError for seed 33")

        print("B. Stage C: main method vs ablation (same seed, K=8)")
        structured = make_run(
            root, prepared, "C-structured", seed=11, method_id="sbona_structured", stage="C", k=8
        )
        ablation = make_run(
            root,
            prepared,
            "C-independent",
            seed=11,
            method_id="sbona_structured_independent_heads",
            stage="C",
            k=8,
        )
        age(ablation)
        got = _find_run(
            outputs,
            stage="C",
            prepared=prepared,
            cfg=cfg_with(method_id="sbona_structured"),
            num_candidates=8,
            label="candidate",
        )
        check("structured candidate", got == structured / "checkpoint_best.pt", got.parent.name)

        print("C. Stage D Router: two runs from different seeds")
        d11 = make_run(
            root, prepared, "D-seed11", seed=11, method_id="sbona_structured", stage="D", k=8
        )
        d22 = make_run(
            root, prepared, "D-seed22", seed=22, method_id="sbona_structured", stage="D", k=8
        )
        age(d22)

        class Args:
            router_checkpoint = None

        for seed, want in ((11, d11), (22, d22)):
            got = _router_checkpoint_for(
                Args(), outputs, prepared, cfg_with(seed=seed, method_id="sbona_structured")
            )
            check(f"seed {seed} Router", got == want / "checkpoint_best.pt", got.parent.name)
        got = _router_checkpoint_for(
            Args(), outputs, prepared, cfg_with(seed=33, method_id="sbona_structured")
        )
        check("missing seed Router", got is None, "None")

        print("D. aggregate: one seed measured on val and test")
        artifact = root / "metrics.json"
        artifact.write_text("{}", encoding="utf-8")
        header = [
            "run_id",
            "method",
            "dataset",
            "backbone",
            "split",
            "seed",
            "protocol_id",
            "data_hash",
            "metric",
            "value",
            "units",
            "deployable",
            "artifact",
        ]
        rows = [
            ["run-eval-val", "sbona_base", "PEMS08", "staeformer", "val", "11", "v1", "abc",
             "raw_mae_all", "0.80", "u", "true", str(artifact)],
            ["run-eval-test", "sbona_base", "PEMS08", "staeformer", "test", "11", "v1", "abc",
             "raw_mae_all", "1.20", "u", "true", str(artifact)],
        ]
        manifest = root / "experiments.csv"

        def write_rows() -> None:
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader()
                writer.writerows([dict(zip(header, row, strict=True)) for row in rows])

        write_rows()
        document = aggregate_experiments(manifest, root / "aggregate.json", root / "aggregate.csv")
        groups = document["groups"]
        check("test table has one row", len(groups) == 1, f"{len(groups)} row(s)")
        check("test table keeps the test value", groups[0]["value"] == 1.20, groups[0]["value"])
        check("no seeds/mean/std fields", "seeds" not in groups[0] and "std" not in groups[0], "absent")
        check("val is not averaged in", groups[0]["split"] == "test", groups[0]["split"])

        rows.append(["run-eval-dupe", "sbona_base", "PEMS08", "staeformer", "test", "11", "v1",
                     "abc", "raw_mae_all", "1.20", "u", "true", str(artifact)])
        write_rows()
        try:
            aggregate_experiments(manifest, root / "aggregate.json")
            check("duplicate test row is rejected", False, "accepted a duplicate")
        except ValueError as exc:
            check("duplicate test row is rejected", "duplicate test row" in str(exc), "ValueError")

        rows[2][5] = "22"
        write_rows()
        try:
            aggregate_experiments(manifest, root / "aggregate.json")
            check("two seeds are rejected", False, "accepted two seeds")
        except ValueError as exc:
            check("two seeds are rejected", "mixes 2 seeds" in str(exc), "ValueError")
    finally:
        if args.keep:
            print(f"\ntemporary root kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)

    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("all selection/aggregation checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
