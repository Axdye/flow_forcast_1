"""Evaluation: metrics, oracles, diversity, regimes, statistics, profiling."""

from sbona.evaluation.artifacts import (
    RunArtifacts,
    aggregate_experiments,
    collect_environment,
    create_run_id,
    read_json,
    validate_metrics_document,
    write_json,
)
from sbona.evaluation.diversity import (
    action_distances,
    branch_errors,
    candidate_health,
    entropy_metrics,
    hard_usage,
    trajectory_distances,
)
from sbona.evaluation.metrics import (
    MetricAccumulator,
    metric_result,
    per_sample_abs_error,
    per_sample_sq_error,
)
from sbona.evaluation.oracle import (
    OracleSelection,
    candidate_errors,
    gather_candidates,
    oracle_cost,
    oracle_mae,
    oracle_metrics,
    oracle_mse,
    retention,
    select_whole_sample,
)
from sbona.evaluation.profiler import measure_latency, profile_callable, profile_model
from sbona.evaluation.regimes import (
    assign_flow_regime,
    flow_regime_thresholds,
    regime_metrics,
)
from sbona.evaluation.statistics import (
    block_sign_permutation,
    holm_correction,
    paired_moving_block_bootstrap,
    per_origin_records,
)

__all__ = [
    "RunArtifacts",
    "aggregate_experiments",
    "collect_environment",
    "create_run_id",
    "read_json",
    "validate_metrics_document",
    "write_json",
    "action_distances",
    "branch_errors",
    "candidate_health",
    "entropy_metrics",
    "hard_usage",
    "trajectory_distances",
    "MetricAccumulator",
    "metric_result",
    "per_sample_abs_error",
    "per_sample_sq_error",
    "OracleSelection",
    "candidate_errors",
    "gather_candidates",
    "oracle_cost",
    "oracle_mae",
    "oracle_metrics",
    "oracle_mse",
    "retention",
    "select_whole_sample",
    "measure_latency",
    "profile_callable",
    "profile_model",
    "assign_flow_regime",
    "flow_regime_thresholds",
    "regime_metrics",
    "block_sign_permutation",
    "holm_correction",
    "paired_moving_block_bootstrap",
    "per_origin_records",
]
