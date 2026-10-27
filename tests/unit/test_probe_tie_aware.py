from __future__ import annotations

import torch

from sbona.training.probe import _evaluate_probe


def test_top1_tie_aware_counts_near_ties() -> None:
    data = {
        "abs_sum": torch.tensor([[1.0, 1.0], [0.0, 2.0]]),
        "count": torch.tensor([1.0, 1.0]),
        "cost": torch.zeros(2, 2),
        "soft_target": torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
    }
    # Row 0 selects candidate 1, which is not the argmin but is an exact
    # near-tie with candidate 0; row 1 selects the true best candidate.
    index = torch.tensor([1, 0])
    result = _evaluate_probe(data, index, "test")
    # Fixed argmin picks index 0 in row 0, so strict top-1 is 50%.
    assert result["top1_accuracy"] == 0.5
    # Tie-aware scoring credits the near-tie in row 0 as well.
    assert result["top1_accuracy_tie_aware"] == 1.0
    assert result["top1_accuracy_tie_aware"] > result["top1_accuracy"]
    assert result["tie_fraction"] == 0.5
    # Selected MAE: (row0 candidate 1 = 1, row1 candidate 0 = 0) / 2 = 0.5.
    assert result["selected_mae"] == 0.5
    assert result["oracle_mae"] == 0.5
    assert result["regret"] == 0.0
