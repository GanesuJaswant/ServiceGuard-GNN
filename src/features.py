import pandas as pd
import numpy as np


def build_features(
    trace: pd.DataFrame,
    service_names,
    baseline
):
    """
    Build the four node features used by ServiceGuard-GNN.

    Features per service:
        1. Raw latency
        2. Ratio to normal baseline
        3. Rank
        4. Contribution to total trace latency

    Returns:
        DataFrame with 4 features per service.
    """

    # -----------------------------------------
    # Validate service columns
    # -----------------------------------------

    missing = [
        service
        for service in service_names
        if service not in trace.columns
    ]

    if missing:
        raise ValueError(
            f"Missing service columns: {missing}"
        )

    # -----------------------------------------
    # Raw latency
    # -----------------------------------------

    raw = trace[
        service_names
    ].astype(float)

    # -----------------------------------------
    # Baseline
    # -----------------------------------------

    baseline = pd.Series(
        baseline,
        index=service_names,
        dtype=float
    )

    baseline = baseline.replace(
        0,
        1e-8
    )

    # -----------------------------------------
    # Ratio to normal
    # -----------------------------------------

    ratio = raw / baseline

    # -----------------------------------------
    # Rank
    # -----------------------------------------

    rank = raw.rank(
        axis=1,
        method="average",
        ascending=False
    )

    # -----------------------------------------
    # Contribution
    # -----------------------------------------

    row_sum = raw.sum(
        axis=1
    ).replace(
        0,
        1e-8
    )

    contribution = raw.div(
        row_sum,
        axis=0
    )

    # -----------------------------------------
    # Combine
    # -----------------------------------------

    features = pd.concat(
        [
            raw,
            ratio.add_suffix("_ratio"),
            rank.add_suffix("_rank"),
            contribution.add_suffix(
                "_contribution"
            )
        ],
        axis=1
    )

    return features