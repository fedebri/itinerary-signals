"""Stage 2: seven statistical experts and a user-cross-fitted probability gate.

Run from the repository root after stage 1:
    python -m pipeline.base_recommender
    python -m pipeline.base_recommender --public
Only prepared training partitions are read. Stage 3 evaluates both models.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import softmax
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from pipeline.data_features import (
    CITY_COLUMNS,
    FINAL_CONTEXT_COLUMNS,
    final_row_context,
)

TRAIN_FILES = {
    "development": Path("data/processed/development_train.joblib"),
    "public": Path("data/processed/public_train.joblib"),
}
RESULTS_DIR = Path("results/gru_showcase")
FOLD_SEED = 20261012

# Shared with stage 3: the same seven distributions and observable gate inputs.
EXPERTS = (
    "lag",
    "terminal",
    "origin",
    "recent_route",
    "visit_return",
    "origin_return",
    "context",
)
GATE_FEATURES = (
    "log_prefix_stays",
    "log_last_route_events",
    "log_origin_trips",
    "log_pair_trips",
    "mean_log_visited_trials",
    "context_mean_log_trials",
    "context_known_share",
    "lag_top4_low_support_share",
    "lag_top4_mean_start_share",
    "lag_top4_mass",
    "origin_agrees_lag",
    "pair_agrees_lag",
    "context_agrees_lag",
)


# %% 1. Full-catalogue ranking, including deterministic score ties.
def top_indices(scores: np.ndarray, prior: np.ndarray, count: int) -> np.ndarray:
    """Rank positive sparse-evidence scores with the existing prior/ID tie rule.

    Catalog indices are already ascending city-ID strings. Partitioning retains
    all cutoff ties before sorting, avoiding an arbitrary argpartition tie choice.
    """
    positive = np.flatnonzero(scores > 0)
    if len(positive) > count:
        cutoff = np.partition(scores[positive], -count)[-count]
        positive = positive[scores[positive] >= cutoff]
    order = np.lexsort((positive, -prior[positive], -scores[positive]))
    return positive[order[:count]]


# %% 2. Fitting: lag buckets, city/endpoint roles, returns and context correction.
def fit_experts(events, requests, labels):
    """Learn the seven expert distributions from supplied fitting users only."""
    catalog = pd.Index(events["city_id"].unique(), name="city_id").sort_values()
    bookings = events.groupby("city_id").size().reindex(catalog)
    prior = bookings.to_numpy() / bookings.sum()
    city_users = (
        events[["user_id", "city_id"]].drop_duplicates().groupby("city_id").size()
    )
    # Lag expert: later reservations from every observed city, not just next stops.
    previous = events.groupby("utrip_id", sort=False)["city_id"].shift()
    transitions = pd.DataFrame(
        {"source": previous, "city_id": events["city_id"]}
    ).dropna()
    source_support = transitions.groupby("source").size()
    shape = (len(catalog), len(catalog))
    lag_counts = {k: csr_matrix(shape, dtype=np.float32) for k in (1, 2, 3, 4)}
    maximum = int(events.groupby("utrip_id").size().max())
    for lag in range(1, maximum):
        earlier = events.groupby("utrip_id", sort=False)["city_id"].shift(lag)
        known = earlier.notna()
        lag_counts[min(lag, 4)] += csr_matrix(
            (
                np.ones(int(known.sum()), dtype=np.float32),
                (
                    catalog.get_indexer(earlier[known]),
                    catalog.get_indexer(events.loc[known, "city_id"]),
                ),
            ),
            shape=shape,
        )

    fitting = requests[["utrip_id", "city_history"]].merge(
        labels[["utrip_id", "target_city_id"]], on="utrip_id", validate="one_to_one"
    )
    starts = catalog.get_indexer(fitting["city_history"].str[0])
    targets = catalog.get_indexer(fitting["target_city_id"])
    # Terminal, origin and recent-route experts all learn true trip endpoints.
    start_counts = np.bincount(starts, minlength=len(catalog))
    end_counts = np.bincount(targets, minlength=len(catalog))
    terminal = (end_counts + 10 * prior) / (len(fitting) + 10)
    origin_counts = csr_matrix((np.ones(len(fitting)), (starts, targets)), shape=shape)
    pairs = fitting.loc[fitting["city_history"].str.len().ge(2)]
    pair_values = [tuple(history[-2:]) for history in pairs["city_history"]]
    pair_keys = pd.MultiIndex.from_tuples(
        sorted(set(pair_values)), names=["penultimate", "last"]
    )
    pair_counts = csr_matrix(
        (
            np.ones(len(pairs)),
            (
                pair_keys.get_indexer(pair_values),
                catalog.get_indexer(pairs["target_city_id"]),
            ),
        ),
        shape=(len(pair_keys), len(catalog)),
    )
    visits = fitting.assign(
        visited_city=fitting["city_history"].map(
            lambda history: list(dict.fromkeys(history))
        )
    ).explode("visited_city")
    visit_indices = catalog.get_indexer(visits["visited_city"])
    prefix_counts = np.bincount(visit_indices, minlength=len(catalog))
    returned = visits["visited_city"].eq(visits["target_city_id"])
    visit_returns = np.bincount(visit_indices[returned], minlength=len(catalog))
    origin_returns = np.bincount(starts[starts == targets], minlength=len(catalog))
    # Return trials count each prefix city once per trip. Beta(1, 1) global
    # rates and strength-10 shrinkage keep unobserved returns possible.
    visit_base = (visit_returns.sum() + 1) / (prefix_counts.sum() + 2)
    origin_base = (origin_returns.sum() + 1) / (len(fitting) + 2)
    roles = pd.DataFrame(
        {
            "start_trips": start_counts,
            "end_trips": end_counts,
            "prefix_trips": prefix_counts,
            "start_share": start_counts / len(fitting),
            "terminal_probability": terminal,
            "visit_return_probability": (visit_returns + 10 * visit_base)
            / (prefix_counts + 10),
            "origin_return_probability": (origin_returns + 10 * origin_base)
            / (start_counts + 10),
        },
        index=catalog,
    )

    final = (
        events.loc[events["utrip_id"].isin(fitting["utrip_id"])]
        .groupby("utrip_id", sort=False)
        .tail(1)
    )
    context = final_row_context(final)
    city_indices = catalog.get_indexer(final["city_id"])
    categories, lifts = {}, {}
    # Context corrections use allowed endpoint fields and terminal-city priors.
    for column in FINAL_CONTEXT_COLUMNS:
        values = context[column]
        known = values.notna().to_numpy()
        category = pd.Index(values.dropna().unique()).sort_values()
        matrix = csr_matrix(
            (
                np.ones(int(known.sum()), dtype=np.float64),
                (category.get_indexer(values.loc[known]), city_indices[known]),
            ),
            shape=(len(category), len(catalog)),
        )
        matrix.sum_duplicates()
        matrix.data = np.log1p(matrix.data / (10 * terminal[matrix.indices]))
        categories[column], lifts[column] = category, matrix
    return {
        "catalog": catalog,
        "prior": prior,
        "city_users": city_users,
        "source_support": source_support,
        "lag_counts": lag_counts,
        "lag_totals": {
            k: np.asarray(matrix.sum(axis=1)).ravel()
            for k, matrix in lag_counts.items()
        },
        "roles": roles,
        "origin_counts": origin_counts,
        "origin_totals": np.asarray(origin_counts.sum(axis=1)).ravel(),
        "pair_keys": pair_keys,
        "pair_counts": pair_counts,
        "pair_totals": np.asarray(pair_counts.sum(axis=1)).ravel(),
        "context_categories": categories,
        "context_lifts": lifts,
        "context_support": {
            column: requests[column].value_counts().to_dict()
            for column in FINAL_CONTEXT_COLUMNS
        },
        "city_user_counts": city_users.reindex(catalog).to_numpy(),
    }


# %% 3. Prediction: seven normalized experts and 13 support/agreement features.
def score_experts(state, request):
    """Return full-catalogue expert scores and 13 observable gate features."""
    catalog, prior, roles = state["catalog"], state["prior"], state["roles"]
    indices = catalog.get_indexer(request.city_history)
    visited = np.unique(indices[indices >= 0])
    first = indices[0] if len(indices) else -1
    terminal = roles["terminal_probability"].to_numpy()
    if len(indices):
        weights = 0.5 ** np.arange(len(indices) - 1, -1, -1)
        lag = np.zeros(len(catalog), dtype=np.float32)
        for position, index in enumerate(indices):
            weight = weights[position] / weights.sum()
            if index < 0:
                lag += weight * prior
                continue
            bucket = min(len(indices) - position, 4)
            counts = state["lag_counts"][bucket].getrow(index).toarray().ravel()
            lag += (
                weight
                * (counts + 10 * prior)
                / (state["lag_totals"][bucket][index] + 10)
            )
        lag = lag.astype(np.float64)
    else:
        lag = prior.copy()
    lag /= lag.sum()
    origin_trials = state["origin_totals"][first] if first >= 0 else 0.0
    origin = (
        (state["origin_counts"].getrow(first).toarray().ravel() + 10 * terminal)
        / (origin_trials + 10)
        if first >= 0
        else terminal.copy()
    )
    pair_index = (
        state["pair_keys"].get_indexer([tuple(request.city_history[-2:])])[0]
        if len(indices) >= 2
        else -1
    )
    pair_trials = state["pair_totals"][pair_index] if pair_index >= 0 else 0.0
    recent_route = (
        (state["pair_counts"].getrow(pair_index).toarray().ravel() + 10 * terminal)
        / (pair_trials + 10)
        if pair_index >= 0
        else terminal.copy()
    )
    visit_return = terminal.copy()
    visit_return[visited] = roles["visit_return_probability"].to_numpy()[visited]
    visit_return /= visit_return.sum()
    origin_return = terminal.copy()
    if first >= 0:
        origin_return[first] = roles["origin_return_probability"].iloc[first]
    origin_return /= origin_return.sum()
    adjustment = np.zeros(len(catalog), dtype=np.float64)
    for column in FINAL_CONTEXT_COLUMNS:
        value = getattr(request, column)
        if pd.isna(value):
            continue
        category = state["context_categories"][column].get_indexer([value])[0]
        if category >= 0:
            row = state["context_lifts"][column].getrow(category)
            adjustment[row.indices] += row.data / len(FINAL_CONTEXT_COLUMNS)
    context = softmax(np.log(lag) + 0.5 * adjustment)
    probabilities = np.vstack(
        [lag, terminal, origin, recent_route, visit_return, origin_return, context]
    )
    picks = top_indices(lag, prior, 4)
    lag_first = picks[0]
    context_trials = np.array(
        [
            state["context_support"][column].get(getattr(request, column), 0)
            for column in FINAL_CONTEXT_COLUMNS
        ],
        dtype=float,
    )
    last_route_trials = (
        state["source_support"].get(request.city_history[-1], 0) if len(indices) else 0
    )
    features = np.array(
        [
            np.log1p(len(indices)),
            np.log1p(last_route_trials),
            np.log1p(origin_trials),
            np.log1p(pair_trials),
            (
                np.log1p(roles["prefix_trips"].to_numpy()[visited]).mean()
                if len(visited)
                else 0
            ),
            np.log1p(context_trials).mean(),
            (context_trials > 0).mean(),
            (
                (state["city_user_counts"][picks] >= 1)
                & (state["city_user_counts"][picks] <= 20)
            ).mean(),
            roles["start_share"].to_numpy()[picks].mean(),
            lag[picks].sum(),
            np.argmax(origin) == lag_first,
            np.argmax(recent_route) == lag_first,
            np.argmax(context) == lag_first,
        ],
        dtype=float,
    )
    return probabilities, features


# %% 4. Support-aware mixture loss, shared with the GRU extension.
def _mixture_loss(parameters, design, target_probabilities, lag_floor, l2):
    """Exact mixture log loss and gradient; unsupported targets stay misses."""
    coefficients = parameters.reshape(design.shape[1], target_probabilities.shape[1])
    weights = softmax(design @ coefficients, axis=1)
    residual = (weights * target_probabilities).sum(axis=1)
    mixture = lag_floor * target_probabilities[:, 0] + (1 - lag_floor) * residual
    safe = np.maximum(mixture, 1e-12)
    slopes = coefficients[1:]
    loss = -np.log(safe).mean() + 0.5 * l2 * np.square(slopes).sum()
    derivative = (
        -(1 - lag_floor)
        * weights
        * (target_probabilities - residual[:, None])
        / safe[:, None]
        / len(design)
    )
    derivative[mixture <= 1e-12] = 0
    gradient = design.T @ derivative
    gradient[1:] += l2 * slopes
    return loss, gradient.ravel()


def rank_recommendations(experts, gate, requests, neural_batches=None):
    """Rank the base mixture or its GRU extension over the same full catalogue.

    These two callers share score ordering, gate scaling and the lag floor.
    Neural batches, when supplied, follow the request and catalogue order.
    """
    stream = iter(neural_batches) if neural_batches is not None else None
    rows, weight_batches = [], []
    for start in range(0, len(requests), 32):
        batch = requests.iloc[start : start + 32]
        scores, features = zip(
            *(score_experts(experts, row) for row in batch.itertuples(index=False))
        )
        features = pd.DataFrame(features, columns=GATE_FEATURES)
        design = np.column_stack(
            [np.ones(len(features)), gate["scaler"].transform(features)]
        )
        weights = (1 - gate["lag_floor"]) * softmax(
            design @ gate["coefficients"], axis=1
        )
        weights[:, 0] += gate["lag_floor"]
        weight_batches.append(weights)
        neural_scores = next(stream) if stream is not None else None
        for offset, (trip, statistical_scores) in enumerate(
            zip(batch["utrip_id"], scores)
        ):
            distributions = (
                np.vstack([statistical_scores, neural_scores[offset]])
                if neural_scores is not None
                else statistical_scores
            )
            combined = weights[offset] @ distributions
            picks = experts["catalog"][
                top_indices(combined, experts["prior"], 4)
            ].tolist()
            rows.append([trip, *picks, *([pd.NA] * (4 - len(picks)))])
        if start // 5000 != (start + len(batch)) // 5000:
            print(
                f"  scored {start + len(batch):,}/{len(requests):,} trips", flush=True
            )
    return pd.DataFrame(rows, columns=["utrip_id", *CITY_COLUMNS]), np.vstack(
        weight_batches
    )


# %% 5. Fit on local training data: three whole-user folds, then all fitting users.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public", action="store_true", help="Also fit the full-training public model."
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    runtime = {
        name: version(name)
        for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
    }
    source_sha256 = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path("pipeline/data_features.py"),
            Path("pipeline/base_recommender.py"),
        )
    }

    for phase in ["development", "public"] if args.public else ["development"]:
        training = joblib.load(TRAIN_FILES[phase])
        events, requests, labels = (
            training["events"],
            training["requests"],
            training["labels"],
        )
        directory = args.output_dir / phase
        directory.mkdir(parents=True, exist_ok=True)
        signature = joblib.hash(
            (
                [
                    (
                        tuple(frame.columns),
                        tuple(map(str, frame.dtypes)),
                        pd.util.hash_pandas_object(frame, index=True).to_numpy(),
                    )
                    for frame in (events, requests, labels)
                ],
                source_sha256,
                runtime,
                phase,
            )
        )
        print(
            f"\n{phase}: {len(events):,} stays, {len(requests):,} endpoints; seven experts",
            flush=True,
        )
        users = pd.Index(events["user_id"].unique()).sort_values()
        eligible_users = np.sort(requests["user_id"].unique())
        sampled_users = (
            np.random.default_rng(FOLD_SEED).choice(
                eligible_users, 20000, replace=False
            )
            if len(eligible_users) > 20000
            else eligible_users
        )
        target_lookup = labels.set_index("utrip_id")["target_city_id"]
        parts, assignments = [], []
        for fold, (fit_indices, held_indices) in enumerate(
            KFold(n_splits=3, shuffle=True, random_state=FOLD_SEED).split(users)
        ):
            fitting_users, held_users = users[fit_indices], users[held_indices]
            held = requests.loc[
                requests["user_id"].isin(held_users)
                & requests["user_id"].isin(sampled_users)
            ].reset_index(drop=True)
            assignments.append(
                pd.DataFrame(
                    {
                        "user_id": held_users,
                        "fold": fold,
                        "scored": held_users.isin(sampled_users),
                    }
                )
            )
            cache = directory / f"base_fold_{fold}.joblib"
            if cache.exists():
                part = joblib.load(cache)
                if part["signature"] != signature:
                    raise ValueError(
                        f"Stale base cache: {cache}; use a new --output-dir."
                    )
                print(f"  fold {fold}: reload completed expert predictions", flush=True)
            else:
                fold_events = events.loc[events["user_id"].isin(fitting_users)]
                fold_requests = requests.loc[requests["user_id"].isin(fitting_users)]
                fold_labels = labels.loc[labels["user_id"].isin(fitting_users)]
                experts = fit_experts(fold_events, fold_requests, fold_labels)
                features, targets = [], []
                unsupported = 0
                for row in held.itertuples(index=False):
                    probabilities, evidence = score_experts(experts, row)
                    target = experts["catalog"].get_indexer(
                        [target_lookup.loc[row.utrip_id]]
                    )[0]
                    features.append(evidence)
                    targets.append(
                        probabilities[:, target]
                        if target >= 0
                        else np.zeros(len(EXPERTS))
                    )
                    unsupported += int(target < 0)
                part = {
                    "signature": signature,
                    "requests": held[["utrip_id", "user_id"]],
                    "features": pd.DataFrame(features, columns=GATE_FEATURES),
                    "probabilities": pd.DataFrame(targets, columns=EXPERTS),
                    "audit": {
                        "fold": fold,
                        "fitting_users": len(fitting_users),
                        "held_users": len(held_users),
                        "scored_requests": len(held),
                        "unsupported_targets": unsupported,
                        "overlapping_users": len(
                            set(fitting_users).intersection(held_users)
                        ),
                    },
                }
                temporary = cache.with_suffix(".tmp")
                joblib.dump(part, temporary, compress=3)
                temporary.replace(cache)
                print(f"  fold {fold}: {len(held):,} held-user requests", flush=True)
            parts.append(part)
        pd.concat(assignments, ignore_index=True).to_csv(
            directory / "user_folds.csv", index=False
        )
        pd.DataFrame([part["audit"] for part in parts]).to_csv(
            directory / "oof_audit.csv", index=False
        )

        # %% 6. Support-aware gate: train-only scaling, constrained mixture NLL.
        features = pd.concat([part["features"] for part in parts], ignore_index=True)
        probabilities = pd.concat(
            [part["probabilities"] for part in parts], ignore_index=True
        )
        scaler = StandardScaler().fit(features[list(GATE_FEATURES)])
        design = np.column_stack(
            [np.ones(len(features)), scaler.transform(features[list(GATE_FEATURES)])]
        )
        target_probabilities = probabilities[list(EXPERTS)].to_numpy()
        result = minimize(
            _mixture_loss,
            np.zeros(design.shape[1] * len(EXPERTS)),
            args=(design, target_probabilities, 0.5, 0.01),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 200, "ftol": 1e-9},
        )
        if not result.success:
            raise RuntimeError(f"Blend optimization failed: {result.message}")
        gate = {
            "scaler": scaler,
            "coefficients": result.x.reshape(design.shape[1], len(EXPERTS)),
            "lag_floor": 0.5,
            "objective": float(result.fun),
        }
        expert_losses = dict(
            zip(
                EXPERTS,
                (
                    -np.log(np.maximum(target_probabilities, 1e-12)).mean(axis=0)
                ).tolist(),
            )
        )
        summary = {
            "phase": phase,
            "fitting_users": len(users),
            "fitting_requests": len(requests),
            "oof_requests": len(features),
            "oof_destination_log_loss_by_expert": expert_losses,
            "gate_training_objective": float(result.fun),
            "gate_features": list(GATE_FEATURES),
            "gate_l2": 0.01,
            "lag_floor": 0.5,
            "fold_seed": FOLD_SEED,
            "source_sha256": source_sha256,
            "runtime": runtime,
        }
        (directory / "base_training.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(f"  OOF expert log losses: {expert_losses}", flush=True)
        print(f"  gate training objective: {result.fun:.6f}", flush=True)

        # %% 7. Refit the same seven experts on all fitting users; no test access.
        output = directory / "base_model.joblib"
        if output.exists():
            fitted = joblib.load(output)
            if fitted["signature"] != signature:
                raise ValueError(f"Stale base model: {output}; use a new --output-dir.")
            experts = fitted["experts"]
        else:
            experts = fit_experts(events, requests, labels)
        joblib.dump(
            {"signature": signature, "experts": experts, "gate": gate},
            output,
            compress=3,
        )
        print(f"  {len(experts['catalog']):,} catalogue cities -> {output}", flush=True)
