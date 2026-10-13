"""Stage 3: full-prefix GRU, eight-expert gate and held-out evaluation.

Run after stages 1 and 2: python -m pipeline.gru_extension
Add --public to evaluate the masked challenge benchmark; stage 2 must also have
been run with --public. Only this stage reads held-out requests and public truth.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from pipeline.data_features import (
    CITY_COLUMNS,
    FINAL_CONTEXT_COLUMNS,
    NUMERIC_FEATURES,
    collate_stays,
    encode_examples,
    final_row_context,
    prepare_challenge_requests,
)
from pipeline.base_recommender import (
    EXPERTS,
    GATE_FEATURES,
    TRAIN_FILES,
    _mixture_loss,
    rank_recommendations,
)

DEVELOPMENT_HOLDOUT = Path("data/processed/development_holdout.joblib")
TEST_CSV = Path("data/multi-destination-trip-dataset/test_set.csv")
TRUTH_CSV = Path("data/multi-destination-trip-dataset/ground_truth.csv")
RESULTS_DIR = Path("results/gru_showcase")
TRAINING_SEED = 20261013
BOOTSTRAP_SEED = 42


# %% 1. GRU architecture: embeddings, full prefix, explicit first/last stays,
# six context embeddings, five numeric summaries, shared input/output city head.
class ItineraryGRU(nn.Module):
    """Ordered full-prefix GRU with shared input/output city embeddings."""

    def __init__(self, vocabulary):
        super().__init__()
        width = 32
        self.city = nn.Embedding(len(vocabulary["cities"]) + 2, width, padding_idx=0)
        # City 0 is padding and 1 is the unknown token. The latter starts at
        # zero; it can appear when inference sees a city outside the catalogue.
        with torch.no_grad():
            self.city.weight[1].zero_()
        self.country = nn.Embedding(len(vocabulary["countries"]) + 1, 8, padding_idx=0)
        self.context = nn.ModuleList(
            [
                nn.Embedding(
                    len(vocabulary["contexts"][name]) + 1,
                    8,
                    padding_idx=0,
                )
                for name in FINAL_CONTEXT_COLUMNS
            ]
        )
        self.token = nn.Sequential(nn.Linear(width + 8 + 1, width), nn.Tanh())
        self.gru = nn.GRU(width, width, batch_first=True)
        self.trip = nn.Sequential(
            nn.Linear(
                4 * width + 6 * 8 + len(NUMERIC_FEATURES),
                64,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, width),
        )
        self.city_bias = nn.Parameter(torch.zeros(len(vocabulary["cities"])))

    def forward(self, batch):
        """Score every fitted city from the ordered visible prefix and context."""
        city = batch["cities"]
        mask = city.ne(0)
        tokens = (
            self.token(
                torch.cat(
                    [
                        self.city(city),
                        self.country(batch["countries"]),
                        torch.log1p(batch["nights"])[..., None],
                    ],
                    dim=-1,
                )
            )
            * mask[..., None]
        )
        lengths = batch["lengths"]
        device_lengths = lengths.to(city.device)
        # Packing prevents padded positions from changing the GRU summary.
        # PyTorch expects these lengths on CPU even when tokens are elsewhere.
        packed = pack_padded_sequence(
            tokens, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        summary = hidden[-1]
        rows = torch.arange(len(city), device=city.device)
        last = tokens[rows, device_lengths - 1]
        penultimate = tokens[rows, (device_lengths - 2).clamp(min=0)]
        # A one-stay or empty prefix has no genuine penultimate stay.
        penultimate = penultimate * device_lengths.ge(2)[:, None]
        context = torch.cat(
            [layer(batch["context"][:, i]) for i, layer in enumerate(self.context)],
            dim=1,
        )
        trip = self.trip(
            torch.cat(
                [summary, tokens[:, 0], last, penultimate, context, batch["numeric"]],
                dim=1,
            )
        )
        # Reuse fitted city embeddings as output vectors; padding and unknown
        # tokens are excluded because neither is a recommendable city.
        logits = trip @ self.city.weight[2:].T + self.city_bias
        return logits


# %% 2. Full-catalogue inference; labels are never an input.
def neural_probabilities(neural, requests):
    """Yield full-catalogue probabilities in batches, without receiving labels."""
    examples = encode_examples(requests, neural["vocabulary"])
    loader = DataLoader(examples, batch_size=32, collate_fn=collate_stays)
    neural["network"].eval()
    for batch in loader:
        with torch.inference_mode():
            probabilities = neural["network"](batch).softmax(dim=1).numpy()
        yield probabilities


# %% 3. Load training-only artifacts and train three fold GRUs plus one refit.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public",
        action="store_true",
        help="Also evaluate the masked public benchmark.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="12 for the recorded model; 1 for a smoke run.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    runtime = {
        name: version(name)
        for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "torch")
    }
    source_sha256 = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path("pipeline/data_features.py"),
            Path("pipeline/base_recommender.py"),
            Path("pipeline/gru_extension.py"),
        )
    }
    # The signatures bind checkpoints to the actual source and prepared data;
    # even a comment edit requires a fresh output directory before rerunning.
    # This dictionary records the run; the estimator/optimizer specifications
    # appear literally below where their library APIs are called.
    training_parameters = {
        "epochs": args.epochs,
        "seed": TRAINING_SEED,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip": 5.0,
        "embedding_dim": 32,
        "country_dim": 8,
        "context_dim": 8,
        "hidden_dim": 64,
        "dropout": 0.1,
    }

    for phase in ["development", "public"] if args.public else ["development"]:
        directory = args.output_dir / phase
        training = joblib.load(TRAIN_FILES[phase])
        events, requests, labels = (
            training["events"],
            training["requests"],
            training["labels"],
        )
        base = joblib.load(directory / "base_model.joblib")
        expected_signature = joblib.hash(
            (
                [
                    (
                        tuple(frame.columns),
                        tuple(map(str, frame.dtypes)),
                        pd.util.hash_pandas_object(frame, index=True).to_numpy(),
                    )
                    for frame in (events, requests, labels)
                ],
                {
                    name: value
                    for name, value in source_sha256.items()
                    if name != "pipeline/gru_extension.py"
                },
                {name: value for name, value in runtime.items() if name != "torch"},
                phase,
            )
        )
        if base["signature"] != expected_signature:
            raise ValueError(
                f"Base model does not match current training data/code: {directory}; rerun stage 2 in a new --output-dir."
            )
        signature = joblib.hash(
            (
                base["signature"],
                source_sha256,
                args.epochs,
                runtime,
                sys.version,
                torch.get_num_threads(),
            )
        )
        folds = pd.read_csv(directory / "user_folds.csv", dtype={"user_id": "string"})
        target_lookup = labels.set_index("utrip_id")["target_city_id"]
        parts = [
            joblib.load(directory / f"base_fold_{fold}.joblib") for fold in range(3)
        ]
        neural_targets, neural_audit = [], []
        print(
            f"\n{phase}: GRU, {args.epochs} epochs, AdamW lr=0.001, batch=128, seed={TRAINING_SEED}",
            flush=True,
        )

        for fold in [0, 1, 2, "refit"]:
            # Fold models supply held-user probabilities for gate training.
            # The refit model uses all fitting users for final inference.
            if fold == "refit":
                fit_events, fit_requests, fit_labels = events, requests, labels
                checkpoint = directory / "gru.pt"
            else:
                held_users = folds.loc[folds["fold"].eq(fold), "user_id"]
                fit_events = events.loc[~events["user_id"].isin(held_users)]
                fit_requests = requests.loc[~requests["user_id"].isin(held_users)]
                fit_labels = labels.loc[~labels["user_id"].isin(held_users)]
                checkpoint = directory / f"gru_fold_{fold}.pt"
            checkpoint_signature = f"{signature}:fold={fold}"
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if saved["signature"] != checkpoint_signature:
                    raise ValueError(
                        f"Stale GRU checkpoint: {checkpoint}; use a new --output-dir."
                    )
                vocabulary = {
                    "cities": pd.Index(saved["cities"], name="city_id", dtype="string"),
                    "countries": pd.Index(saved["countries"], dtype="string"),
                    "contexts": {
                        name: pd.Index(values, dtype="string")
                        for name, values in saved["contexts"].items()
                    },
                }
                network = ItineraryGRU(vocabulary)
                network.load_state_dict(saved["state_dict"])
                print(f"  {fold}: reload completed checkpoint", flush=True)
            else:
                # Fold-local vocabularies include all fitting stays, including
                # endpoints and singletons. Held users supply no fitting evidence.
                torch.manual_seed(TRAINING_SEED)
                context = final_row_context(fit_events)
                vocabulary = {
                    "cities": pd.Index(
                        fit_events["city_id"].unique(), name="city_id"
                    ).sort_values(),
                    "countries": pd.Index(
                        fit_events["hotel_country"].dropna().unique()
                    ).sort_values(),
                    "contexts": {
                        name: pd.Index(context[name].dropna().unique()).sort_values()
                        for name in FINAL_CONTEXT_COLUMNS
                    },
                }
                targets = fit_labels.set_index("utrip_id")["target_city_id"].loc[
                    fit_requests["utrip_id"]
                ]
                examples = encode_examples(fit_requests, vocabulary, targets)
                network = ItineraryGRU(vocabulary)
                torch.manual_seed(TRAINING_SEED + 1)
                optimizer = torch.optim.AdamW(
                    network.parameters(), lr=0.001, weight_decay=0.0001
                )
                # The full-city softmax makes each true endpoint compete with
                # the complete fitted catalogue rather than sampled negatives.
                loss_function = nn.CrossEntropyLoss()
                training_log = []
                for epoch in range(args.epochs):
                    loader = DataLoader(
                        examples,
                        batch_size=128,
                        shuffle=True,
                        generator=torch.Generator().manual_seed(TRAINING_SEED + epoch),
                        collate_fn=collate_stays,
                    )
                    network.train()
                    loss_sum, samples = 0.0, 0
                    for batch in loader:
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_function(network(batch), batch["target"])
                        loss.backward()
                        nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                        optimizer.step()
                        count = len(batch["target"])
                        samples += count
                        loss_sum += float(loss.detach()) * count
                    training_log.append(
                        {
                            "epoch": epoch + 1,
                            "city_loss": loss_sum / samples,
                            "samples": samples,
                        }
                    )
                    print(
                        f"  {fold}: epoch {epoch + 1}/{args.epochs}, loss {loss_sum / samples:.4f}",
                        flush=True,
                    )
                saved = {
                    "signature": checkpoint_signature,
                    "training_parameters": training_parameters,
                    "cities": vocabulary["cities"].tolist(),
                    "countries": vocabulary["countries"].tolist(),
                    "contexts": {
                        name: values.tolist()
                        for name, values in vocabulary["contexts"].items()
                    },
                    "state_dict": network.state_dict(),
                    "training_log": training_log,
                    "fitting_users": sorted(fit_events["user_id"].unique().tolist()),
                }
                temporary = checkpoint.with_suffix(".tmp")
                torch.save(saved, temporary)
                temporary.replace(checkpoint)
            network.eval()
            neural = {"network": network, "vocabulary": vocabulary}
            pd.DataFrame(saved["training_log"]).to_csv(
                directory / f"gru_{fold}_training.csv", index=False
            )

            if fold != "refit":
                # Score exactly the requests used by the seven held-user experts.
                held = parts[fold]["requests"].merge(
                    requests, on=["utrip_id", "user_id"], validate="one_to_one"
                )
                target_indices = vocabulary["cities"].get_indexer(
                    target_lookup.loc[held["utrip_id"]]
                )
                cache = directory / f"gru_fold_{fold}_targets.joblib"
                if cache.exists():
                    cached = joblib.load(cache)
                    if cached["signature"] != checkpoint_signature:
                        raise ValueError(
                            f"Stale neural OOF cache: {cache}; use a new --output-dir."
                        )
                    values = cached["probabilities"]
                else:
                    # A held-user endpoint absent from this fold's catalogue
                    # contributes zero target probability to the gate loss.
                    values = np.zeros(len(held), dtype=np.float64)
                    offset = 0
                    for distribution in neural_probabilities(neural, held):
                        selected = target_indices[offset : offset + len(distribution)]
                        known = selected >= 0
                        batch_values = np.zeros(len(distribution), dtype=np.float64)
                        batch_values[known] = distribution[
                            np.flatnonzero(known), selected[known]
                        ]
                        values[offset : offset + len(distribution)] = batch_values
                        offset += len(distribution)
                    joblib.dump(
                        {"signature": checkpoint_signature, "probabilities": values},
                        cache,
                        compress=3,
                    )
                neural_targets.extend(values)
                neural_audit.append(
                    {
                        "fold": fold,
                        "fitting_users": len(saved["fitting_users"]),
                        "scored_requests": len(held),
                        "unsupported_targets": int((target_indices < 0).sum()),
                        "overlapping_users": len(
                            set(saved["fitting_users"]).intersection(held_users)
                        ),
                    }
                )
        pd.DataFrame(neural_audit).to_csv(directory / "gru_oof_audit.csv", index=False)

        # %% 4. Extend the gate to eight experts using the same held-user rows.
        features = pd.concat([part["features"] for part in parts], ignore_index=True)
        probabilities = pd.concat(
            [part["probabilities"] for part in parts], ignore_index=True
        )
        probabilities["gru"] = neural_targets
        # Refit only the gate here. Its inputs are the same held-user rows and
        # support features used by the seven-expert reference.
        scaler = StandardScaler().fit(features[list(GATE_FEATURES)])
        design = np.column_stack(
            [np.ones(len(features)), scaler.transform(features[list(GATE_FEATURES)])]
        )
        target_probabilities = probabilities[[*EXPERTS, "gru"]].to_numpy()
        result = minimize(
            _mixture_loss,
            np.zeros(design.shape[1] * 8),
            args=(design, target_probabilities, 0.5, 0.01),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 200, "ftol": 1e-9},
        )
        if not result.success:
            raise RuntimeError(f"Blend optimization failed: {result.message}")
        gate = {
            "scaler": scaler,
            "coefficients": result.x.reshape(design.shape[1], 8),
            "lag_floor": 0.5,
            "objective": float(result.fun),
        }
        joblib.dump(gate, directory / "gru_gate.joblib", compress=3)
        print(f"  eight-expert gate training objective: {result.fun:.6f}", flush=True)

        # %% 5. Only now open held-out requests; rank and export before truth.
        if phase == "development":
            holdout = joblib.load(DEVELOPMENT_HOLDOUT)
            evaluation_requests = holdout["requests"].reset_index(drop=True)
        else:
            evaluation_requests, audit = prepare_challenge_requests(
                pd.read_csv(TEST_CSV, dtype="string")
            )
            audit.to_csv(directory / "prefix_audit.csv", index=False)
        endpoint_predictions, endpoint_weights = rank_recommendations(
            base["experts"], base["gate"], evaluation_requests
        )
        predictions, weights = rank_recommendations(
            base["experts"],
            gate,
            evaluation_requests,
            neural_probabilities(neural, evaluation_requests),
        )
        endpoint_predictions.to_csv(directory / "endpoint_predictions.csv", index=False)
        prediction_path = directory / "predictions.csv"
        predictions.to_csv(prediction_path, index=False)
        prediction_sha256 = hashlib.sha256(prediction_path.read_bytes()).hexdigest()

        # %% 6. Single-target HitRate@4 / NDCG@4 on every held-out request.
        # Access labels only after both prediction files have been written.
        if phase == "development":
            truth = holdout["labels"][["utrip_id", "target_city_id"]]
        else:
            truth = pd.read_csv(
                TRUTH_CSV, dtype="string", usecols=["utrip_id", "city_id"]
            ).rename(columns={"city_id": "target_city_id"})
        scored = evaluation_requests[["utrip_id", "user_id"]].merge(
            truth, on="utrip_id", how="left", validate="one_to_one"
        )
        if scored["target_city_id"].isna().any():
            raise ValueError(f"Ground truth must cover every {phase} request.")
        model_metrics = {}
        for name, lists in (
            ("endpoint_reference", endpoint_predictions),
            ("gru_blend", predictions),
        ):
            aligned = scored.merge(
                lists, on="utrip_id", how="left", validate="one_to_one"
            )
            matches = (
                aligned[CITY_COLUMNS]
                .eq(aligned["target_city_id"], axis=0)
                .fillna(False)
            )
            scored[f"{name}_hit"] = matches.any(axis=1).astype(int)
            # With one relevant endpoint, reciprocal log rank is NDCG@4.
            scored[f"{name}_ndcg"] = matches.mul(
                1 / np.log2(np.arange(2, 6)), axis=1
            ).sum(axis=1)
            model_metrics[name] = {
                "hits": int(scored[f"{name}_hit"].sum()),
                "hitrate_at_4_pct": 100 * float(scored[f"{name}_hit"].mean()),
                "ndcg_at_4_pct": 100 * float(scored[f"{name}_ndcg"].mean()),
            }

        # %% 7. Paired user-cluster bootstrap: repeated trips stay together.
        clusters = (
            scored.groupby("user_id", sort=True)
            .agg(
                requests=("utrip_id", "size"),
                endpoint_hit=("endpoint_reference_hit", "sum"),
                gru_hit=("gru_blend_hit", "sum"),
                endpoint_ndcg=("endpoint_reference_ndcg", "sum"),
                gru_ndcg=("gru_blend_ndcg", "sum"),
            )
            .to_numpy(dtype=float)
        )
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        bootstrap = np.empty((2000, 4))
        for iteration in range(2000):
            # Resample users, not requests, to retain within-user dependence;
            # one shared draw makes the model comparison paired.
            totals = clusters[rng.integers(0, len(clusters), size=len(clusters))].sum(
                axis=0
            )
            bootstrap[iteration] = 100 * totals[1:] / totals[0]
        intervals = np.quantile(bootstrap, [0.025, 0.975], axis=0)
        for name, hit_column, ndcg_column in (
            ("endpoint_reference", 0, 2),
            ("gru_blend", 1, 3),
        ):
            model_metrics[name]["hitrate_at_4_95pct_interval_pct"] = intervals[
                :, hit_column
            ].tolist()
            model_metrics[name]["ndcg_at_4_95pct_interval_pct"] = intervals[
                :, ndcg_column
            ].tolist()
        gain_interval = np.quantile(
            bootstrap[:, 1] - bootstrap[:, 0], [0.025, 0.975]
        ).tolist()
        metrics = {
            "phase": phase,
            "requests": len(scored),
            "users": int(scored["user_id"].nunique()),
            "models": model_metrics,
            "gain_pp_vs_endpoint": model_metrics["gru_blend"]["hitrate_at_4_pct"]
            - model_metrics["endpoint_reference"]["hitrate_at_4_pct"],
            "paired_95pct_gain_interval_pp": gain_interval,
            "bootstrap_resamples": 2000,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "catalogue_cities": len(base["experts"]["catalog"]),
            "unsupported_targets": int(
                (
                    base["experts"]["catalog"].get_indexer(scored["target_city_id"]) < 0
                ).sum()
            ),
            "mean_gate_weights": dict(
                zip([*EXPERTS, "gru"], weights.mean(axis=0).tolist())
            ),
            "mean_endpoint_gate_weights": dict(
                zip(EXPERTS, endpoint_weights.mean(axis=0).tolist())
            ),
            "gate_training_objective": float(result.fun),
            "gate_l2": 0.01,
            "lag_floor": 0.5,
            "training_parameters": training_parameters,
            "source_sha256": source_sha256,
            "prediction_sha256": prediction_sha256,
            "runtime": runtime,
            "torch_threads": torch.get_num_threads(),
            "public_labels_previously_inspected": phase == "public",
        }
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        for name, values in model_metrics.items():
            print(
                f"  {phase} {name}: HitRate@4={values['hitrate_at_4_pct']:.2f}%, NDCG@4={values['ndcg_at_4_pct']:.2f}%",
                flush=True,
            )
        print(
            f"  paired gain: {metrics['gain_pp_vs_endpoint']:+.2f} pp, 95% interval {gain_interval}",
            flush=True,
        )
        print(f"  artifacts: {directory}", flush=True)
