"""Exercise the published stages and input boundaries using synthetic fixtures."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from pipeline.base_recommender import fit_experts, rank_recommendations
from pipeline.data_features import (
    CITY_COLUMNS,
    file_digest,
    prepare_challenge_requests,
    prepare_training_data,
)


@pytest.fixture
def benchmark_rows():
    return pd.DataFrame(
        {
            "user_id": ["1"] * 3,
            "utrip_id": ["trip"] * 3,
            "city_id": ["1", "2", "0"],
            "hotel_country": ["country", "country", "concealed"],
            "checkin": ["2016-01-01", "2016-01-02", "2016-01-03"],
            "checkout": ["2016-01-02", "2016-01-03", "2016-01-04"],
            "booker_country": ["booker"] * 3,
            "device_class": ["desktop"] * 3,
            "affiliate_id": ["1"] * 3,
        }
    )


@pytest.mark.parametrize("padded_rows", [[0], [0, 1, 2]])
def test_benchmark_normalizes_ids_without_losing_requests(benchmark_rows, padded_rows):
    expected, expected_audit = prepare_challenge_requests(benchmark_rows)
    benchmark_rows.loc[padded_rows, "utrip_id"] = " trip "
    benchmark_rows.loc[padded_rows, "user_id"] = " 1 "
    actual, audit = prepare_challenge_requests(benchmark_rows)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(audit, expected_audit)
    assert actual.iloc[0].city_history == ("1", "2")


@pytest.mark.parametrize(
    "tied_row, other_row, tied_prefix_rows", [(1, 0, 2), (2, 1, 1)]
)
def test_benchmark_ties_keep_request_with_empty_history(
    benchmark_rows, tied_row, other_row, tied_prefix_rows
):
    benchmark_rows.loc[tied_row, "checkin"] = benchmark_rows.loc[other_row, "checkin"]
    requests, audit = prepare_challenge_requests(benchmark_rows)
    assert len(requests) == len(audit) == 1
    assert not audit.iloc[0].usable_prefix
    assert audit.iloc[0].tied_checkin_rows == tied_prefix_rows
    for name in ("city_history", "country_history", "night_history"):
        assert requests.iloc[0][name] == ()


def test_benchmark_normalization_exposes_conflicting_users(benchmark_rows):
    benchmark_rows.loc[0, ["utrip_id", "user_id"]] = [" trip ", "2"]
    with pytest.raises(ValueError, match="exactly one user"):
        prepare_challenge_requests(benchmark_rows)


@pytest.mark.parametrize("checkout", ["2016-01-03", "2016-01-05", "2016-01-06"])
def test_final_tie_does_not_depend_on_checkout_or_input_order(benchmark_rows, checkout):
    benchmark_rows.loc[1, "checkout"] = "2016-01-05"
    benchmark_rows.loc[2, ["checkin", "checkout"]] = ["2016-01-02", checkout]
    requests, audit = prepare_challenge_requests(benchmark_rows)
    shuffled, shuffled_audit = prepare_challenge_requests(benchmark_rows.iloc[::-1])
    pd.testing.assert_frame_equal(requests, shuffled)
    pd.testing.assert_frame_equal(audit, shuffled_audit)
    assert requests.iloc[0].city_history == ()
    assert not audit.iloc[0].usable_prefix
    assert requests.iloc[0].final_stay_nights == str(
        (pd.Timestamp(checkout) - pd.Timestamp("2016-01-02")).days
    )


def test_multiple_concealed_rows_at_latest_checkin_are_rejected(benchmark_rows):
    benchmark_rows.loc[1, ["city_id", "checkin"]] = ["0", "2016-01-03"]
    with pytest.raises(ValueError, match="exactly one dated final reservation"):
        prepare_challenge_requests(benchmark_rows)


def test_lag_reference_has_hand_calculated_ranking(benchmark_rows):
    trips = []
    for user, cities in enumerate([(1, 2), (1, 2), (1, 3), (4, 5)], start=1):
        trip = benchmark_rows.iloc[:2].copy()
        trip["user_id"] = str(user)
        trip["utrip_id"] = f"trip_{user}"
        trip["city_id"] = list(map(str, cities))
        trips.append(trip)
    events, requests, labels = prepare_training_data(
        pd.concat(trips, ignore_index=True)
    )
    experts = fit_experts(events, requests, labels)
    predictions, weights = rank_recommendations(experts, None, requests.iloc[:1])
    # Source 1 has counts (0, 2, 1, 0, 0). Adding ten popularity pseudo-counts
    # yields (3.75, 4.5, 2.25, 1.25, 1.25), with the last tie broken by city ID.
    assert predictions[CITY_COLUMNS].iloc[0].tolist() == ["2", "1", "3", "4"]
    np.testing.assert_array_equal(weights, [[1, 0, 0, 0, 0, 0, 0]])
    with pytest.raises(ValueError, match="require a fitted mixture gate"):
        rank_recommendations(experts, None, requests, neural_batches=iter([]))


def test_file_digest_streams_multiple_blocks(tmp_path):
    payload = b"fixture" * 300000
    path = tmp_path / "artifact"
    path.write_bytes(payload)
    assert file_digest(path) == hashlib.sha256(payload).hexdigest()


def test_published_evidence_hashes_and_score_contract():
    directory = (
        Path(__file__).resolve().parents[1] / "results/published/2026-10-13"
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["snapshot_date"] == "2026-10-13"
    assert set(manifest["files"]) == {
        "data_audit.csv", "development_scores.csv", "public_scores.csv",
        "gru_gate_weights.csv", "public_paired_comparisons.csv",
        "neural_protocol.json", "neural_selection.json",
        "public_evaluation_record.json", "public_verification.json",
    }
    for name, record in manifest["files"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == (
            record["sha256"]
        )
    for filename, requests, users, methods in (
        ("development_scores.csv", 43647, 40011, 10),
        ("public_scores.csv", 70662, 68502, 12),
    ):
        scores = pd.read_csv(directory / filename).set_index("method")
        assert len(scores) == methods and scores.index.is_unique
        assert {"lag", "endpoint_blend", "blend_gru"}.issubset(scores.index)
        assert scores["requests"].eq(requests).all()
        assert scores["users"].eq(users).all()
        for metric in ("hitrate_at_4_pct", "ndcg_at_4_pct"):
            assert scores[metric].between(0, 100).all()
        assert scores["complete_list_coverage_pct"].eq(100).all()
        assert scores["missing_prediction_rows"].eq(0).all()
    paired = pd.read_csv(directory / "public_paired_comparisons.csv")
    comparison = paired.loc[
        paired["cohort"].eq("overall") & paired["candidate"].eq("blend_gru")
    ]
    assert len(comparison) == 1
    result = comparison.iloc[0]
    assert result.reference == "endpoint_blend"
    assert (result.reference_hits, result.candidate_hits, result.net_hits) == (
        35699, 36553, 854
    )
    assert result.candidate_hitrate_pct == pytest.approx(
        scores.loc["blend_gru", "hitrate_at_4_pct"]
    )
    weights = pd.read_csv(directory / "gru_gate_weights.csv", index_col=0)
    assert weights["mean_weight"].sum() == pytest.approx(1)
    assert weights.loc["lag", "mean_weight"] >= 0.5
    protocol = json.loads((directory / "neural_protocol.json").read_text())
    assert protocol["development_seed"] == 43
    assert protocol["neural_config"]["seed"] == 20261013
    evaluation = json.loads((directory / "public_evaluation_record.json").read_text())
    assert evaluation["requests"] == 70662
    assert evaluation["train_test_user_overlap"] == 9496
    assert evaluation["train_test_trip_overlap"] == 0


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("three_stage_recommender")
    repository = Path(__file__).resolve().parents[1]
    (root / "pipeline").mkdir()
    for name in ("data_features.py", "base_recommender.py", "gru_extension.py"):
        shutil.copy2(repository / "pipeline" / name, root / "pipeline" / name)
    shutil.copy2(repository / "Makefile", root / "Makefile")
    data = root / "data/multi-destination-trip-dataset"
    data.mkdir(parents=True)
    rows = []
    for user in range(1, 19):
        cities = [1, 2, 3, 1] if user % 4 == 0 else [1, 2, 3, 100 + user]
        if user % 3 == 0:
            cities = [1, 2, 1, 3, 2, 3, 2, cities[-1]]
        for position, city in enumerate(cities):
            start = pd.Timestamp("2016-01-01") + pd.Timedelta(days=position)
            rows.append(
                {
                    "user_id": str(user),
                    "utrip_id": f"trip_{user}",
                    "city_id": str(city),
                    "hotel_country": f"country_{city % 2}",
                    "checkin": str(start.date()),
                    "checkout": str((start + pd.Timedelta(days=1)).date()),
                    "booker_country": "booker",
                    "device_class": "desktop",
                    "affiliate_id": str(user % 3),
                }
            )
    valid = pd.DataFrame(rows)
    singleton = valid.iloc[:1].assign(user_id="20", utrip_id="singleton", city_id="77")
    bad_city = valid.iloc[:4].assign(user_id="21", utrip_id="bad_city")
    bad_city.loc[1, "city_id"] = "0"
    bad_stay = valid.iloc[:4].assign(user_id="22", utrip_id="bad_stay")
    bad_stay.loc[1, "checkout"] = bad_stay.loc[1, "checkin"]
    tied = valid.iloc[:4].assign(user_id="23", utrip_id="tied")
    tied.loc[1, "checkin"] = tied.loc[0, "checkin"]
    conflicting = valid.iloc[:4].assign(user_id="24", utrip_id="conflicting")
    conflicting.loc[1, "user_id"] = "25"
    pd.concat(
        [valid, singleton, bad_city, bad_stay, tied, conflicting], ignore_index=True
    ).to_csv(data / "train_set.csv", index=False)
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    outputs = []
    # Test files do not exist while the data and base stages run.
    for module, arguments in (
        ("pipeline.data_features", []),
        ("pipeline.base_recommender", ["--public"]),
    ):
        result = subprocess.run(
            [sys.executable, "-m", module, *arguments],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(result.stdout)
    test = valid.loc[valid["utrip_id"].isin(["trip_1", "trip_2", "trip_3"])].copy()
    test["utrip_id"] = "public_" + test["utrip_id"]
    final = test.groupby("utrip_id", sort=False).tail(1).index
    test.loc[final, ["utrip_id", "city_id"]].to_csv(
        data / "ground_truth.csv", index=False
    )
    test.loc[final, "city_id"] = "0"
    test.loc[final, "hotel_country"] = "concealed"
    # One padded ID previously disappeared from the evaluation denominator.
    test.loc[test["utrip_id"].eq("public_trip_1"), "utrip_id"] = " public_trip_1 "
    test.to_csv(data / "test_set.csv", index=False)
    command = [
        sys.executable,
        "-m",
        "pipeline.gru_extension",
        "--epochs",
        "1",
        "--public",
    ]
    result = subprocess.run(
        command, cwd=root, env=environment, text=True, capture_output=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outputs.append(result.stdout)
    return root, environment, command, outputs


def test_three_stages_rows_boundaries_metrics_and_resume(completed_run):
    root, environment, command, outputs = completed_run
    summary = json.loads((root / "results/gru_showcase/data_summary.json").read_text())
    assert summary["split_seed"] == 43
    assert summary["input_sha256"] == {
        "data/multi-destination-trip-dataset/train_set.csv": hashlib.sha256(
            (root / "data/multi-destination-trip-dataset/train_set.csv").read_bytes()
        ).hexdigest()
    }
    all_training = joblib.load(root / "data/processed/public_train.joblib")
    fitting = joblib.load(root / "data/processed/development_train.joblib")
    holdout = joblib.load(root / "data/processed/development_holdout.joblib")
    assert len(all_training["events"]) == 97
    assert len(all_training["requests"]) == len(all_training["labels"]) == 18
    assert set(all_training["events"]["utrip_id"]) == {
        *(f"trip_{user}" for user in range(1, 19)),
        "singleton",
    }
    assert not set(fitting["events"]["user_id"]).intersection(
        holdout["events"]["user_id"]
    )
    assert "target_city_id" not in all_training["requests"]
    assert "hotel_country" not in all_training["requests"]
    longest = all_training["requests"].set_index("utrip_id").loc["trip_3"]
    assert longest.city_history == ("1", "2", "1", "3", "2", "3", "2")
    assert longest.country_history == (
        "country_1",
        "country_0",
        "country_1",
        "country_1",
        "country_0",
        "country_1",
        "country_0",
    )
    assert longest.night_history == (1,) * 7
    assert "epoch 1/1" in outputs[-1]
    original = {}
    original_metrics = {}
    for phase in ("development", "public"):
        directory = root / "results/gru_showcase" / phase
        metrics = json.loads((directory / "metrics.json").read_text())
        assert metrics["training_parameters"]["epochs"] == 1
        assert metrics["mean_gate_weights"]["lag"] >= 0.5
        assert np.isclose(sum(metrics["mean_gate_weights"].values()), 1)
        assert len(metrics["paired_95pct_gain_interval_pp"]) == 2
        assert metrics["bootstrap_resamples"] == 2000
        original_metrics[phase] = metrics
        for model in ("lag_reference", "endpoint_reference", "gru_blend"):
            assert 0 <= metrics["models"][model]["hitrate_at_4_pct"] <= 100
            assert 0 <= metrics["models"][model]["ndcg_at_4_pct"] <= 100
        for filename in ("oof_audit.csv", "gru_oof_audit.csv"):
            audit = pd.read_csv(directory / filename)
            assert len(audit) == 3 and audit["overlapping_users"].eq(0).all()
        assignments = pd.read_csv(
            directory / "user_folds.csv", dtype={"user_id": "string"}
        )
        for fold in range(3):
            saved = torch.load(directory / f"gru_fold_{fold}.pt", weights_only=True)
            held = assignments.loc[assignments["fold"].eq(fold), "user_id"]
            assert not set(saved["fitting_users"]).intersection(held)
        base = joblib.load(directory / "base_model.joblib")
        assert metrics["base_model_signature"] == base["signature"]
        refit = torch.load(directory / "gru.pt", weights_only=True)
        assert metrics["gru_checkpoint_signature"] == refit["signature"]
        assert metrics["command"] == command
        assert metrics["python_version"] == sys.version
        assert metrics["training_evaluation_user_overlap"] == (
            0 if phase == "development" else 3
        )
        assert metrics["development_labels_previously_inspected"] == (
            phase == "development"
        )
        expected_inputs = (
            {
                "data/processed/development_train.joblib",
                "data/processed/development_holdout.joblib",
            }
            if phase == "development"
            else {
                "data/processed/public_train.joblib",
                "data/multi-destination-trip-dataset/test_set.csv",
                "data/multi-destination-trip-dataset/ground_truth.csv",
            }
        )
        assert set(metrics["input_sha256"]) == expected_inputs
        for name, digest in metrics["input_sha256"].items():
            assert digest == hashlib.sha256((root / name).read_bytes()).hexdigest()
        truth = (
            holdout["labels"].set_index("utrip_id")["target_city_id"]
            if phase == "development"
            else pd.read_csv(
                root / "data/multi-destination-trip-dataset/ground_truth.csv",
                dtype="string",
            ).set_index("utrip_id")["city_id"]
        )
        for model, filename in (
            ("lag_reference", "lag_predictions.csv"),
            ("endpoint_reference", "endpoint_predictions.csv"),
            ("gru_blend", "predictions.csv"),
        ):
            path = directory / filename
            assert metrics["prediction_sha256_by_model"][model] == hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            lists = pd.read_csv(path, dtype="string").set_index("utrip_id")
            assert set(lists.index) == set(truth.index)
            assert lists[CITY_COLUMNS].nunique(axis=1).eq(4).all()
            ranks = [
                cities.index(truth.loc[trip]) + 1 if truth.loc[trip] in cities else 0
                for trip, cities in zip(
                    lists.index, lists[CITY_COLUMNS].values.tolist()
                )
            ]
            scores = metrics["models"][model]
            assert scores["hits"] == sum(rank > 0 for rank in ranks)
            assert scores["hitrate_at_4_pct"] == pytest.approx(
                100 * sum(rank > 0 for rank in ranks) / len(ranks)
            )
            assert scores["ndcg_at_4_pct"] == pytest.approx(
                100 * sum(1 / np.log2(rank + 1) if rank else 0 for rank in ranks)
                / len(ranks)
            )
        predictions = pd.read_csv(directory / "predictions.csv", dtype="string")
        assert len(predictions) == metrics["requests"]
        city_columns = [f"city_id_{rank}" for rank in range(1, 5)]
        assert predictions[city_columns].nunique(axis=1).eq(4).all()
        assert set(predictions[city_columns].to_numpy().ravel()).issubset(
            base["experts"]["catalog"]
        )
        original[phase] = predictions
    assert len(original["public"]) == 3

    # Labels and the concealed row's hotel country cannot alter predictions.
    data = root / "data/multi-destination-trip-dataset"
    truth_path = data / "ground_truth.csv"
    truth = pd.read_csv(truth_path, dtype="string")
    truth["city_id"] = "999999"
    truth.to_csv(truth_path, index=False)
    test_path = data / "test_set.csv"
    masked = pd.read_csv(test_path, dtype="string")
    masked.loc[masked["city_id"].eq("0"), "hotel_country"] = "another_secret"
    masked.to_csv(test_path, index=False)
    second = subprocess.run(
        command, cwd=root, env=environment, text=True, capture_output=True, timeout=120
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert "epoch 1/1" not in second.stdout
    for phase, predictions in original.items():
        pd.testing.assert_frame_equal(
            predictions,
            pd.read_csv(
                root / "results/gru_showcase" / phase / "predictions.csv",
                dtype="string",
            ),
        )
    metrics = json.loads(
        (root / "results/gru_showcase/public/metrics.json").read_text()
    )
    assert metrics["models"]["gru_blend"]["hits"] == 0
    assert metrics["unsupported_targets"] == 3
    previous = original_metrics["public"]
    for name in ("test_set.csv", "ground_truth.csv"):
        key = f"data/multi-destination-trip-dataset/{name}"
        assert metrics["input_sha256"][key] != previous["input_sha256"][key]
    for key in (
        "prediction_sha256_by_model", "base_model_signature", "gru_checkpoint_signature"
    ):
        assert metrics[key] == previous[key]
    for scores in metrics["models"].values():
        assert scores["hitrate_at_4_95pct_interval_pct"] == [0, 0]
        assert scores["ndcg_at_4_95pct_interval_pct"] == [0, 0]
    assert metrics["paired_95pct_gain_interval_pp"] == [0, 0]


def test_real_boundary_failures(completed_run):
    root, environment, command, _ = completed_run
    stale = subprocess.run(
        [*command[:-3], "--epochs", "2", "--public"],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert stale.returncode != 0 and "Stale GRU checkpoint" in stale.stderr
    truth_path = root / "data/multi-destination-trip-dataset/ground_truth.csv"
    original_truth = truth_path.read_bytes()
    truth = pd.read_csv(truth_path, dtype="string")
    pd.concat([truth, truth.iloc[:1].assign(utrip_id="extra_trip")]).to_csv(
        truth_path, index=False
    )
    try:
        mismatched = subprocess.run(
            command, cwd=root, env=environment, text=True,
            capture_output=True, timeout=120
        )
        assert mismatched.returncode != 0
        assert "match exactly the public request IDs" in mismatched.stderr
    finally:
        truth_path.write_bytes(original_truth)
    path = root / "data/multi-destination-trip-dataset/test_set.csv"
    test = pd.read_csv(path, dtype="string")
    test.loc[test["city_id"].eq("0"), "city_id"] = "1"
    test.to_csv(path, index=False)
    unmasked = subprocess.run(
        command, cwd=root, env=environment, text=True, capture_output=True, timeout=120
    )
    assert unmasked.returncode != 0 and "concealed city" in unmasked.stderr
