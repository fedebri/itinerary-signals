"""Exercise the three published stages using actual local CSV files."""

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
    for phase in ("development", "public"):
        directory = root / "results/gru_showcase" / phase
        metrics = json.loads((directory / "metrics.json").read_text())
        assert metrics["training_parameters"]["epochs"] == 1
        assert metrics["mean_gate_weights"]["lag"] >= 0.5
        assert np.isclose(sum(metrics["mean_gate_weights"].values()), 1)
        assert len(metrics["paired_95pct_gain_interval_pp"]) == 2
        assert metrics["bootstrap_resamples"] == 2000
        for model in ("endpoint_reference", "gru_blend"):
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
    path = root / "data/multi-destination-trip-dataset/test_set.csv"
    test = pd.read_csv(path, dtype="string")
    test.loc[test["city_id"].eq("0"), "city_id"] = "1"
    test.to_csv(path, index=False)
    unmasked = subprocess.run(
        command, cwd=root, env=environment, text=True, capture_output=True, timeout=120
    )
    assert unmasked.returncode != 0 and "concealed city" in unmasked.stderr
