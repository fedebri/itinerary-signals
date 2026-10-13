"""Stage 1: local CSV loading, whole-user split and predictor-only features.

Run from the repository root: python -m pipeline.data_features
No learned preprocessing is fitted here. Vocabularies and count statistics are
learned separately inside each fitting fold in stages 2 and 3.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence

# Local inputs, outputs and the whole-user split seed.
TRAIN_CSV = Path("data/multi-destination-trip-dataset/train_set.csv")
DEVELOPMENT_TRAIN = Path("data/processed/development_train.joblib")
DEVELOPMENT_HOLDOUT = Path("data/processed/development_holdout.joblib")
PUBLIC_TRAIN = Path("data/processed/public_train.joblib")
DATA_SUMMARY = Path("results/gru_showcase/data_summary.json")
SPLIT_SEED = 43

RAW_COLUMNS = [
    "user_id",
    "checkin",
    "checkout",
    "city_id",
    "device_class",
    "affiliate_id",
    "booker_country",
    "hotel_country",
    "utrip_id",
]
CITY_COLUMNS = ["city_id_1", "city_id_2", "city_id_3", "city_id_4"]
FINAL_CONTEXT_COLUMNS = (
    "final_booker_country",
    "final_device_class",
    "final_affiliate_id",
    "final_checkin_month",
    "final_checkin_weekday",
    "final_stay_nights",
)
NUMERIC_FEATURES = (
    "log_stays",
    "log_distinct_cities",
    "log_mean_nights",
    "log_last_nights",
    "log_total_nights",
)


# %% 1. Allowed final-row context: six fields, excluding the destination labels.
def final_row_context(rows: pd.DataFrame) -> pd.DataFrame:
    """Extract only supplied non-label fields from one final row per trip.

    Dates are stay dates, not reservation-creation timestamps. Missing context
    remains missing and supplies no fitted context correction at inference.
    City, hotel country, user and trip ID values are never context predictors.
    """
    context = rows[
        ["utrip_id", "booker_country", "device_class", "affiliate_id"]
    ].astype("string")
    context = context.apply(lambda x: x.str.strip().replace("", pd.NA))
    context = context.rename(
        columns={name: f"final_{name}" for name in context if name != "utrip_id"}
    )
    checkin = pd.to_datetime(rows["checkin"], format="%Y-%m-%d", errors="coerce")
    checkout = pd.to_datetime(rows["checkout"], format="%Y-%m-%d", errors="coerce")
    context["final_checkin_month"] = checkin.dt.month.astype("Int64").astype("string")
    context["final_checkin_weekday"] = checkin.dt.dayofweek.astype("Int64").astype(
        "string"
    )
    nights = (checkout - checkin).dt.days
    context["final_stay_nights"] = (
        nights.where(nights.gt(0)).astype("Int64").astype("string")
    )
    return context


# %% 2. Complete training trips: retain singletons as evidence, hide true endpoints.
def prepare_training_data(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preserve the recorded whole-trip filters and one endpoint per trip.

    Singletons remain fitting observations, but supply no supervised request.
    Invalid stays, conflicting users or tied check-ins exclude the whole trip.
    """
    rows = raw[RAW_COLUMNS].astype("string").reset_index(drop=True)
    rows = rows.apply(lambda column: column.str.strip().replace("", pd.NA))
    for name in ("checkin", "checkout"):
        rows[name] = pd.to_datetime(rows[name], format="%Y-%m-%d", errors="coerce")
    valid = rows["utrip_id"].notna()
    for name in ("user_id", "city_id"):
        numeric = pd.to_numeric(rows[name], errors="coerce")
        valid &= numeric.notna() & numeric.gt(0) & numeric.mod(1).eq(0)
    valid &= (
        rows["checkin"].notna()
        & rows["checkout"].notna()
        & rows["checkout"].gt(rows["checkin"])
    )
    valid = valid.fillna(False)
    groups = rows.groupby("utrip_id")
    # One bad reservation disqualifies its complete trip; dropping only that row
    # could create a different itinerary or a false endpoint.
    whole_trip_valid = valid.groupby(rows["utrip_id"]).all()
    tied = (
        rows.duplicated(["utrip_id", "checkin"], keep=False)
        .groupby(rows["utrip_id"])
        .any()
    )
    usable = whole_trip_valid.copy()
    for name, check in (
        ("valid IDs and positive-duration stays", whole_trip_valid),
        ("one user per trip", groups["user_id"].nunique().eq(1)),
        ("distinct check-ins", ~tied),
    ):
        usable &= check
        retained = rows["utrip_id"].isin(usable.index[usable])
        print(
            f"  {name}: {int(retained.sum()):,} stays, {int(usable.sum()):,} trips",
            flush=True,
        )
    events = (
        rows.loc[rows["utrip_id"].isin(usable.index[usable])]
        .sort_values(["utrip_id", "checkin", "checkout"], kind="stable")
        .copy()
    )
    sizes = events.groupby("utrip_id").size()
    eligible = events.loc[events["utrip_id"].isin(sizes.index[sizes.ge(2)])]
    # The true last stay supplies the label; every earlier stay remains visible.
    # Singletons still live in events, where they inform city priors and counts.
    final = eligible.groupby("utrip_id", sort=False).tail(1)
    prefix = eligible.drop(index=final.index).copy()
    prefix["nights"] = (prefix["checkout"] - prefix["checkin"]).dt.days
    requests = (
        prefix.groupby("utrip_id", sort=False)
        .agg(
            user_id=("user_id", "first"),
            city_history=("city_id", tuple),
            country_history=("hotel_country", tuple),
            night_history=("nights", tuple),
        )
        .reset_index()
        .merge(final_row_context(final), on="utrip_id", validate="one_to_one")
    )
    labels = final[["utrip_id", "user_id", "city_id"]].rename(
        columns={"city_id": "target_city_id"}
    )
    return events, requests, labels


# %% 3. Masked benchmark requests: same prefix/context semantics, no truth input.
def prepare_challenge_requests(
    test_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep every masked-test trip, using empty histories for invalid prefixes.

    The last dated row must conceal its city. Its hotel country is never read.
    The separate audit records prefix validity; it is not a predictor input.
    """
    rows = test_rows[RAW_COLUMNS].astype("string")
    missing_trip = rows["utrip_id"].isna() | rows["utrip_id"].str.strip().eq("")
    numeric_users = pd.to_numeric(rows["user_id"], errors="coerce")
    invalid_user = (
        numeric_users.isna() | numeric_users.le(0) | numeric_users.mod(1).ne(0)
    )
    if missing_trip.any() or invalid_user.any():
        raise ValueError(
            "Benchmark rows require nonblank trip IDs and positive integer user IDs."
        )
    if rows.groupby("utrip_id")["user_id"].nunique().ne(1).any():
        raise ValueError("Each benchmark trip must have exactly one user.")
    for column in ("checkin", "checkout"):
        rows[column] = pd.to_datetime(rows[column], format="%Y-%m-%d", errors="coerce")
    ordered = rows.sort_values(
        ["utrip_id", "checkin", "checkout"], kind="stable", na_position="first"
    )
    # Find the benchmark's masked last row before building histories. Its city
    # and hotel country must never enter predictor features.
    final = ordered.groupby("utrip_id").tail(1)
    final_city = pd.to_numeric(final["city_id"], errors="coerce")
    masked = (
        final["city_id"].isna() | final["city_id"].str.strip().eq("") | final_city.eq(0)
    ).fillna(False)
    if not masked.all() or final["checkin"].isna().any():
        raise ValueError(
            "Expected a dated final reservation with concealed city (blank or zero) for every trip."
        )
    prefix = ordered.drop(index=final.index).copy()
    numeric_cities = pd.to_numeric(prefix["city_id"], errors="coerce")
    prefix["invalid_city"] = (
        numeric_cities.isna() | numeric_cities.le(0) | numeric_cities.mod(1).ne(0)
    )
    prefix["invalid_stay"] = (
        prefix["checkin"].isna()
        | prefix["checkout"].isna()
        | prefix["checkout"].le(prefix["checkin"])
    )
    prefix["tied_checkin"] = prefix.duplicated(["utrip_id", "checkin"], keep=False)
    summary = prefix.groupby("utrip_id").agg(
        known_stays=("utrip_id", "size"),
        invalid_city_rows=("invalid_city", "sum"),
        invalid_stay_rows=("invalid_stay", "sum"),
        tied_checkin_rows=("tied_checkin", "sum"),
    )
    audit = final[["utrip_id"]].join(summary, on="utrip_id").fillna(0)
    audit[summary.columns] = audit[summary.columns].astype(int)
    audit["usable_prefix"] = audit["known_stays"].gt(0) & audit[
        ["invalid_city_rows", "invalid_stay_rows", "tied_checkin_rows"]
    ].eq(0).all(axis=1)
    # Keep every request in the benchmark. An unusable observed prefix becomes
    # an empty history so downstream experts can fall back to fitted priors.
    # Use raw observed values with the original trimming semantics. Concealed
    # rows never contribute city, country or duration history.
    visible = test_rows.loc[prefix.index, RAW_COLUMNS].astype("string")
    visible = visible.apply(lambda column: column.str.strip().replace("", pd.NA))
    visible = visible.loc[
        visible["utrip_id"].isin(audit.loc[audit["usable_prefix"], "utrip_id"])
    ]
    for column in ("checkin", "checkout"):
        visible[column] = pd.to_datetime(visible[column], format="%Y-%m-%d")
    visible["nights"] = (visible["checkout"] - visible["checkin"]).dt.days
    groups = visible.groupby("utrip_id", sort=False)
    requests = final[["utrip_id", "user_id"]].merge(
        final_row_context(test_rows.loc[final.index]),
        on="utrip_id",
        validate="one_to_one",
    )
    for name, column in [
        ("city_history", "city_id"),
        ("country_history", "hotel_country"),
        ("night_history", "nights"),
    ]:
        histories = groups[column].agg(tuple)
        requests[name] = [histories.get(trip, ()) for trip in requests["utrip_id"]]
    return requests, audit.reset_index(drop=True)


# %% 4. Neural features: reserved unknown tokens, full histories and five summaries.
def encode_examples(requests, vocabulary, targets=None):
    """Encode variable-length prefixes as ordinary tuples for PyTorch batching.

    Index.get_indexer returns -1 for unseen values. Offsets reserve city 0 for
    padding and 1 for unknown, while other categorical fields use 0 for unknown.
    """
    context = np.column_stack(
        [
            vocabulary["contexts"][name].get_indexer(requests[name]) + 1
            for name in FINAL_CONTEXT_COLUMNS
        ]
    )
    labels = (
        vocabulary["cities"].get_indexer(targets)
        if targets is not None
        else np.full(len(requests), -100)
    )
    examples = []
    histories = requests[
        ["city_history", "country_history", "night_history"]
    ].itertuples(index=False, name=None)
    for (cities, countries, nights), values, target in zip(histories, context, labels):
        examples.append(
            (
                vocabulary["cities"].get_indexer(cities) + 2,
                vocabulary["countries"].get_indexer(countries) + 1,
                np.asarray(nights, dtype=np.float32),
                values,
                len(set(cities)),
                int(target),
            )
        )
    return examples


def collate_stays(examples):
    """Pad without truncation and derive the same five numerical prefix inputs.

    City 0 is padding, 1 is unknown, and real city indices start at 2. An empty
    prefix uses one padding token for packing; its length feature remains zero.
    """
    cities, countries, nights, context, numeric, targets = ([] for _ in range(6))
    lengths = []
    for city, country, night, values, distinct, target in examples:
        size = len(city)
        # Packed GRUs require a positive sequence length even when the usable
        # prefix is empty; numeric features still record the actual size zero.
        lengths.append(max(size, 1))
        cities.append(torch.as_tensor(city if size else [0], dtype=torch.long))
        countries.append(torch.as_tensor(country if size else [0], dtype=torch.long))
        nights.append(torch.as_tensor(night if size else [0.0], dtype=torch.float32))
        context.append(values)
        # Log scaling keeps long itineraries and stays from dominating the
        # embeddings while preserving their ordering and zero values.
        numeric.append(
            np.log1p(
                [
                    size,
                    distinct,
                    np.mean(night) if size else 0,
                    night[-1] if size else 0,
                    np.sum(night),
                ]
            )
        )
        targets.append(target)
    return {
        "cities": pad_sequence(cities, batch_first=True),
        "countries": pad_sequence(countries, batch_first=True),
        "nights": pad_sequence(nights, batch_first=True),
        "lengths": torch.tensor(lengths),
        "context": torch.as_tensor(np.asarray(context), dtype=torch.long),
        "numeric": torch.as_tensor(np.asarray(numeric), dtype=torch.float32),
        "target": torch.tensor(targets),
    }


# %% 5. Load real data, split whole users FIRST, then prepare each partition.
if __name__ == "__main__":
    raw = pd.read_csv(TRAIN_CSV, dtype="string", usecols=RAW_COLUMNS)
    raw = raw.apply(lambda column: column.str.strip().replace("", pd.NA))
    users = pd.Index(raw["user_id"].dropna().unique()).sort_values()
    fitting_users, development_users = train_test_split(
        users.to_numpy(), test_size=0.2, random_state=SPLIT_SEED
    )
    print(f"Input: {len(raw):,} stays, {len(users):,} users", flush=True)
    print(
        f"Whole-user split: {len(fitting_users):,} fitting / {len(development_users):,} held out",
        flush=True,
    )

    # Assign entire raw trips before filtering: conflicting-user trips must not
    # become valid merely because their rows landed in different partitions.
    trip_user = raw.groupby("utrip_id")["user_id"].transform("first")
    partitions = (
        (
            "development_train",
            DEVELOPMENT_TRAIN,
            raw.loc[trip_user.isin(fitting_users)],
        ),
        (
            "development_holdout",
            DEVELOPMENT_HOLDOUT,
            raw.loc[trip_user.isin(development_users)],
        ),
        ("public_train", PUBLIC_TRAIN, raw),
    )
    summaries = []
    for name, output, rows in partitions:
        print(f"\nPrepare {name}", flush=True)
        events, requests, labels = prepare_training_data(rows)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"events": events, "requests": requests, "labels": labels},
            output,
            compress=3,
        )
        summary = {
            "partition": name,
            "input_stays": len(rows),
            "usable_stays": len(events),
            "usable_trips": int(events["utrip_id"].nunique()),
            "endpoint_requests": len(requests),
            "fitting_cities": int(events["city_id"].nunique()),
            "users": int(events["user_id"].nunique()),
            "output": str(output),
        }
        summaries.append(summary)
        print(f"  {len(requests):,} true-endpoint requests -> {output}", flush=True)
    DATA_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    DATA_SUMMARY.write_text(
        json.dumps({"split_seed": SPLIT_SEED, "partitions": summaries}, indent=2) + "\n"
    )
