# Hybrid itinerary recommender

Recommend **four possible final cities for a multi-stop hotel trip**, using its
observed itinerary and the context available for the final reservation. The task
uses the anonymized Booking.com Multi-Destination Trips dataset.

The best model combines seven statistical experts with a GRU sequence model.
A learned blending gate adjusts their contributions for each trip.

**Public HitRate@4: 51.73% · NDCG@4: 42.00% · 70,662 trips.**

## Task and data

Each row describes a hotel reservation. Reservations share a trip ID and are
ordered by check-in date. The local training file contains **1,166,835
reservations, 217,686 trips, 200,153 users and 39,901 city IDs** before filtering.

| Information | How it is used |
|---|---|
| Cities and hotel countries of earlier stays | The complete ordered itinerary, including repeated visits. |
| Check-in and check-out dates | Stay durations and calendar features. These are stay dates, not booking timestamps. |
| Booker country, device and affiliate | Categorical context supplied with a reservation. |
| User and trip IDs | Grouping, support counts and leakage-safe splits; excluded from predictive inputs. |
| City of the final stay | The supervised target, separated from the visible itinerary. |

For example, a training trip `A → B → C → D` becomes the input `A → B → C`
with target `D`. The observed part is called the **prefix**. The model scores
all cities in its fitting catalogue and returns four distinct suggestions.
A previously visited city remains eligible: trips can end where they started.

The benchmark conceals the last reservation's city and hotel country, but keeps
some final-row context. The model uses its **booker country, device, affiliate,
check-in month, weekday and stay duration**. Training reproduces that same
information boundary; the final hotel country is excluded from predictors.

From a recommender-system perspective, the trip plays the role of a session.
Predictions depend on the current itinerary and patterns learned across trips,
rather than a learned profile for each user. Bookings provide **implicit positive
feedback**: an observed destination is a training outcome, while an unobserved
city does not establish a traveler's dislike. Anonymous city IDs are categorical;
the model learns their relationships from travel behavior.

## Recommendation process

1. **Split users and prepare trips.** Hold out 20% of users for development
   evaluation. Within each partition, order stays and exclude complete trips
   with invalid IDs, invalid dates or durations, conflicting users, or tied
   check-ins. Keeping the whole trip together avoids constructing an artificial
   itinerary by dropping individual bad stays.
2. **Build one endpoint example per eligible trip.** Separate the last city from
   all earlier stays, retaining the full prefix, repeated visits and allowed
   final-row context. Single-stay trips contribute fitting evidence but cannot
   provide a supervised prefix/endpoint pair.
3. **Estimate seven statistical distributions.** Learn route patterns, city
   roles, endpoint frequencies, return probabilities and context corrections
   from fitting users. Smooth count estimates toward broader priors so limited
   observations do not produce extreme probabilities.
4. **Train the GRU on complete prefixes.** Encode cities, countries and durations,
   read the ordered stays, and combine the sequence representation with context
   and trip summaries. Predict the true endpoint with a city softmax.
5. **Learn how to combine the models.** Use three inner user folds: fit the
   statistical experts and GRU on two folds and score users in the third. These
   out-of-fold predictions train the blending gate. Refit the experts and GRU
   on all fitting users before evaluating the resulting recommender.
6. **Rank and evaluate.** Combine the eight probability distributions, rank the
   complete fitting catalogue and export four cities per trip. Evaluate the
   exported lists against the hidden endpoints; public ground truth is opened
   only after prediction export.

## Model architecture and decisions

### Statistical base: explicit travel patterns

Each expert produces a normalized probability distribution over the same city
catalogue. Their separation makes the behavioral assumptions visible.

| Expert | Signal and modeling decision |
|---|---|
| Lag | Directed evidence for later stays from every prefix city. Groups city pairs by their separation in the itinerary (1/2/3/4+ stays) and gives recent prefix stays more weight. |
| Terminal | How often a city is the true endpoint; distinguishes endpoint frequency from overall booking popularity. |
| Origin | Final destination conditional on the first city; retains information that a recent-stay-only model would miss. |
| Recent route | Final destination conditional on the last two cities jointly; captures an interaction beyond either city alone. |
| Visit return | Evidence for ending in any previously visited city. Each city contributes one return trial per trip, even after repeated visits. |
| Origin return | Evidence for returning to the starting city; models round-trip structure explicitly. |
| Context | Corrects the lag distribution using the six allowed final-row fields. |

City start/end/prefix counts describe **city roles**. Return probabilities use
smoothed success/trial counts, while route and endpoint distributions use count
smoothing of strength 10. Depending on the expert, the broader prior is booking
popularity, terminal-city frequency or a global return rate. These are behavioral
estimates; the return experts normalize their adjusted scores to form comparable
destination distributions.

### GRU extension: learned sequence representation

The count experts specify useful travel patterns directly. The **gated recurrent
unit (GRU)** learns additional interactions from the order and content of the
complete itinerary. The two approaches form a hybrid recommender.

| Component | Representation and purpose |
|---|---|
| Stay token | City embedding (32), hotel-country embedding (8) and `log1p(nights)`, projected to width 32. IDs are learned categories, not numerical coordinates. |
| Sequence encoder | One-layer GRU of width 32 reads every observed stay, without truncation. |
| Explicit position features | First, last and penultimate stay vectors accompany the GRU summary, preserving direct access to the origin and recent route. |
| Final-row context | Six categorical embeddings of width 8. |
| Trip summaries | `log1p` of stay count, distinct-city count, mean nights, last-stay nights and total nights. |
| City prediction head | A 64-unit ReLU layer with dropout 0.1 produces a 32-dimensional trip vector. Dot products with the shared input/output city embeddings, plus city biases, give city logits. |

Training uses **multiclass cross-entropy on the true endpoint** and a full-catalogue
softmax. Sharing the city embeddings ties how a city is represented in the
history to how it is scored as a destination. Unknown input categories have
reserved tokens; cities absent from the fitting catalogue cannot be recommended.

### Support-aware blending: weights depend on the trip

The gate uses **13 observable support and agreement features**, including prefix
length, route/origin/pair counts, return and context evidence, city roles and
agreement between experts. Support describes how much fitting evidence backs a
signal. This lets the mixture respond to the evidence available for a particular
trip instead of using one fixed average for every itinerary.

For prefix and context `x`, the final city probability is:

$$
p(c\mid x)=0.5\,p_{lag}(c\mid x)
+0.5\sum_e w_e(x)\,p_e(c\mid x).
$$

A linear model of the 13 standardized features produces eight expert logits;
softmax converts them to weights `w_e(x)` that sum to one. The residual mixture
includes all seven statistical experts and the GRU, so **lag retains at least
50% weight**. The gate minimizes destination log loss with L2 penalty 0.01;
standardization is fitted on the out-of-fold training rows.

In the recorded public run, mean weights were **50.00% lag, 48.36% GRU** and
1.64% across the remaining experts. Most probability weight is therefore assigned
to historical transition evidence and the learned sequence representation.

## Training and evaluation

Development uses an **80/20 whole-user split, seed 43**. Keeping every user's
trips on one side avoids sharing that user's travel history across fitting and
evaluation. Three inner user folds (seed 20261012) supply gate-training predictions
for at most 20,000 users; all other-fold users still supply fitting evidence.
Vocabularies, counts and neural weights are learned within each fitting fold.

The GRU uses AdamW, learning rate 0.001, weight decay 0.0001, batch size 128,
12 epochs, gradient clipping 5 and seed 20261013. Both training and evaluation
score the complete fitting catalogue. Evaluation introduces no sampled candidate
set or injected correct destination, and unavailable targets remain misses.

Two metrics assess the four-city lists:

- **HitRate@4:** the percentage of trips whose actual endpoint appears anywhere
  in the four recommendations.
- **NDCG@4:** gives more credit when that endpoint ranks earlier. With one relevant
  city, the score is `1 / log2(rank + 1)` for a hit and zero otherwise.

The **endpoint reference** is the seven-expert statistical blend, providing a
baseline for the contribution of the GRU.

| Population | Trips | Endpoint reference HitRate@4 | GRU blend HitRate@4 |
|---|---:|---:|---:|
| Development, held-out users | 43,647 | 49.18% | **50.37%** |
| Public benchmark | 70,662 | 50.52% | **51.73%** |

The GRU blend achieves public **NDCG@4 of 42.00%** and hits **36,553 targets**,
adding **854 hits (+1.21 percentage points)** over the endpoint reference.
The exploratory 95% gain interval is **[+1.03, +1.40] pp**, from 2,000 paired
user-cluster bootstrap resamples (seed 42). Resampling whole users preserves
within-user dependence when comparing the two models.

These scores come from the original full training run. Public labels were
previously inspected, so the benchmark is descriptive rather than an independent
test or official challenge placing. The task measures recovery of a recorded
endpoint. Anonymous city IDs provide no distances, and the dataset supplies no
availability information; offline accuracy does not establish online usefulness.

## Code and local execution

The three scripts expose the analytical steps directly. Their numbered sections
follow the architecture above; [Makefile](Makefile) runs the stages in order.

| Stage | File | Main responsibilities |
|---|---|---|
| 1. Data and features | [data_features.py](pipeline/data_features.py) | CSV loading, user split, cleaning, prefix/label separation, allowed context, categorical encoding and neural summaries. |
| 2. Base recommender | [base_recommender.py](pipeline/base_recommender.py) | Seven experts, city roles and returns, support features, user cross-fitting, mixture loss and base gate. |
| 3. GRU extension and evaluation | [gru_extension.py](pipeline/gru_extension.py) | Network architecture, training loop, eight-expert gate, prediction, metrics and paired intervals. |

Data preparation uses pandas. Sparse counts, softmax and optimization use SciPy;
folds and scaling use scikit-learn. PyTorch supplies the GRU and training APIs.
The only model class is `ItineraryGRU`.

Place the official CSV files locally:

```text
data/multi-destination-trip-dataset/
    train_set.csv
    test_set.csv
    ground_truth.csv
```

Create the environment and run the development evaluation from the repository root:

```bash
conda env create -f environment.yml
conda activate booking-recommender
make
```

To also refit on all training users and evaluate the public benchmark:

```bash
make PUBLIC=--public
```

Each phase fits three fold GRUs and one full-data GRU, using four CPU threads.
Completed folds and networks reload from cache. `make EPOCHS=1` uses a shorter
training budget for a smoke run. When changing that budget, give both model
stages the same fresh `--output-dir` to keep cached runs separate.

The stages can also be run individually:

```bash
python -m pipeline.data_features
python -m pipeline.base_recommender
python -m pipeline.gru_extension
```

File paths are declared at the top of the scripts. Prepared partitions go to
`data/processed/`; predictions, metrics, training losses, mean gate weights and
fold audits go to `results/gru_showcase/`. Raw data, weights and historical
experiments stay local. Only the third stage opens evaluation requests.

Run the integration tests with:

```bash
make test
```

Tests execute the three stages on local CSV fixtures and check filtering, user
separation, prefix/label boundaries, output schemas and cache reload. The recorded
model run used Python 3.10.6 and PyTorch 2.7.1 on CPU.

Dataset: Goldenberg and Levin, *Booking.com Multi-Destination Trips Dataset*,
SIGIR 2021 ([paper](https://doi.org/10.1145/3404835.3463240),
[official data and terms](https://github.com/bookingcom/ml-dataset-mdt)).
The neural city head and GRU adapt ideas from
[NVIDIA's challenge solution](https://ceur-ws.org/Vol-2855/challenge_short_2.pdf).
