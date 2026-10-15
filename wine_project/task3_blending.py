"""
task3_blending.py

Task 3 - Improve the quality of a single wine through blending it with one 
or more wines from a list of candidates and their physiochemical properties.

Required function:

choose_candidates()

Please choose wines for testing this program from the
winequality_red_260918_TEST.csv file only!
"""

import numpy as np
import pandas as pd

import winequality

# Source of input and candidate wines to test this program:
TEST_FILE = "winequality_red_260918_TEST.csv"

# Blending search settings:
STEP = 0.05        # try candidate shares in 5% increments
MIN_PRIMARY = 0.50 # my wine must stay at least 50% of the blend
MAX_EXTRA = 3      # add at most 3 extra wines
PRIMARY_QUAL = 5   # quality score of the wine I start blending from


def wine_vector(wine, order):
    """
    Return the wine's property values as a numpy array, in the given order.

    wine  - dict of physiochemical values keyed by property name
    order - property names in the order the model expects them

    Assumes every name in `order` is a key in `wine`.  Lining the values up
    this way lets us blend and score wines with fast array arithmetic.
    """
    return np.array([wine[name] for name in order], dtype=float)


def blend_dict(wines, ratios, order):
    """
    Blend several wines into one, by volume-weighted average of each property.

    wines  - list of wine dicts to mix together
    ratios - the volume share of each wine, aligned with `wines`
    order  - property names to produce in the result

    Assumes `ratios` lines up with `wines` and sums to 1.  Returns a new wine
    dict whose value for each property is sum(ratio_i * wine_i[property]).
    """
    mix = np.zeros(len(order), dtype=float)
    for wine, ratio in zip(wines, ratios):
        mix += ratio * wine_vector(wine, order)
    return {name: value for name, value in zip(order, mix)}


def score_many(model, rows, order):
    """
    Predict the quality of many blends in a single model.predict() call.

    model - a fitted forest, as returned by winequality.build_model()
    rows  - list of blend vectors, each a numpy array in `order`
    order - property names, matching the order used to fit the model

    Returns a numpy array of predicted scores, one per row.  Scoring the
    whole batch at once is far faster than one call per blend.  The columns
    are named so the model recognises its own features.
    """
    frame = pd.DataFrame(rows, columns=order)
    return model.predict(frame)


def choose_candidates(primary, candidates, model):
    """
    Find which candidate wine blends with your primary wine will
    give the best quality.

    primary    - physiochemical properties of the wine from your estate
    candidates - list of wines to consider, taken from TEST_FILE
    model      - a fitted forest from winequality.build_model()

    The search is greedy: starting from the primary wine alone, it tries every
    candidate at 5% steps, keeps the single best blend, then tries adding
    another on top (earlier choices stay fixed), up to MAX_EXTRA extra wines,
    stopping as soon as nothing improves.  The primary wine always keeps at
    least MIN_PRIMARY of the blend.  All the trial blends of a round are scored
    together in one model.predict() batch; the winning blend is re-scored with
    winequality.predict().

    Assumes build_model() still reads the TRAIN file, but only to *fit* the
    model -- every wine blended or scored here comes from the TEST file.

    Returns (blend, score, chosen, primary_share):
        blend         - the winning blend as a wine dict
        score         - its winequality.predict() score (raw float)
        chosen        - list of (candidate_index, ratio) wines added
        primary_share - the primary wine's final share of the blend
    """
    order = list(model.feature_names_in_)
    primary_vec = wine_vector(primary, order)
    cand_vecs = [wine_vector(c, order) for c in candidates]

    # How many 5% steps the extra wines may take up in total.
    max_total_steps = round((1 - MIN_PRIMARY) / STEP)

    base = np.zeros(len(order), dtype=float)  # fixed part from chosen wines
    used_steps = 0                            # 5% steps already given away
    chosen = []                               # (candidate_index, ratio) so far
    used = set()                              # candidate indices already used
    best = float(score_many(model, [primary_vec], order)[0])

    for _ in range(MAX_EXTRA):
        # Build every trial blend for this round: an unused candidate given a
        # share of k*STEP, with the primary shrinking to make room.
        trial_vecs = []
        trial_meta = []  # (candidate_index, ratio) matching each trial vector
        for i, cand_vec in enumerate(cand_vecs):
            if i in used:
                continue
            for k in range(1, max_total_steps - used_steps + 1):
                ratio = k * STEP
                primary_share = 1 - used_steps * STEP - ratio
                trial_vecs.append(primary_share * primary_vec
                                  + base + ratio * cand_vec)
                trial_meta.append((i, ratio))

        if not trial_vecs:
            break  # no room left for another wine

        scores = score_many(model, trial_vecs, order)
        winner = int(np.argmax(scores))
        if scores[winner] <= best:
            break  # nothing improved, so stop adding wines

        # Lock in the best candidate of this round.
        best = float(scores[winner])
        i, ratio = trial_meta[winner]
        base += ratio * cand_vecs[i]
        used_steps += round(ratio / STEP)
        used.add(i)
        chosen.append((i, ratio))

    # Rebuild the winning blend and score it with winequality.predict().
    primary_share = 1 - used_steps * STEP
    wines = [primary] + [candidates[i] for i, _ in chosen]
    ratios = [primary_share] + [ratio for _, ratio in chosen]
    blend = blend_dict(wines, ratios, order)
    score = winequality.predict(model, blend, rounded=False)

    return blend, score, chosen, primary_share


def drop_quality(wine):
    """Return a copy of a wine dict without its 'quality' label.

    winequality.predict() only accepts the physiochemical properties, so the
    'quality' column must be removed before a wine can be scored or blended.
    """
    return {name: value for name, value in wine.items()
            if name != winequality.TARGET}


def print_result(primary_row, base_score, blend, score,
                 chosen, candidate_rows, primary_share):
    """Print the primary wine, the blend it found, and both scores.

    primary_row   - TEST row number of the primary wine
    base_score    - the primary wine's quality on its own
    blend         - the winning blend as a wine dict
    score         - the blend's quality score
    chosen        - list of (candidate_index, ratio) wines added
    candidate_rows- TEST row number for each candidate, by index
    primary_share - the primary wine's final share of the blend
    """
    print(f"Primary wine: TEST row {primary_row} (base score {base_score:.3f})")
    print()
    if chosen:
        print("Blend with:")
        for index, ratio in chosen:
            print(f"  TEST row {candidate_rows[index]} at {ratio:.0%}")
    else:
        print("No blend improved the primary wine.")
    print(f"  my wine at {primary_share:.0%}")
    print()
    print("Blended properties:")
    for name, value in blend.items():
        print(f"  {name}: {value:.5g}")
    print()
    print(f"Blend score: {score:.3f}")


if __name__ == "__main__":
    model = winequality.build_model()

    # Read every TEST wine as a dict; its position here is its TEST row number.
    records = pd.read_csv(TEST_FILE, sep=winequality.SEP).to_dict("records")

    # Start from the first quality-5 wine; the rest become candidates.
    primary_row = next(i for i, wine in enumerate(records)
                       if wine[winequality.TARGET] == PRIMARY_QUAL)
    primary = drop_quality(records[primary_row])
    candidates = [drop_quality(wine) for i, wine in enumerate(records)
                  if i != primary_row]
    candidate_rows = [i for i in range(len(records)) if i != primary_row]

    blend, score, chosen, primary_share = choose_candidates(primary,
                                                             candidates, model)
    base_score = winequality.predict(model, primary, rounded=False)
    print_result(primary_row, base_score, blend, score,
                 chosen, candidate_rows, primary_share)
