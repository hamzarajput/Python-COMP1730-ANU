"""
task2_theoreticalwine.py

Task 2 - Make a 'perfect' wine.

Invents theoretical wines at the extremes of the quality model:

Implements two required functions (but your solution may have
as many others as are necessary):
perfect()
    - a theoretical wine with as high a predicted quality as
    can be reached with the input model provided
awful()
    -  the converse: the worst predicted quality possible with the
    input model provided

Use both use winequality.build_model() and winequality.predict()
without changes.  Although additional helper functions may be
added to the winequality module.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import winequality

TSNE_FILE = "task2_tsne.png"

STEPS = 50       # how many values to try across each property's range
N_STARTS = 10    # how many training wines to start the search from

# Quality bands for the tSNE plot (same as Task 1): low = 3-4, medium = 5-6,
# high = 7-8, so the rare low and high wines stand out from the common ones.
LOW_MAX = 4
MEDIUM_MAX = 6


def property_ranges(model, train_file=winequality.TRAIN_FILE):
    """
    Find the smallest and largest value of each property in the training file.

    model      - a fitted model; its feature_names_in_ gives the property
                 names, in the order winequality.predict() expects
    train_file - the training csv to read the ranges from

    Returns a dict mapping each property name to (minimum, maximum).
    """
    names = list(model.feature_names_in_)
    train = pd.read_csv(train_file, sep=winequality.SEP)
    properties = train[names]
    return {name: (properties[name].min(), properties[name].max())
            for name in names}


def _search(model, start_wine, ranges, maximise):
    """
    Improve one wine by coordinate ascent.

    Sweep each property across its whole range, keep the single value that
    best improves the predicted score, and repeat until a full pass over all
    properties makes no improvement (a local optimum).

    start_wine - dict of starting property values (a real training wine)
    ranges     - dict name -> (min, max), from property_ranges()
    maximise   - True to push the score up (perfect), False to push it down

    Returns (wine, score): the improved wine dict and its batch-predicted score.
    """
    names = list(model.feature_names_in_)
    wine = dict(start_wine)
    score = float(model.predict(pd.DataFrame([wine], columns=names))[0])

    improved = True
    while improved:
        improved = False
        for name in names:
            low, high = ranges[name]
            candidates = np.linspace(low, high, STEPS)
            # One candidate wine per grid value, changing only this property.
            trial = pd.DataFrame([wine] * STEPS, columns=names)
            trial[name] = candidates
            scores = model.predict(trial)      # score all at once, for speed
            best = scores.argmax() if maximise else scores.argmin()
            better = (scores[best] > score if maximise
                      else scores[best] < score)
            if better:
                wine[name] = float(candidates[best])
                score = float(scores[best])
                improved = True
    return wine, score


def _best_wine(model, maximise):
    """
    Search from the 10 best- (or worst-) predicted training wines and return
    the single best resulting wine dict.

    maximise - True for perfect() (start from the 10 best, push the score up),
               False for awful()  (start from the 10 worst, push it down)
    """
    names = list(model.feature_names_in_)
    ranges = property_ranges(model)

    train = pd.read_csv(winequality.TRAIN_FILE, sep=winequality.SEP)
    properties = train[names]
    scores = model.predict(properties)         # score every training wine

    # Take the 10 best- (or worst-) predicted wines as starting points.
    order = np.argsort(scores)
    starts = order[-N_STARTS:] if maximise else order[:N_STARTS]

    best_wine = None
    best_score = None
    for i in starts:
        start_wine = properties.iloc[i].to_dict()
        wine, score = _search(model, start_wine, ranges, maximise)
        if best_score is None or (score > best_score if maximise
                                  else score < best_score):
            best_wine, best_score = wine, score
    return best_wine


def perfect(model=None):
    """Invent a theoretical wine with a very high predicted quality score.

    model: model from winequality.build_model()

    Returns (wine, score): the physicochemical properties as a dict keyed for
    winequality.predict(), and the quality score it predicts.
    """
    if model is None:
        model = winequality.build_model()
    wine = _best_wine(model, maximise=True)
    score = winequality.predict(model, wine)
    return wine, score


def awful(model=None):
    """Invent a theoretical wine with the worst predicted quality score.

    Returns (wine, score), as perfect() does.
    """
    if model is None:
        model = winequality.build_model()
    wine = _best_wine(model, maximise=False)
    score = winequality.predict(model, wine)
    return wine, score


def plot_on_tsne(perfect_wine, awful_wine, model=None, perplexity=60,
                 output_file=TSNE_FILE):
    """
    Rerun the Task 1 tSNE with the two invented wines added and mark them.

    perfect_wine, awful_wine - property dicts, as returned by perfect()/awful()
    model       - fitted model, used only for the property order; built if None
    perplexity  - sklearn.manifold.TSNE parameter
    output_file - where to save the picture

    The invented wines are embedded alongside the real wines and drawn on top
    as large labelled stars.  The two wines are passed in, never recomputed.
    """
    if model is None:
        model = winequality.build_model()
    names = list(model.feature_names_in_)

    data = winequality.load_data(winequality.DATA_FILE)
    properties = data[:, :-1]
    quality = data[:, -1]

    # Add the two invented wines as extra rows, in the model's column order,
    # before scaling so they share the real wines' embedding space.
    extra = np.array([[perfect_wine[n] for n in names],
                      [awful_wine[n] for n in names]])
    all_properties = np.vstack([properties, extra])

    scaled = StandardScaler().fit_transform(all_properties)
    embedding = TSNE(perplexity=perplexity,
                     random_state=0).fit_transform(scaled)

    real = embedding[:-2]
    perfect_point = embedding[-2]
    awful_point = embedding[-1]

    # Three quality groups, as in Task 1.
    low = quality <= LOW_MAX
    medium = (quality > LOW_MAX) & (quality <= MEDIUM_MAX)
    high = quality > MEDIUM_MAX

    fig, ax = plt.subplots(figsize=(8, 8))
    # Common medium wines first as a grey backdrop, then the rare groups.
    ax.scatter(real[medium, 0], real[medium, 1], s=10, alpha=0.4,
               color="lightgrey", label=f"medium (5-6): {int(medium.sum())}")
    ax.scatter(real[low, 0], real[low, 1], s=10, alpha=0.7, color="crimson",
               marker="v", label=f"low (3-4): {int(low.sum())}")
    ax.scatter(real[high, 0], real[high, 1], s=10, alpha=0.7, color="royalblue",
               marker="^", label=f"high (7-8): {int(high.sum())}")

    # The two invented wines, as large labelled stars on top.
    ax.scatter(*perfect_point, s=300, color="gold", marker="*",
               edgecolor="black", zorder=5, label="perfect")
    ax.scatter(*awful_point, s=300, color="black", marker="*",
               edgecolor="white", zorder=5, label="awful")
    ax.annotate("perfect", perfect_point, textcoords="offset points",
                xytext=(8, 8))
    ax.annotate("awful", awful_point, textcoords="offset points",
                xytext=(8, 8))

    ax.set_xlabel("tSNE 1")
    ax.set_ylabel("tSNE 2")
    ax.set_title(f"tSNE of wine properties with invented wines "
                 f"(perplexity={perplexity})")
    ax.legend()

    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    model = winequality.build_model()
    perfect_wine, perfect_score = perfect(model)
    awful_wine, awful_score = awful(model)
    print("perfect:", perfect_wine, perfect_score)
    print("awful:", awful_wine, awful_score)
    plot_on_tsne(perfect_wine, awful_wine, model)
