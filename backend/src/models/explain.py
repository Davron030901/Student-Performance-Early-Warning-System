"""
Explainability for the early-warning model.

Design decision (documented in LIMITATIONS.md): demographic features
(gender, region, disability, age_band, imd_band, highest_education) are used
by the model and are audited separately for fairness (see fairness.py), but
they are deliberately EXCLUDED from the human-readable "top factors" shown to
advisors. Telling an advisor "region: London" is a risk factor invites
demographic profiling in a human intervention and isn't actionable; telling
them "no assessment submitted in the first three weeks" is both accurate and
something a human can act on. The model itself may still legally use
demographics; the explanation surface just doesn't foreground them.
"""
import numpy as np
import pandas as pd
import shap
# matplotlib is intentionally NOT imported at module level. This module is
# imported by src/api/main.py (for top_factors_for_student, used on every
# prediction), and the deployed API installs requirements-api.txt, which
# excludes matplotlib on purpose — it's only needed for the training-time
# report plot below. A top-level `import matplotlib` here would make the API
# fail at startup in any environment that installed the runtime-only
# requirements, which is exactly the environment it actually deploys into.
# This was caught by testing against a venv built strictly from
# requirements-api.txt rather than the ambient dev environment.

# Only these engineered features are eligible to appear in a per-student explanation.
BEHAVIORAL_FEATURES = {
    "vle_total_clicks", "vle_active_days", "vle_distinct_sites", "vle_click_trend",
    "vle_days_since_last_click", "n_submitted", "avg_early_score", "pct_on_time",
    "avg_days_early", "late_registration",
}


def _phrase(feature: str, value: float, reference: dict[str, float] | None = None) -> tuple[str, bool] | None:
    """
    Picks the wording from the feature's ACTUAL VALUE, not from the sign of its
    SHAP contribution.

    Why: SHAP tells you how much a feature moved this prediction relative to the
    model's baseline, which is not the same as whether the underlying behaviour
    is good or bad. A student whose engagement is falling can still have that
    feature register a small negative contribution, and if the wording followed
    the SHAP sign we would tell an advisor "engagement has been picking up" about
    a student whose trend is clearly downward. That single contradiction would
    (rightly) cost us the advisor's trust in every other factor on the page.

    So: SHAP decides WHICH factors are worth showing and in what order; the value
    decides WHAT WE SAY about them. `reference` carries training-set medians for
    the features that only mean something relative to the cohort.
    """
    ref = reference or {}
    templates = {
        "vle_total_clicks": ("Low engagement with course materials so far",
                              "Strong engagement with course materials so far"),
        "vle_active_days": ("Few active days on the course site so far",
                             "Consistent activity on the course site"),
        "vle_distinct_sites": ("Interacting with very few course resources",
                                "Exploring a wide range of course resources"),
        "vle_click_trend": ("Engagement has been dropping off recently",
                             "Engagement has been holding up recently"),
        "vle_days_since_last_click": ("Has not logged into the course site recently",
                                       "Recently active on the course site"),
        # The reassuring wording deliberately does NOT claim completeness: this
        # function sees only the submission COUNT, never how many were due, so
        # "Submitted the early assessments" was overclaiming — a student who
        # submitted 1 of 2 was told they had submitted them.
        "n_submitted": ("Missed one or more early assessments",
                         "Has submitted early assessment work"),
        "avg_early_score": ("Early assessment scores are below average",
                             "Early assessment scores are above average"),
        "pct_on_time": ("A pattern of late submissions",
                         "A pattern of on-time submissions"),
        "avg_days_early": ("Submissions have been cutting it close to deadlines",
                            "Submissions have been comfortably ahead of deadlines"),
        "late_registration": ("Registered for the course later than most students",
                               "Registered for the course on time"),
    }
    risky_phrase, safe_phrase = templates.get(feature, (feature, feature))

    # Special cases where the value has a plain, non-relative meaning.
    if feature == "n_submitted":
        return (risky_phrase, True) if value <= 0 else (safe_phrase, False)
    if feature == "avg_early_score":
        if value < 0:
            return ("No assessment submitted yet, so there is no early score", True)
        return (risky_phrase, True) if value < 50 else (safe_phrase, False)
    if feature == "vle_click_trend":
        return (risky_phrase, True) if value < 0 else (safe_phrase, False)
    if feature == "vle_days_since_last_click":
        return (risky_phrase, True) if value > 10 else (safe_phrase, False)
    if feature == "pct_on_time":
        return (risky_phrase, True) if value < 0.8 else (safe_phrase, False)
    if feature == "late_registration":
        return (risky_phrase, True) if value >= 1 else (safe_phrase, False)
    if feature == "avg_days_early":
        return (risky_phrase, True) if value < 1 else (safe_phrase, False)

    # Everything else is only meaningful against the cohort, so compare to the
    # training-set median. With no reference available, stay silent rather than
    # guess a direction we cannot support.
    median = ref.get(feature)
    if median is None:
        return None
    return (risky_phrase, True) if value < median else (safe_phrase, False)


def compute_shap_values(model, X_transformed: pd.DataFrame):
    """model is a fitted tree model (XGBoost/RandomForest); X_transformed is
    the fully-numeric feature matrix (after dummy-encoding) it was trained on."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):  # some sklearn RF versions return a per-class list
        shap_values = shap_values[1]
    return shap_values, explainer


def save_global_importance_plot(shap_values: np.ndarray, X_transformed: pd.DataFrame, out_path: str):
    """Training/reporting only — never called from the API. Imported lazily so
    this module stays importable in the runtime-only deployment environment,
    which does not install matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    shap.summary_plot(shap_values, X_transformed, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _base_feature_name(dummy_col: str, base_features: list[str]) -> str:
    """Maps a one-hot dummy column (e.g. 'region_London Region') back to its
    original feature name ('region'), so demographic dummies can be grouped
    and filtered together."""
    for base in base_features:
        if dummy_col == base or dummy_col.startswith(base + "_"):
            return base
    return dummy_col


def top_factors_with_impact(
    shap_row: np.ndarray,
    feature_row: pd.Series,
    all_feature_columns: list[str],
    base_feature_names: list[str],
    top_n: int = 4,
    reference: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Returns (phrase, signed SHAP impact) pairs — the phrase and the number
    that produced it, together.

    This exists because pairing them after the fact is easy to get wrong and
    hard to notice: an earlier version of the cohort builder ranked the
    impacts separately and zipped them onto the phrases, which silently
    attached ~31% of phrases to a SHAP value from an unrelated feature. The
    dashboard draws its up/down direction bars from that number, so a third of
    them pointed the wrong way while the text beside them read correctly.
    Returning both from one place makes that class of bug impossible.
    """
    contributions = []
    for col, sv in zip(all_feature_columns, shap_row):
        base = _base_feature_name(col, base_feature_names)
        if base not in BEHAVIORAL_FEATURES:
            continue
        contributions.append((base, col, float(sv)))

    best_per_base = {}
    for base, col, sv in contributions:
        if base not in best_per_base or abs(sv) > abs(best_per_base[base][1]):
            best_per_base[base] = (col, sv)

    ranked = sorted(best_per_base.items(), key=lambda kv: abs(kv[1][1]), reverse=True)

    scored: list[tuple[str, float]] = []
    for base, (col, shap_value) in ranked:
        value = feature_row.get(col, np.nan)
        if pd.isna(value):
            continue
        phrased = _phrase(base, float(value), reference)
        if phrased:
            phrase, raises_risk = phrased
            # Magnitude comes from SHAP (how much this mattered); the SIGN comes
            # from the phrase's own direction. Those two genuinely disagree
            # sometimes — SHAP measures displacement from the model's baseline,
            # not whether the behaviour is good or bad — and the dashboard draws
            # its up/down arrow and colour from this number. Signing it by SHAP
            # instead produced a blue "Lowers the score" arrow next to "Low
            # engagement with course materials" on about a quarter of factors.
            # Same principle as _phrase: the value decides what we claim.
            impact = abs(float(shap_value)) * (1.0 if raises_risk else -1.0)
            scored.append((phrase, impact))

    results = scored[:top_n]

    # Ranking purely by magnitude can fill every slot with protective factors
    # for a student who is nonetheless flagged — leaving an advisor reading
    # "Strong engagement / Recently active" beside a 74% risk score, with no
    # indication of what the concern actually is. That explanation is true but
    # useless: it says why the student isn't worse, not why they're on the list.
    # So if there is a genuine concern anywhere in the factors, guarantee it a
    # slot by displacing the weakest protective one. Nothing is invented here —
    # the factor was already computed and simply lost the magnitude contest.
    if results and not any(impact > 0 for _phrase_text, impact in results):
        strongest_concern = next(((p, i) for p, i in scored if i > 0), None)
        if strongest_concern is not None:
            results = results[:-1] + [strongest_concern]

    return results


def top_factors_for_student(
    shap_row: np.ndarray,
    feature_row: pd.Series,
    all_feature_columns: list[str],
    base_feature_names: list[str],
    top_n: int = 4,
    reference: dict[str, float] | None = None,
) -> list[str]:
    """Phrases only. SHAP magnitude decides which factors appear and in what
    order; the feature's own value decides how each one is worded (see _phrase
    for why those are deliberately separated)."""
    return [
        phrase for phrase, _impact in top_factors_with_impact(
            shap_row, feature_row, all_feature_columns, base_feature_names, top_n, reference
        )
    ]


def behavioural_reference_medians(X_encoded: pd.DataFrame, base_feature_names: list[str]) -> dict[str, float]:
    """Training-set medians for the behavioural features, saved into metadata so
    inference can say 'below average' with something to compare against."""
    ref = {}
    for col in X_encoded.columns:
        base = _base_feature_name(col, base_feature_names)
        if base in BEHAVIORAL_FEATURES and base == col:
            ref[base] = float(X_encoded[col].median())
    return ref
