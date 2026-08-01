# Limitations, risks and next steps

## 1. How leakage is prevented

The single hardest requirement in this brief is that a prediction made at, say,
30% of the way through a course must not use anything that would not have been
known at that moment. This is enforced structurally rather than by convention:

- **One cutoff function.** `src/data/cutoff.py::apply_checkpoint_cutoff()` is the
  only path through which event data reaches feature engineering. Training and
  inference both call it, so they cannot drift apart.
- **Assessments are filtered on due date, not submission date.** A student who
  submits an assessment early does not make that assessment's score available to
  the model, because at real inference time other students would not have reached
  it yet.
- **A deny-list of outcome-bearing columns.** `final_result`,
  `date_unregistration`, raw `score` and `date_submitted` are listed in
  `FORBIDDEN_FEATURE_COLUMNS` and asserted against in `src/data/pipeline.py`,
  which raises before training can start if any of them appear.
- **`date_unregistration` deserves specific mention.** It is the most seductive
  leak in OULAD: it correlates almost perfectly with the Withdrawn label because
  it *is* the withdrawal. It is excluded everywhere.
- **Five automated tests** (`tests/test_leakage.py`) verify all of the above
  against the raw CSVs, including an independent recomputation of assessment
  eligibility that does not reuse the cutoff function's own output — so a bug
  inside the cutoff function itself would still be caught.

## 2. Fairness findings

Recall and precision were measured per demographic group on the held-out
presentation (`reports/fairness_report.csv`). Recall spread between the best- and
worst-served group:

| Attribute | Recall range | Gap |
|---|---|---|
| Disability | 0.778 – 0.810 | 0.032 |
| Gender | 0.763 – 0.798 | 0.035 |
| Age band | 0.750 – 0.841 | 0.091 |
| **Region** | **0.667 – 0.905** | **0.238** |

**The region gap is the finding that matters.** A 24-point recall spread means
that on this data, whether a struggling student gets noticed depends noticeably
on which region they are in. Some of that is small-sample noise — the region
groups are 41–82 students each on the test set — but it is too large to dismiss,
and it is exactly the kind of disparity that would quietly entrench itself if the
system were deployed unexamined.

No mitigation has been applied, per the brief's scope. If this moved toward real
use, the honest options are: drop region from the feature set and re-measure;
apply per-group threshold calibration; or set a recall floor as an explicit
constraint during model selection. All three trade something away, and that
trade-off is a decision for the institution, not for the model.

**Explanations are worded from feature values, not SHAP signs.** SHAP tells you
how far a feature moved a prediction away from the model's baseline; it does not
tell you whether the underlying behaviour is good or bad, and the two can point
in opposite directions. An early version of this code took the wording from the
SHAP sign and consequently told advisors that a student whose engagement trend
was clearly negative had "engagement picking up recently". A single visible
contradiction like that would reasonably cost the advisor's trust in every other
factor on the page. SHAP now decides which factors appear and in what order; the
feature's actual value decides what is said about it, compared against
training-set medians stored in the model metadata. Two regression tests in
`tests/test_api.py` hold this in place.

**Demographics are excluded from explanations.** The model uses demographic
features, but `src/models/explain.py` restricts advisor-facing "top factors" to
behavioural and academic signals only. Telling an advisor that a student's region
raised their risk score invites profiling in the human intervention that follows,
and is not actionable. Telling them "no assessment submitted in the first three
weeks" is both accurate and something a person can act on.

## 3. Data limitations

- **The shipped data is synthetic** (see `DATASET.md`). Metrics demonstrate that
  the pipeline works; they are not a forecast of real-world accuracy.
- **OULAD itself is narrow.** It covers one UK distance-learning institution,
  2013–2014, 7 modules. Engagement patterns in a Moodle-style VLE at a distance
  university will not transfer cleanly to an in-person institution, a different
  LMS, or a 2026 cohort.
- **Withdrawal and failure are merged into one label.** They are different events
  with different interventions — a student quietly disengaging needs a different
  conversation than one who is present but struggling. Merging them was a
  deliberate simplification and is the first thing worth revisiting.
- **Missing demographic values** are present in real OULAD (`imd_band` in
  particular) and are handled by one-hot encoding without a separate missingness
  indicator.

## 4. Model risks and assumptions

- **The checkpoint is a fraction of course length, not a pedagogically chosen
  moment.** 30% of a 268-day module falls in a very different place than 30% of a
  234-day one, in terms of how many assessments have actually happened. Modules
  with front-loaded or back-loaded assessment schedules will behave differently.
- **Early quiet is not the same as early risk.** Some students start slowly and
  finish well. The model has no way to distinguish "hasn't started yet" from
  "won't start", and this is the main source of its false positives.
- **Calibration was checked, not guaranteed.** Risk bands are cut at fixed
  thresholds (0.33 / 0.66). The reliability curve is in
  `reports/calibration.png`, but on a new cohort the bands should be recalibrated
  rather than assumed to still mean the same thing.
- **The multi-checkpoint results are close together** (recall 0.786 / 0.781 /
  0.813 at 20% / 30% / 50%). Earlier prediction costs surprisingly little
  accuracy here — but on the real data this curve is likely to be steeper, and it
  is the curve, not a single number, that should drive the choice of checkpoint.
- **Cohort drift.** The held-out evaluation uses the most recent presentation
  specifically to test generalisation across cohorts, which is the realistic
  deployment scenario. Performance would still need re-checking each term.

## 5. Ethical and operational concerns

- **An early-warning score is not a verdict.** It is a prompt for a human to look
  more closely. The interface language ("Needs a check-in", not "Will fail") is
  deliberate, and any deployment should keep the model advisory.
- **Self-fulfilling prophecy risk.** If flagged students are treated differently
  in ways that harm them — lowered expectations, extra scrutiny — the model's
  predictions become causal rather than predictive. This is a real, documented
  risk in learning analytics deployments.
- **Consent and governance are out of scope here.** OULAD is public and
  anonymised. Running this on live institutional student data would require
  institutional review, a lawful basis under GDPR/FERPA-equivalent regulation, a
  student-facing transparency notice, and a route for students to contest a score.
- **Intervention capacity is finite.** A recall-first threshold produces more
  flags than a precision-first one. If staff can only reach 20 students a week,
  the threshold should be set to match that capacity, not to maximise a metric.

## 6. Recommended next steps

1. Re-run the whole pipeline against the real OULAD CSVs and re-measure everything.
2. Separate withdrawal from failure into a multi-class or two-model formulation.
3. Investigate and mitigate the regional recall gap before any pilot.
4. Add per-module models or module as an explicit interaction — module structures
   differ enough that one global model is likely leaving accuracy on the table.
5. Tune the decision threshold against the support team's actual weekly capacity.
6. Track prediction-to-outcome drift each presentation, and retrain on a schedule
   rather than once.
