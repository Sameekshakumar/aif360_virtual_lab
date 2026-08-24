import { useState } from "react"
import axios from "axios"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, LineChart, Line, ReferenceLine
} from "recharts"

const NAVY   = "#1a1f3c"
const NAVY2  = "#252b4a"
const BG     = "#f0f2f8"
const WHITE  = "#ffffff"
const CORAL  = "#e8836a"
const CORAL2 = "#FCAF98"
const BLUE   = "#8b9fd4"
const MUTED  = "#8a8fa8"
const TEXT   = "#1a1f3c"
const BORDER = "#bfcefb"

const DATASETS = ["adult", "german", "compas"]
const ATTRS    = { adult: ["sex","race"], german: ["sex","age"], compas: ["sex","race"] }

const DATASET_INFO = {
  adult:  { full: "Adult Census",      desc: "US census income data predicting whether income exceeds $50K/year.", rows: "48,842", attrs: "14", label: "Income >$50K", source: "UCI Machine Learning Repository", sourceUrl: "https://archive.ics.uci.edu/ml/datasets/adult" },
  german: { full: "German Credit",     desc: "German bank credit risk data classifying applicants as good or bad risk.", rows: "1,000",  attrs: "20", label: "Good Credit", source: "UCI Machine Learning Repository", sourceUrl: "https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)" },
  compas: { full: "COMPAS Recidivism", desc: "Criminal recidivism predictions by Northpointe used in US courts.", rows: "7,214",  attrs: "7",  label: "No Recidivism", source: "ProPublica · compas-analysis", sourceUrl: "https://github.com/propublica/compas-analysis" },
}

const PROTECTED_LABELS = {
  sex:  { privileged: "Male",      unprivileged: "Female" },
  race: { privileged: "White",     unprivileged: "Non-white" },
  age:  { privileged: "Age ≥ 25",  unprivileged: "Age < 25" },
}

const DATASET_CONTEXT = {
  adult:  "income above $50K",
  german: "good credit rating",
  compas: "no recidivism within two years",
}

const ALGORITHMS = [
  { label: "Reweighing",               sub: "Pre-processing", endpoint: "/run-reweighing",               beforeKey: "before_reweighing", afterKey: "after_reweighing", type: "bar",  icon: "/scale.png" },
  { label: "Disparate Impact Remover", sub: "Pre-processing", endpoint: "/run-disparate-impact-remover",  beforeKey: null,               afterKey: null,              type: "line", icon: "/customer-support.png" },
  { label: "Adversarial Debiasing",    sub: "In-processing",  endpoint: "/run-adversarial-debiasing",     beforeKey: "before_debiasing", afterKey: "after_debiasing", type: "bar",  icon: "/brainstorm.png" },
  { label: "Exponentiated Gradient",   sub: "In-processing",  endpoint: "/run-exponentiated-gradient",    beforeKey: "before_egr",       afterKey: "after_egr",       type: "bar",  icon: "/bar-chart.png" },
]

const METRIC_LABELS = {
  balanced_accuracy:       "Balanced Accuracy",
  disparate_impact:        "Disparate Impact",
  avg_odds_difference:     "Avg Odds Difference",
  equal_opp_difference:    "Equal Opp Difference",
  statistical_parity_diff: "Statistical Parity Diff",
}

const ALGO_INFO = [
  { desc: "Assigns different weights to training examples to reduce discrimination before model training. Upweights underrepresented group-label combinations so the model sees a more balanced view of the data.", idea: "If women with high income are underrepresented, their records get higher weights during training." },
  { desc: "Edits feature values to increase group fairness while preserving rank-ordering within groups. The repair_level parameter controls how aggressively features are modified.", idea: "At repair_level=1.0, feature distributions become identical across groups. At 0.0, no change." },
  { desc: "Simultaneously trains a classifier to predict labels and an adversary to predict the protected attribute. The classifier is penalized when the adversary succeeds, forcing it to make predictions that don't encode group information.", idea: "Like a GAN — the predictor and adversary compete, resulting in a fairer classifier." },
  { desc: "Reduces fairness-constrained learning to a sequence of cost-sensitive classification problems. Uses the Equalized Odds constraint to ensure similar true positive and false positive rates across groups.", idea: "Iteratively adjusts sample weights until the fairness constraint is satisfied within a tolerance." },
]

const PAGES      = ["Introduction", "Glossary", "Datasets", "Algorithms", "Lab", "Security"]
const PAGE_ICONS = { Introduction: "/presentation.png", Glossary: "/dictionary.png", Datasets: "/folder.png", Algorithms: "/engineering.png", Lab: "/microscope.png", Security: "/cyber-security.png" }

const METRICS_INFO = [
  {
    key: "disparate_impact",
    name: "Disparate Impact",
    formula: "P(Ŷ=1 | unprivileged) / P(Ŷ=1 | privileged)",
    ideal: "1.0 (equal rates). Legal threshold: ≥ 0.8 (the '80% rule')",
    interpret: "Measures the ratio of favorable outcome rates between unprivileged and privileged groups. A value of 0.5 means the unprivileged group is half as likely to receive a favorable outcome.",
    range: "0 to ∞. Values below 0.8 or above 1.25 indicate potential discrimination.",
  },
  {
    key: "statistical_parity_diff",
    name: "Statistical Parity Difference",
    formula: "P(Ŷ=1 | unprivileged) − P(Ŷ=1 | privileged)",
    ideal: "0.0 (no difference between groups)",
    interpret: "The raw difference in favorable outcome rates. A value of -0.2 means the unprivileged group receives favorable outcomes 20 percentage points less often.",
    range: "-1 to 1. Values close to 0 indicate fairness. Negative = unprivileged group disadvantaged.",
  },
  {
    key: "avg_odds_difference",
    name: "Average Odds Difference",
    formula: "½ × [(FPR_unpriv − FPR_priv) + (TPR_unpriv − TPR_priv)]",
    ideal: "0.0 (equal true and false positive rates across groups)",
    interpret: "Averages the gap in true positive rates and false positive rates between groups. Captures both whether qualified people get approved equally AND whether unqualified people get rejected equally.",
    range: "-1 to 1. Negative = unprivileged group has lower TPR and/or higher FPR.",
  },
  {
    key: "equal_opp_difference",
    name: "Equal Opportunity Difference",
    formula: "TPR_unprivileged − TPR_privileged",
    ideal: "0.0 (equal true positive rates)",
    interpret: "Focuses only on true positive rates — are qualified people from both groups equally likely to be correctly identified? Useful when false negatives are the primary concern (e.g. denying a loan to someone who would repay).",
    range: "-1 to 1. Negative = qualified unprivileged individuals are less likely to get favorable outcomes.",
  },
  {
    key: "balanced_accuracy",
    name: "Balanced Accuracy",
    formula: "½ × (TPR + TNR)",
    ideal: "1.0 (perfect). Practically, ≥ 0.70 is considered acceptable.",
    interpret: "Average of sensitivity (true positive rate) and specificity (true negative rate). Used instead of regular accuracy because it accounts for class imbalance — important since favorable outcomes are often the minority class.",
    range: "0.5 (random guessing) to 1.0 (perfect). Drops slightly after mitigation — this is the typical fairness-accuracy tradeoff.",
  },
]

// ── Security experiment data (real results) ────────────────────────────────
const POISON_SEX = [
  { pct: "0%",  accuracy: 0.6961, disparate_impact: 0.1823 },
  { pct: "2%",  accuracy: 0.6880, disparate_impact: 0.2009 },
  { pct: "5%",  accuracy: 0.6788, disparate_impact: 0.2604 },
  { pct: "10%", accuracy: 0.6636, disparate_impact: 0.3266 },
  { pct: "15%", accuracy: 0.6520, disparate_impact: 0.4443 },
  { pct: "20%", accuracy: 0.6405, disparate_impact: 0.6037 },
]
const POISON_RACE = [
  { pct: "0%",  accuracy: 0.6970, disparate_impact: 0.4404 },
  { pct: "2%",  accuracy: 0.6922, disparate_impact: 0.4995 },
  { pct: "5%",  accuracy: 0.6861, disparate_impact: 0.6128 },
  { pct: "10%", accuracy: 0.6805, disparate_impact: 0.8384 },
  { pct: "15%", accuracy: 0.6750, disparate_impact: 1.0044 },
  { pct: "20%", accuracy: 0.6709, disparate_impact: 1.2312 },
]

// ── Dynamic explanation generator ────────────────────────────────────────────
function generateExplanation(result, algoIdx, dataset, attr) {
  const algo      = ALGORITHMS[algoIdx]
  const labels    = PROTECTED_LABELS[attr] || { privileged: "privileged group", unprivileged: "unprivileged group" }
  const context   = DATASET_CONTEXT[dataset] || "a favorable outcome"
  const unpriv    = labels.unprivileged
  const priv      = labels.privileged

  if (algo.type === "line") {
    const levels    = result.repair_levels
    const firstPass = levels.find(r => r.disparate_impact >= 0.8)
    const last      = levels[levels.length - 1]
    const lines = []
    lines.push(`At repair level 0 (no repair), the disparate impact is ${levels[0].disparate_impact} — meaning ${unpriv} individuals are only ${Math.round(levels[0].disparate_impact * 100)}% as likely as ${priv} individuals to receive ${context}.`)
    if (firstPass) {
      lines.push(`The model crosses the legal fairness threshold (0.8) at repair level ${firstPass.repair_level}, achieving a disparate impact of ${firstPass.disparate_impact}. This means feature values need to be modified at least ${Math.round(firstPass.repair_level * 100)}% of the way toward equal distributions before the model becomes legally fair.`)
    } else {
      lines.push(`The model does not cross the 0.8 legal threshold even at full repair (level 1.0), reaching only ${last.disparate_impact}. This suggests the bias in this dataset is deeply encoded in the feature structure and cannot be fully corrected by feature repair alone.`)
    }
    lines.push(`At full repair (level 1.0), disparate impact is ${last.disparate_impact}. ${last.disparate_impact >= 0.8 ? "The model is fair by the 80% rule at this level." : "Even full repair is insufficient — consider combining with an in-processing method."}`)
    return lines
  }

  const bKey  = algo.beforeKey
  const aKey  = algo.afterKey
  const before = result[bKey]
  const after  = result[aKey]
  const lines = []

  // Dataset bias
  const mdb = result.dataset_bias?.mean_diff_before
  const mda = result.dataset_bias?.mean_diff_after
  if (mdb !== undefined) {
    lines.push(`In the raw dataset, ${unpriv} individuals are ${Math.abs(Math.round(mdb * 100))}% less likely to receive ${context} than ${priv} individuals (mean difference: ${mdb}). After applying ${algo.label}, this gap ${Math.abs(mda) < Math.abs(mdb) ? `reduced to ${mda}` : `changed to ${mda}`}.`)
  }

  // Disparate impact
  const diB = before?.disparate_impact, diA = after?.disparate_impact
  if (diB !== undefined) {
    const improved = diA > diB
    const passedThreshold = diA >= 0.8 && diB < 0.8
    lines.push(`Disparate impact improved from ${diB} → ${diA} (${improved ? "+" : ""}${(diA - diB).toFixed(4)}). ${passedThreshold ? `This is a major win — the model now crosses the legal 0.8 threshold, meaning ${unpriv} individuals are at least 80% as likely to receive ${context} as ${priv} individuals.` : diA >= 0.8 ? `The model remains above the 0.8 legal threshold.` : `The model still falls below the 0.8 legal threshold (${diA} < 0.8) — bias has been reduced but not eliminated.`}`)
  }

  // Avg odds
  const aoB = before?.avg_odds_difference, aoA = after?.avg_odds_difference
  if (aoB !== undefined) {
    lines.push(`Average odds difference moved from ${aoB} → ${aoA}. ${Math.abs(aoA) < Math.abs(aoB) ? `This improvement means both the true positive rate gap and false positive rate gap between ${unpriv} and ${priv} groups have narrowed.` : `The odds difference increased slightly, indicating a tradeoff between different fairness criteria.`} A value close to 0 indicates equalized odds across groups.`)
  }

  // Accuracy tradeoff
  const accB = before?.balanced_accuracy, accA = after?.balanced_accuracy
  if (accB !== undefined) {
    const drop = (accA - accB).toFixed(4)
    lines.push(`Balanced accuracy changed from ${accB} → ${accA} (${drop > 0 ? "+" : ""}${drop}). ${accA >= accB ? "Accuracy was maintained or improved — an ideal outcome." : `A drop of ${Math.abs(drop)} is the expected fairness-accuracy tradeoff. This small accuracy cost is generally considered acceptable given the fairness gains achieved.`}`)
  }

  return lines
}

// ── Shared styles ─────────────────────────────────────────────────────────────
const card  = { background: WHITE, borderRadius: 16, padding: 28, boxShadow: "0 2px 12px rgba(26,31,60,0.07)", marginBottom: 20 }
const selSt = { width: "100%", padding: "10px 14px", borderRadius: 10, border: `1.5px solid ${BORDER}`, fontSize: 14, fontFamily: "'Inter',sans-serif", fontWeight: 500, color: TEXT, background: WHITE, outline: "none", cursor: "pointer" }
const lblSt = { display: "block", marginBottom: 6, fontWeight: 700, fontSize: 12, color: MUTED, letterSpacing: 0.8, textTransform: "uppercase" }
const thSt  = { padding: "12px 16px", textAlign: "center", fontWeight: 700, fontSize: 11, color: MUTED, letterSpacing: 0.8, borderBottom: `1px solid ${BORDER}`, textTransform: "uppercase" }
const tdSt  = { padding: "14px 16px", borderBottom: `1px solid ${BG}`, fontSize: 13, textAlign: "center" }

// ── Commented code snippets for each algorithm ────────────────────────────────
const ALGO_CODE = [
`# ── REWEIGHING (Pre-processing) ──────────────────────────────────────────────
# Goal: Make the training data fairer by adjusting how much each record
# "counts" during training, before the model ever sees it.

from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Step 1: Define which group is "privileged" and which is "unprivileged"
# Privileged = historically advantaged group (e.g. Male, White)
# Unprivileged = historically disadvantaged group (e.g. Female, Non-white)
privileged_groups   = [{"sex": 1}]   # Male = 1
unprivileged_groups = [{"sex": 0}]   # Female = 0

# Step 2: Measure bias in the RAW dataset before doing anything
# mean_difference = P(favorable | unprivileged) - P(favorable | privileged)
# A value of -0.19 means females are 19% less likely to earn >$50K
metric_before = BinaryLabelDatasetMetric(
    train,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)
print("Mean difference BEFORE reweighing:", metric_before.mean_difference())

# Step 3: Apply Reweighing
# This assigns a "weight" to each training record.
# Records from underrepresented (group, outcome) pairs get higher weights
# so the model pays more attention to them during training.
# Example: A high-income female record gets a higher weight than a
# high-income male record, because high-income females are underrepresented.
RW = Reweighing(
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)
RW.fit(train)                        # Learn the correct weights
train_transformed = RW.transform(train)  # Apply weights to training data

// Step 4: Verify the dataset is now more balanced
metric_after = BinaryLabelDatasetMetric(
    train_transformed,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)
print("Mean difference AFTER reweighing:", metric_after.mean_difference())
# Should be much closer to 0 (perfect equality)

# Step 5: Train a Logistic Regression on BOTH original and reweighed data
# We compare them to measure the effect of reweighing
scaler = StandardScaler()
X_train_orig = scaler.fit_transform(train.features)

# sample_weight passes the instance weights to the logistic regression
# This is how the reweighing actually affects the model
model_orig = LogisticRegression(max_iter=500)
model_orig.fit(
    X_train_orig,
    train.labels.ravel(),
    sample_weight=train.instance_weights       # original weights (all ~1.0)
)

model_fair = LogisticRegression(max_iter=500)
model_fair.fit(
    scaler.fit_transform(train_transformed.features),
    train_transformed.labels.ravel(),
    sample_weight=train_transformed.instance_weights  # reweighed weights
)

# Step 6: Find optimal classification threshold on validation set
# Default threshold is 0.5 (predict positive if probability > 0.5)
# But optimal fairness/accuracy may occur at a different threshold
# We scan 100 thresholds and pick the one with best balanced accuracy
best_threshold = 0.5
best_balanced_accuracy = 0
for t in np.linspace(0.01, 0.99, 100):
    predictions = (model_orig.predict_proba(X_val)[:, 1] > t).astype(int)
    # balanced_accuracy = 0.5 * (true_positive_rate + true_negative_rate)
    ba = 0.5 * (sum((predictions == 1) & (y_val == 1)) / sum(y_val == 1) +
                sum((predictions == 0) & (y_val == 0)) / sum(y_val == 0))
    if ba > best_balanced_accuracy:
        best_balanced_accuracy = ba
        best_threshold = t

# Step 7: Evaluate fairness metrics on the test set
cm = ClassificationMetric(
    test, predictions,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)
print("Disparate Impact:", cm.disparate_impact())
# Ideal: 1.0. Legal threshold: >= 0.8 (the '80% rule')
# disparate_impact = P(favorable|female) / P(favorable|male)

print("Average Odds Difference:", cm.average_odds_difference())
# Ideal: 0.0. Measures gap in both TPR and FPR between groups.

print("Equal Opportunity Difference:", cm.equal_opportunity_difference())
# Ideal: 0.0. Measures gap in TPR (true positive rate) only.

print("Statistical Parity Difference:", cm.statistical_parity_difference())
# Ideal: 0.0. Raw difference in favorable outcome rates.`,

`# ── DISPARATE IMPACT REMOVER (Pre-processing) ────────────────────────────────
# Goal: Edit the actual feature values in the dataset so that the
# distributions of features become more similar across groups.
# Unlike reweighing which changes weights, this changes the data itself.

from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Step 1: Scale features to [0, 1] range
# Needed because the repair algorithm works on feature distributions
scaler = MinMaxScaler()
dataset.features = scaler.fit_transform(dataset.features)

# Step 2: Find which column index corresponds to our protected attribute
# We need this to remove it from the features before training
# (the model should not directly use sex/race to make predictions)
protected_index = dataset.feature_names.index("sex")

# Step 3: Apply Disparate Impact Remover at different repair levels
# repair_level=0.0 → no change (original data)
# repair_level=1.0 → full repair (distributions become identical across groups)
# repair_level=0.5 → halfway between original and fully repaired
results = []
for level in np.linspace(0.0, 1.0, 11):   # test 11 levels: 0.0, 0.1, ..., 1.0

    di_remover = DisparateImpactRemover(
        repair_level=level,
        sensitive_attribute="sex"
    )

    # fit_transform: learn the repair transformation and apply it
    # The algorithm adjusts feature values so their distributions
    # become more similar between male and female groups
    repaired_data = di_remover.fit_transform(dataset)

    # Remove the protected attribute from features before training
    # A fair model should make predictions WITHOUT using sex directly
    X = np.delete(repaired_data.features, protected_index, axis=1)
    y = repaired_data.labels.ravel()

    # Train logistic regression on repaired data
    model = LogisticRegression(class_weight="balanced", solver="liblinear")
    model.fit(X, y)

    # Measure disparate impact at this repair level
    # If disparate impact >= 0.8, the model passes the legal fairness test
    predictions = repaired_data.copy(deepcopy=True)
    predictions.labels = model.predict(X).reshape(-1, 1)

    metric = BinaryLabelDatasetMetric(
        predictions,
        privileged_groups=[{"sex": 1}],
        unprivileged_groups=[{"sex": 0}]
    )
    di = metric.disparate_impact()
    print(f"Repair level {level:.1f} → Disparate Impact: {di:.4f} "
          f"{'✓ PASSES' if di >= 0.8 else '✗ FAILS'} 0.8 threshold")
    results.append({"repair_level": level, "disparate_impact": di})

# Key insight: Higher repair levels improve fairness (disparate impact → 1.0)
# but may reduce accuracy because we're modifying the actual data.
# The optimal repair level balances these two competing goals.`,

`# ── ADVERSARIAL DEBIASING (In-processing) ────────────────────────────────────
# Goal: Train a model that is good at predicting the label (income)
# while simultaneously being BAD at predicting the protected attribute (sex).
# This forces the model to make predictions that don't encode gender.

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()   # AIF360 uses TensorFlow 1.x style
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.metrics import ClassificationMetric
from sklearn.preprocessing import MaxAbsScaler

# Step 1: Scale features using MaxAbsScaler
# Scales each feature to [-1, 1] range — required for neural network stability
scaler = MaxAbsScaler()
train.features = scaler.fit_transform(train.features)
test.features  = scaler.transform(test.features)

# Step 2: Train a PLAIN classifier (no debiasing) as our baseline
# This is a standard neural network that only tries to predict income
# We use it to measure how biased an unmodified model is
tf.reset_default_graph()
session1 = tf.Session()

plain_classifier = AdversarialDebiasing(
    privileged_groups=[{"sex": 1}],
    unprivileged_groups=[{"sex": 0}],
    scope_name="plain_classifier",
    debias=False,        # ← No debiasing, just a regular classifier
    sess=session1
)
plain_classifier.fit(train)          # Train for 50 epochs
plain_predictions = plain_classifier.predict(test)
session1.close()

# Step 3: Train the ADVERSARIAL DEBIASING model
# Architecture: Two neural networks competing against each other
#   - Predictor network: tries to predict income from features
#   - Adversary network: tries to predict sex from the predictor's output
#
# During training:
#   - Predictor is rewarded for correctly predicting income
#   - Predictor is PENALIZED when the adversary successfully predicts sex
#   → The predictor learns to encode NO gender information in its predictions
#
# This is similar to a GAN (Generative Adversarial Network):
# the predictor and adversary are locked in a minimax game
tf.reset_default_graph()
session2 = tf.Session()

debiased_classifier = AdversarialDebiasing(
    privileged_groups=[{"sex": 1}],
    unprivileged_groups=[{"sex": 0}],
    scope_name="debiased_classifier",
    debias=True,         # ← Enable adversarial debiasing
    sess=session2
)
debiased_classifier.fit(train)       # Train (takes ~2-3 minutes)
debiased_predictions = debiased_classifier.predict(test)
session2.close()

# Step 4: Compare fairness metrics between plain and debiased models
for label, preds in [("Plain", plain_predictions), ("Debiased", debiased_predictions)]:
    cm = ClassificationMetric(
        test, preds,
        unprivileged_groups=[{"sex": 0}],
        privileged_groups=[{"sex": 1}]
    )
    print(f"\\n── {label} Classifier ──")
    print(f"  Balanced Accuracy:  {0.5*(cm.true_positive_rate()+cm.true_negative_rate()):.4f}")
    print(f"  Disparate Impact:   {cm.disparate_impact():.4f}  (want ≥ 0.8)")
    print(f"  Avg Odds Diff:      {cm.average_odds_difference():.4f}  (want ≈ 0.0)")
    print(f"  Equal Opp Diff:     {cm.equal_opportunity_difference():.4f}  (want ≈ 0.0)")
    print(f"  Stat Parity Diff:   {cm.statistical_parity_difference():.4f}  (want ≈ 0.0)")`,

`# ── EXPONENTIATED GRADIENT REDUCTION (In-processing) ─────────────────────────
# Goal: Find the fairest possible model by framing the fairness problem
# as a mathematical optimization problem with constraints.
# It repeatedly adjusts sample weights until the fairness constraint is met.

from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import (
    ExponentiatedGradientReduction
)
from aif360.metrics import ClassificationMetric
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler

# Step 1: Scale features
scaler = MaxAbsScaler()
train.features = scaler.fit_transform(train.features)
test.features  = scaler.transform(test.features)

# Step 2: Train a baseline logistic regression (no fairness constraint)
# This is our reference point — how biased is a standard model?
baseline_model = LogisticRegression(solver="lbfgs", max_iter=1000)
baseline_model.fit(
    train.features,
    train.labels.ravel(),
    sample_weight=train.instance_weights
)
baseline_predictions = test.copy(deepcopy=True)
baseline_predictions.labels = baseline_model.predict(test.features).reshape(-1, 1)

# Step 3: Apply Exponentiated Gradient Reduction
# The algorithm works as follows:
#   1. Start with equal weights for all training samples
#   2. Train a classifier (logistic regression here)
#   3. Check if the Equalized Odds constraint is satisfied:
#      → Are true positive rates equal across groups?
#      → Are false positive rates equal across groups?
#   4. If NOT satisfied: increase weights for samples where the constraint
#      is violated (using the "exponentiated gradient" update rule)
#   5. Repeat steps 2-4 until the constraint is satisfied or max iterations reached
#
# The base estimator (LogisticRegression) is trained many times with
# different sample weights until a fair solution is found.
base_estimator = LogisticRegression(solver="lbfgs", max_iter=1000)

egr = ExponentiatedGradientReduction(
    estimator=base_estimator,
    constraints="EqualizedOdds",  # Constraint: equal TPR and FPR across groups
    drop_prot_attr=False          # Keep protected attribute visible during training
)
egr.fit(train)
egr_predictions = egr.predict(test)

# Step 4: Compare baseline vs fairness-constrained model
for label, preds in [("Baseline", baseline_predictions), ("EGR Fair", egr_predictions)]:
    cm = ClassificationMetric(
        test, preds,
        unprivileged_groups=[{"sex": 0}],
        privileged_groups=[{"sex": 1}]
    )
    print(f"\\n── {label} ──")
    print(f"  Balanced Accuracy:  {0.5*(cm.true_positive_rate()+cm.true_negative_rate()):.4f}")
    print(f"  Disparate Impact:   {cm.disparate_impact():.4f}  (want ≥ 0.8)")
    print(f"  Avg Odds Diff:      {cm.average_odds_difference():.4f}  (want ≈ 0.0)")
    print(f"  Equal Opp Diff:     {cm.equal_opportunity_difference():.4f}  (want ≈ 0.0)")
    print(f"  Stat Parity Diff:   {cm.statistical_parity_difference():.4f}  (want ≈ 0.0)")

# Key insight: EGR trades off some accuracy to satisfy the fairness constraint.
# The Equalized Odds constraint ensures both groups are treated equally
# in terms of both correct approvals AND correct rejections.`
]

// ── DatasetExplorer component ─────────────────────────────────────────────────
function DatasetExplorer() {
  const [previews, setPreviews]   = useState({})
  const [loading, setLoading]     = useState({})
  const [expanded, setExpanded]   = useState({})

  const loadPreview = async (d) => {
    if (previews[d]) { setExpanded(e => ({ ...e, [d]: !e[d] })); return }
    setLoading(l => ({ ...l, [d]: true }))
    try {
      const res = await axios.get(`http://localhost:8000/dataset-preview/${d}`)
      setPreviews(p => ({ ...p, [d]: res.data }))
      setExpanded(e => ({ ...e, [d]: true }))
    } catch { }
    setLoading(l => ({ ...l, [d]: false }))
  }

  const downloadCSV = (d) => {
    const link = document.createElement("a")
    link.href = `http://localhost:8000/dataset-download/${d}`
    link.download = `${d}_dataset.csv`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {DATASETS.map(d => {
        const info = DATASET_INFO[d]
        const preview = previews[d]
        const isOpen = expanded[d]
        return (
          <div key={d} style={card}>
            {/* Header row */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
              <div>
                <div style={{ fontWeight: 800, color: TEXT, fontSize: 18 }}>{info.full}</div>
                <div style={{ fontSize: 11, fontWeight: 700, color: CORAL, marginTop: 2, letterSpacing: 1, textTransform: "uppercase" }}>{d} dataset</div>
              </div>
              <div style={{ background: `${CORAL2}55`, borderRadius: 8, padding: "4px 10px", fontSize: 12, fontWeight: 700, color: CORAL }}>{info.rows} rows</div>
            </div>
            <p style={{ color: MUTED, fontSize: 13, lineHeight: 1.6, marginBottom: 16 }}>{info.desc}</p>

            {/* Stats */}
            <div style={{ borderTop: `1px solid ${BORDER}`, paddingTop: 14, marginBottom: 16 }}>
              {[["Features", info.attrs, TEXT], ["Label", info.label, CORAL], ["Protected Attrs", ATTRS[d].join(", "), NAVY]].map(([l, v, c]) => (
                <div key={l} style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                  <span style={{ fontSize: 12, color: MUTED, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 }}>{l}</span>
                  <span style={{ fontSize: 12, fontWeight: 700, color: c }}>{v}</span>
                </div>
              ))}
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <span style={{ fontSize: 12, color: MUTED, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 }}>Source</span>
                <a href={info.sourceUrl} target="_blank" rel="noreferrer" style={{ fontSize: 12, fontWeight: 700, color: BLUE, textDecoration: "none" }}>{info.source}</a>
              </div>
            </div>

            {/* Action buttons */}
            <div style={{ display: "flex", gap: 10 }}>
              <button onClick={() => loadPreview(d)} style={{
                padding: "8px 18px", borderRadius: 8, border: `1.5px solid ${BORDER}`,
                background: isOpen ? NAVY : WHITE, color: isOpen ? WHITE : TEXT,
                fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: "'Inter',sans-serif",
              }}>
                {loading[d] ? "Loading..." : isOpen ? "Hide Preview" : "Preview Data"}
              </button>
              <button onClick={() => downloadCSV(d)} style={{
                padding: "8px 18px", borderRadius: 8, border: "none",
                background: CORAL, color: WHITE,
                fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: "'Inter',sans-serif",
              }}>
                Download CSV
              </button>
            </div>

            {/* Preview table */}
            {isOpen && preview && !preview.error && (
              <div style={{ marginTop: 16, overflowX: "auto" }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: MUTED, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5 }}>
                  First 8 rows of {info.full}
                </div>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: NAVY }}>
                      {preview.columns.map(col => (
                        <th key={col} style={{ padding: "8px 10px", color: WHITE, fontWeight: 700, textAlign: "left", whiteSpace: "nowrap", letterSpacing: 0.3 }}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row, i) => (
                      <tr key={i} style={{ background: i % 2 === 0 ? WHITE : BG }}>
                        {row.map((cell, j) => (
                          <td key={j} style={{ padding: "7px 10px", color: TEXT, borderBottom: `1px solid ${BORDER}`, whiteSpace: "nowrap" }}>{cell ?? "—"}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ── AlgorithmsPage component ──────────────────────────────────────────────────
function AlgorithmsPage() {
  const [diffDataset, setDiffDataset] = useState("adult")
  const [diffAttr, setDiffAttr]       = useState("sex")
  const [diffData, setDiffData]       = useState(null)
  const [diffLoading, setDiffLoading] = useState(false)
  const [showDiff, setShowDiff]       = useState(false)

  const loadDiff = async () => {
    setDiffLoading(true)
    try {
      const res = await axios.get(`http://localhost:8000/algo-data-diff/${diffDataset}/${diffAttr}`)
      setDiffData(res.data)
      setShowDiff(true)
    } catch { }
    setDiffLoading(false)
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {ALGORITHMS.map((a, i) => (
        <div key={i} style={card}>
          <div style={{ display: "flex", gap: 20, alignItems: "flex-start" }}>
            <img src={a.icon} style={{ width: 40, height: 40, flexShrink: 0, marginTop: 2 }} />
            <div style={{ flex: 1 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                <span style={{ fontWeight: 800, color: TEXT, fontSize: 18 }}>{a.label}</span>
                <span style={{ background: a.sub === "Pre-processing" ? `${CORAL2}66` : `${BLUE}33`, color: a.sub === "Pre-processing" ? CORAL : NAVY, fontSize: 11, fontWeight: 700, padding: "3px 10px", borderRadius: 6 }}>{a.sub}</span>
              </div>
              <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.7, margin: "0 0 14px" }}>{ALGO_INFO[i].desc}</p>
              <div style={{ background: BG, borderLeft: `3px solid ${CORAL}`, padding: "10px 16px", borderRadius: "0 8px 8px 0" }}>
                <span style={{ fontSize: 13, color: CORAL, fontWeight: 700 }}>Intuition: </span>
                <span style={{ fontSize: 13, color: TEXT }}>{ALGO_INFO[i].idea}</span>
              </div>
            </div>
          </div>
        </div>
      ))}

      {/* Before/After data diff section */}
      <div style={card}>
        <div style={{ fontWeight: 800, color: TEXT, fontSize: 16, marginBottom: 6 }}>See Data Before vs After (Pre-processing Algorithms)</div>
        <p style={{ color: MUTED, fontSize: 13, lineHeight: 1.6, marginBottom: 16 }}>
          Pre-processing algorithms (Reweighing and Disparate Impact Remover) modify the training data before the model is trained.
          Select a dataset and protected attribute to see exactly what changes — either in record weights (Reweighing) or in feature values (Disparate Impact Remover).
          In-processing algorithms (Adversarial Debiasing, Exponentiated Gradient) do not change the data — they change the training process itself, so there is no data diff to show.
        </p>

        {/* Controls */}
        <div style={{ display: "flex", gap: 12, alignItems: "flex-end", marginBottom: 16, flexWrap: "wrap" }}>
          <div>
            <label style={{ display: "block", marginBottom: 4, fontSize: 12, fontWeight: 700, color: MUTED, textTransform: "uppercase", letterSpacing: 0.5 }}>Dataset</label>
            <select value={diffDataset} onChange={e => { setDiffDataset(e.target.value); setDiffAttr(ATTRS[e.target.value][0]); setDiffData(null); setShowDiff(false) }}
              style={{ padding: "8px 12px", borderRadius: 8, border: `1.5px solid ${BORDER}`, fontSize: 13, fontFamily: "'Inter',sans-serif", color: TEXT, background: WHITE }}>
              {DATASETS.map(d => <option key={d} value={d}>{DATASET_INFO[d].full}</option>)}
            </select>
          </div>
          <div>
            <label style={{ display: "block", marginBottom: 4, fontSize: 12, fontWeight: 700, color: MUTED, textTransform: "uppercase", letterSpacing: 0.5 }}>Protected Attribute</label>
            <select value={diffAttr} onChange={e => { setDiffAttr(e.target.value); setDiffData(null); setShowDiff(false) }}
              style={{ padding: "8px 12px", borderRadius: 8, border: `1.5px solid ${BORDER}`, fontSize: 13, fontFamily: "'Inter',sans-serif", color: TEXT, background: WHITE }}>
              {ATTRS[diffDataset].map(a => <option key={a} value={a}>{a.charAt(0).toUpperCase()+a.slice(1)}</option>)}
            </select>
          </div>
          <button onClick={loadDiff} disabled={diffLoading} style={{
            padding: "9px 20px", borderRadius: 8, border: "none",
            background: diffLoading ? MUTED : NAVY, color: WHITE,
            fontSize: 13, fontWeight: 700, cursor: diffLoading ? "not-allowed" : "pointer", fontFamily: "'Inter',sans-serif",
          }}>
            {diffLoading ? "Loading..." : "Load Data Diff"}
          </button>
        </div>

        {/* Reweighing diff */}
        {showDiff && diffData && !diffData.error && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

            {/* Reweighing */}
            <div>
              <div style={{ fontWeight: 700, color: TEXT, fontSize: 15, marginBottom: 4 }}>Reweighing — What Changes</div>
              <p style={{ color: MUTED, fontSize: 13, lineHeight: 1.6, marginBottom: 12 }}>
                Reweighing does <strong>not</strong> change any feature values or labels. It only changes the <strong style={{ color: CORAL }}>weight</strong> column — how much each record "counts" during training. Records from underrepresented (group, outcome) combinations get higher weights.
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, overflowX: "auto" }}>
                {[["Before Reweighing", diffData.reweighing.before, false], ["After Reweighing", diffData.reweighing.after, true]].map(([title, rows, isAfter]) => (
                  <div key={title}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: isAfter ? NAVY : CORAL, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5 }}>{title}</div>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                      <thead>
                        <tr style={{ background: isAfter ? NAVY : CORAL }}>
                          {[diffAttr, diffData.feature_names[1], diffData.feature_names[2], "label", "weight"].map(col => (
                            <th key={col} style={{ padding: "6px 8px", color: WHITE, fontWeight: 700, textAlign: "center", whiteSpace: "nowrap" }}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((row, i) => {
                          const beforeRow = diffData.reweighing.before[i]
                          const weightChanged = isAfter && Math.abs(row.weight - beforeRow.weight) > 0.01
                          return (
                            <tr key={i} style={{ background: i % 2 === 0 ? WHITE : BG }}>
                              {[diffAttr, diffData.feature_names[1], diffData.feature_names[2], "label", "weight"].map(col => (
                                <td key={col} style={{
                                  padding: "6px 8px", textAlign: "center", borderBottom: `1px solid ${BORDER}`,
                                  color: col === "weight" && weightChanged ? CORAL : TEXT,
                                  fontWeight: col === "weight" && weightChanged ? 800 : 400,
                                  background: col === "weight" && weightChanged ? `${CORAL}15` : "transparent",
                                }}>
                                  {col === "label" ? row[col] : col === "weight" ? row.weight : row[col] ?? "—"}
                                </td>
                              ))}
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                ))}
              </div>
              <p style={{ color: MUTED, fontSize: 12, marginTop: 10, fontStyle: "italic" }}>
                Highlighted cells (coral) show weights that changed significantly. Notice how records from underrepresented groups receive higher weights after reweighing.
              </p>
            </div>

            {/* Disparate Impact Remover diff */}
            <div style={{ borderTop: `1px solid ${BORDER}`, paddingTop: 20 }}>
              <div style={{ fontWeight: 700, color: TEXT, fontSize: 15, marginBottom: 4 }}>Disparate Impact Remover — What Changes</div>
              <p style={{ color: MUTED, fontSize: 13, lineHeight: 1.6, marginBottom: 12 }}>
                Disparate Impact Remover changes the actual <strong>feature values</strong> to make their distributions more similar across groups (at repair_level=1.0). Labels and weights stay the same. Notice how numeric feature values shift after repair.
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, overflowX: "auto" }}>
                {[["Before Repair", diffData.di_remover.before, false], ["After Repair (level=1.0)", diffData.di_remover.after, true]].map(([title, rows, isAfter]) => (
                  <div key={title}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: isAfter ? NAVY : CORAL, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5 }}>{title}</div>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                      <thead>
                        <tr style={{ background: isAfter ? NAVY : CORAL }}>
                          {[diffAttr, diffData.feature_names[1], diffData.feature_names[2], "label"].map(col => (
                            <th key={col} style={{ padding: "6px 8px", color: WHITE, fontWeight: 700, textAlign: "center", whiteSpace: "nowrap" }}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((row, i) => {
                          const beforeRow = diffData.di_remover.before[i]
                          return (
                            <tr key={i} style={{ background: i % 2 === 0 ? WHITE : BG }}>
                              {[diffAttr, diffData.feature_names[1], diffData.feature_names[2], "label"].map(col => {
                                const changed = isAfter && col !== "label" && beforeRow && Math.abs((row[col] || 0) - (beforeRow[col] || 0)) > 0.005
                                return (
                                  <td key={col} style={{
                                    padding: "6px 8px", textAlign: "center", borderBottom: `1px solid ${BORDER}`,
                                    color: changed ? NAVY : TEXT, fontWeight: changed ? 800 : 400,
                                    background: changed ? `${BLUE}20` : "transparent",
                                  }}>
                                    {col === "label" ? row[col] : row[col] ?? "—"}
                                  </td>
                                )
                              })}
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                ))}
              </div>
              <p style={{ color: MUTED, fontSize: 12, marginTop: 10, fontStyle: "italic" }}>
                Highlighted cells (blue) show feature values that changed after repair. The protected attribute column itself does not change — only the other features are adjusted.
              </p>
            </div>

          </div>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const [sidebar, setSidebar]         = useState(true)
  const [page, setPage]               = useState("Introduction")
  const [dataset, setDataset]         = useState("adult")
  const [attr, setAttr]               = useState("sex")
  const [algoIdx, setAlgoIdx]         = useState(0)
  const [result, setResult]           = useState(null)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState(null)
  const [showExplain, setShowExplain] = useState(true)
  const [showCode, setShowCode]       = useState(false)

  const algo       = ALGORITHMS[algoIdx]
  const beforeData = result && algo.beforeKey ? result[algo.beforeKey] : null
  const afterData  = result && algo.afterKey  ? result[algo.afterKey]  : null
  const barData    = beforeData
    ? Object.keys(METRIC_LABELS).filter(k => beforeData[k] !== undefined).map(k => ({
        metric: METRIC_LABELS[k], Before: beforeData[k], After: afterData[k],
      }))
    : []

  const run = async () => {
    setLoading(true); setError(null); setResult(null); setShowExplain(true)
    try {
      const r = await axios.post(`http://localhost:8000${algo.endpoint}`, { dataset, protected_attr: attr })
      setResult(r.data)
    } catch { setError("Could not reach backend. Is uvicorn running?") }
    setLoading(false)
  }

  const SW = sidebar ? 230 : 0
  const explanationLines = result ? generateExplanation(result, algoIdx, dataset, attr) : []

  return (
    <>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
      <style>{`*{box-sizing:border-box;margin:0;padding:0;}html,body,#root{background:${BG};min-height:100vh;width:100%;overflow-x:hidden;}`}</style>

      <div style={{ display: "flex", minHeight: "100vh", fontFamily: "'Inter',sans-serif", background: BG, textAlign: "left" }}>

        {/* ── SIDEBAR ── */}
        <div style={{
          width: SW, background: NAVY, flexShrink: 0,
          position: "fixed", top: 0, left: 0, height: "100vh", zIndex: 20,
          overflow: "hidden", transition: "width 0.25s", display: "flex", flexDirection: "column",
        }}>
          {sidebar && <>
            <div style={{ padding: "20px 16px 28px", display: "flex", alignItems: "center", gap: 10 }}>
              <button onClick={() => setSidebar(false)} style={{
                background: "rgba(255,255,255,0.1)", border: "none", borderRadius: 8,
                width: 32, height: 32, cursor: "pointer", display: "flex", alignItems: "center",
                justifyContent: "center", flexShrink: 0,
              }}>
                <img src="/sidebar.png" style={{ width: 16, height: 16, filter: "invert(1) brightness(2)" }} />
              </button>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ width: 36, height: 36, borderRadius: 10, background: `linear-gradient(135deg, ${BLUE}, ${CORAL})`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                  <img src="/scale.png" style={{ width: 20, height: 20, filter: "invert(1) brightness(2)" }} />
                </div>
                <div>
                  <div style={{ color: WHITE, fontWeight: 800, fontSize: 17, letterSpacing: -0.5, lineHeight: 1 }}>AIF360</div>
                  <div style={{ color: MUTED, fontSize: 10, fontWeight: 600, letterSpacing: 2, marginTop: 2 }}>VIRTUAL LAB</div>
                </div>
              </div>
            </div>

            <div style={{ flex: 1, padding: "0 12px" }}>
              {PAGES.map(p => (
                <div key={p} onClick={() => setPage(p)} style={{
                  display: "flex", alignItems: "center", gap: 12, padding: "12px 16px",
                  borderRadius: 12, marginBottom: 4, cursor: "pointer",
                  background: page === p ? CORAL : "transparent",
                  transition: "background 0.15s",
                }}>
                  <img src={PAGE_ICONS[p]} style={{ width: 20, height: 20, filter: "invert(1) brightness(2)", opacity: page === p ? 1 : 0.5, flexShrink: 0 }} />
                  <span style={{ color: page === p ? WHITE : MUTED, fontWeight: page === p ? 700 : 500, fontSize: 14 }}>{p}</span>
                </div>
              ))}
            </div>

            <div style={{ padding: "16px 20px", borderTop: "1px solid rgba(255,255,255,0.07)" }}>
              <div style={{ fontSize: 11, color: "#555a7a", fontWeight: 500 }}>IBM AIF360 · FastAPI · React</div>
            </div>
          </>}
        </div>

        {/* ── TOGGLE (when sidebar closed) ── */}
        {!sidebar && (
          <button onClick={() => setSidebar(true)} style={{
            position: "fixed", top: 16, left: 12, zIndex: 100,
            background: NAVY, border: "none", borderRadius: 8,
            width: 36, height: 36, cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
          }}>
            <img src="/sidebar.png" style={{ width: 18, height: 18, filter: "invert(1) brightness(2)" }} />
          </button>
        )}

        {/* ── MAIN ── */}
        <div style={{
          marginLeft: SW, flex: 1, padding: "36px 40px",
          paddingLeft: sidebar ? 40 : 60,
          transition: "margin-left 0.25s, padding-left 0.25s",
          minWidth: 0, width: `calc(100vw - ${SW}px)`,
        }}>

          {/* Topbar */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 32 }}>
            <div>
              <h1 style={{ fontWeight: 800, fontSize: 30, color: TEXT, margin: 0, lineHeight: 1.2, letterSpacing: -0.8 }}>
                {page === "Introduction" ? "Introduction" : page === "Lab" ? "Bias Mitigation Lab" : page === "Datasets" ? "Dataset Explorer" : page === "Algorithms" ? "Algorithm Reference" : page === "Glossary" ? "Glossary" : "Security Analysis"}
              </h1>
              <p style={{ color: MUTED, fontSize: 14, margin: "5px 0 0", fontWeight: 400 }}>
                {page === "Introduction" ? "Start here — understand what this lab is about and why it matters" : page === "Lab" ? "Run fairness algorithms and visualize results" : page === "Datasets" ? "Explore the datasets used in this lab" : page === "Algorithms" ? "Understand how each algorithm works" : page === "Glossary" ? "Plain-english definitions for every term used in this lab" : "Vulnerability analysis of fairness datasets"}
              </p>
            </div>
            {/* Backend Connected badge block has been removed from right here */}
          </div>

          {/* ══════════ LAB ══════════ */}
          {page === "Lab" && <>

            <div style={card}>
              <div style={{ fontSize: 11, fontWeight: 700, color: CORAL, letterSpacing: 2, marginBottom: 20, textTransform: "uppercase" }}>Configure Experiment</div>
              <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-end" }}>
                <div style={{ flex: 1, minWidth: 140 }}>
                  <label style={lblSt}>Dataset</label>
                  <select value={dataset} onChange={e => { setDataset(e.target.value); setAttr(ATTRS[e.target.value][0]); setResult(null) }} style={selSt}>
                    {DATASETS.map(d => <option key={d} value={d}>{DATASET_INFO[d].full}</option>)}
                  </select>
                </div>
                <div style={{ flex: 1, minWidth: 140 }}>
                  <label style={lblSt}>Protected Attribute</label>
                  <select value={attr} onChange={e => setAttr(e.target.value)} style={selSt}>
                    {ATTRS[dataset].map(a => <option key={a} value={a}>{a.charAt(0).toUpperCase()+a.slice(1)}</option>)}
                  </select>
                </div>
                <div style={{ flex: 2, minWidth: 240 }}>
                  <label style={lblSt}>Algorithm</label>
                  <select value={algoIdx} onChange={e => { setAlgoIdx(Number(e.target.value)); setResult(null) }} style={selSt}>
                    {ALGORITHMS.map((a, i) => <option key={i} value={i}>{a.label} ({a.sub})</option>)}
                  </select>
                </div>
                <button onClick={run} disabled={loading} style={{
                  padding: "11px 32px", border: "none", borderRadius: 10, fontSize: 14, fontWeight: 700,
                  background: loading ? MUTED : `linear-gradient(135deg, ${BLUE}, ${CORAL})`,
                  color: WHITE, cursor: loading ? "not-allowed" : "pointer", fontFamily: "'Inter',sans-serif",
                }}>
                  {loading ? "Running..." : "Run"}
                </button>
              </div>
            </div>

            <div style={{ ...card, background: "#e8eaf5", border: `1.5px solid ${BORDER}`, display: "flex", gap: 16, alignItems: "flex-start" }}>
              <img src={algo.icon} style={{ width: 36, height: 36, flexShrink: 0, marginTop: 2 }} />
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  <span style={{ fontWeight: 700, color: TEXT, fontSize: 16 }}>{algo.label}</span>
                  <span style={{ background: algo.sub === "Pre-processing" ? `${CORAL2}66` : `${BLUE}33`, color: algo.sub === "Pre-processing" ? CORAL : NAVY, fontSize: 11, fontWeight: 700, padding: "3px 10px", borderRadius: 6 }}>{algo.sub}</span>
                </div>
                <p style={{ color: MUTED, fontSize: 13, lineHeight: 1.7, margin: "0 0 10px", textAlign: "left" }}>{ALGO_INFO[algoIdx].desc}</p>
                <div style={{ background: WHITE, borderLeft: `3px solid ${CORAL}`, padding: "8px 14px", borderRadius: "0 8px 8px 0" }}>
                  <span style={{ fontSize: 13, color: CORAL, fontWeight: 600 }}>Intuition: </span>
                  <span style={{ fontSize: 13, color: TEXT }}>{ALGO_INFO[algoIdx].idea}</span>
                </div>
              </div>
            </div>

            {error && <div style={{ background: "#fff0ee", border: `1px solid ${CORAL}`, borderRadius: 10, padding: "12px 16px", color: CORAL, marginBottom: 20, fontWeight: 600 }}>{error}</div>}

            {/* View Code block */}
            <div style={{ ...card, padding: "16px 24px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <div style={{ fontWeight: 700, color: TEXT, fontSize: 15 }}>View Code</div>
                  <div style={{ color: MUTED, fontSize: 12, marginTop: 2 }}>Full Python implementation with step-by-step explanations</div>
                </div>
                <button onClick={() => setShowCode(o => !o)} style={{
                  background: showCode ? BG : NAVY, border: "none", borderRadius: 8,
                  padding: "7px 18px", fontSize: 13, fontWeight: 700,
                  color: showCode ? TEXT : WHITE, cursor: "pointer", fontFamily: "'Inter',sans-serif",
                }}>
                  {showCode ? "Hide" : "Show"}
                </button>
              </div>
              {showCode && (
                <pre style={{
                  marginTop: 16, background: NAVY, color: "#e8eaf0", borderRadius: 10,
                  padding: 20, fontSize: 12, lineHeight: 1.7, overflowX: "auto",
                  whiteSpace: "pre-wrap", wordBreak: "break-word", fontFamily: "monospace",
                }}>
                  {ALGO_CODE[algoIdx]}
                </pre>
              )}
            </div>

            {loading && (
              <div style={{ ...card, textAlign: "center", padding: 48 }}>
                <div style={{ fontWeight: 700, color: TEXT, fontSize: 16 }}>Running {algo.label}...</div>
                <div style={{ color: MUTED, fontSize: 13, marginTop: 8 }}>{algoIdx === 2 ? "Training two neural networks — this takes 2-3 minutes" : "Processing dataset — about 30-60 seconds"}</div>
              </div>
            )}

            {/* ── LINE CHART RESULTS ── */}
            {result && algo.type === "line" && <>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16, marginBottom: 20 }}>
                {[
                  { label: "Dataset",             value: DATASET_INFO[dataset].full,                                                          color: TEXT },
                  { label: "Protected Attribute", value: attr.charAt(0).toUpperCase()+attr.slice(1),                                          color: CORAL },
                  { label: "Passes Threshold",    value: result.repair_levels.filter(r => r.disparate_impact >= 0.8).length + " / 11 levels",  color: NAVY },
                ].map(s => (
                  <div key={s.label} style={{ background: WHITE, borderRadius: 12, padding: "16px 20px", boxShadow: "0 2px 8px rgba(0,0,0,0.05)" }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: MUTED, letterSpacing: 1, textTransform: "uppercase", marginBottom: 4 }}>{s.label}</div>
                    <div style={{ fontSize: 20, fontWeight: 800, color: s.color }}>{s.value}</div>
                  </div>
                ))}
              </div>
              <div style={card}>
                <div style={{ fontWeight: 700, color: TEXT, fontSize: 16, marginBottom: 4 }}>Disparate Impact vs Repair Level</div>
                <div style={{ color: MUTED, fontSize: 12, marginBottom: 20 }}>Dotted line at 0.8 = legal threshold · Dotted line at 1.0 = perfect fairness</div>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={result.repair_levels}>
                    <CartesianGrid strokeDasharray="3 3" stroke={BG} />
                    <XAxis dataKey="repair_level" label={{ value: "Repair Level", position: "insideBottom", offset: -5, fontSize: 12 }} />
                    <YAxis domain={[0.4, 1.2]} />
                    <Tooltip contentStyle={{ borderRadius: 8, border: "none", boxShadow: "0 4px 12px rgba(0,0,0,0.1)" }} />
                    <ReferenceLine y={1.0} stroke={NAVY} strokeDasharray="4 4" />
                    <ReferenceLine y={0.8} stroke={CORAL} strokeDasharray="4 4" />
                    <Line type="monotone" dataKey="disparate_impact" stroke={CORAL} strokeWidth={2.5} dot={{ r: 5, fill: CORAL }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div style={card}>
                <div style={{ fontWeight: 700, color: TEXT, fontSize: 16, marginBottom: 16 }}>Results</div>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead><tr style={{ background: BG }}>
                    <th style={thSt}>Repair Level</th>
                    <th style={thSt}>Disparate Impact</th>
                    <th style={thSt}>Above 0.8 Threshold?</th>
                  </tr></thead>
                  <tbody>
                    {result.repair_levels.map(r => (
                      <tr key={r.repair_level}>
                        <td style={{ ...tdSt, fontWeight: 600 }}>{r.repair_level}</td>
                        <td style={{ ...tdSt, color: r.disparate_impact >= 0.8 ? NAVY : CORAL, fontWeight: 700 }}>{r.disparate_impact}</td>
                        <td style={{ ...tdSt, color: r.disparate_impact >= 0.8 ? NAVY : CORAL, fontWeight: 700 }}>{r.disparate_impact >= 0.8 ? "Yes" : "No"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* EXPLANATION */}
              <div style={{ ...card, background: `${NAVY}08`, border: `1.5px solid ${BORDER}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: showExplain ? 16 : 0 }}>
                  <div style={{ fontWeight: 700, color: TEXT, fontSize: 16 }}>Understand Your Results</div>
                  <button onClick={() => setShowExplain(o => !o)} style={{ background: "none", border: "none", color: CORAL, fontWeight: 700, fontSize: 13, cursor: "pointer" }}>{showExplain ? "Hide" : "Show"}</button>
                </div>
                {showExplain && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    {explanationLines.map((line, i) => (
                      <div key={i} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                        <div style={{ width: 24, height: 24, borderRadius: "50%", background: CORAL, color: WHITE, fontSize: 12, fontWeight: 800, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 1 }}>{i+1}</div>
                        <p style={{ color: TEXT, fontSize: 14, lineHeight: 1.7, margin: 0 }}>{line}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>}

            {/* ── BAR CHART RESULTS ── */}
            {result && algo.type === "bar" && beforeData && <>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, marginBottom: 20 }}>
                {[
                  { label: "Mean Diff Before",     value: result.dataset_bias.mean_diff_before, color: CORAL },
                  { label: "Mean Diff After",      value: result.dataset_bias.mean_diff_after,  color: NAVY },
                  { label: "Balanced Acc. Before", value: beforeData.balanced_accuracy,          color: MUTED },
                  { label: "Balanced Acc. After",  value: afterData.balanced_accuracy,           color: NAVY },
                ].map(s => (
                  <div key={s.label} style={{ background: WHITE, borderRadius: 12, padding: "16px 20px", boxShadow: "0 2px 8px rgba(0,0,0,0.05)" }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: MUTED, letterSpacing: 1, textTransform: "uppercase", marginBottom: 4 }}>{s.label}</div>
                    <div style={{ fontSize: 22, fontWeight: 800, color: s.color }}>{s.value}</div>
                  </div>
                ))}
              </div>
              <div style={card}>
                <div style={{ fontWeight: 700, color: TEXT, fontSize: 16, marginBottom: 4 }}>Fairness Metrics — Before vs After</div>
                <div style={{ color: MUTED, fontSize: 12, marginBottom: 20 }}>Coral = before mitigation · Navy = after mitigation</div>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={barData} margin={{ top: 5, right: 20, left: 0, bottom: 70 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={BG} />
                    <XAxis dataKey="metric" angle={-30} textAnchor="end" interval={0} tick={{ fontSize: 11 }} />
                    <YAxis domain={[-0.5, 1.1]} />
                    <Tooltip contentStyle={{ borderRadius: 8, border: "none", boxShadow: "0 4px 12px rgba(0,0,0,0.1)" }} />
                    <Legend verticalAlign="top" />
                    <Bar dataKey="Before" fill={CORAL} radius={[4,4,0,0]} />
                    <Bar dataKey="After"  fill={NAVY}  radius={[4,4,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div style={card}>
                <div style={{ fontWeight: 700, color: TEXT, fontSize: 16, marginBottom: 16 }}>Results</div>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead><tr style={{ background: BG }}>
                    <th style={{ ...thSt, textAlign: "left" }}>Metric</th>
                    <th style={thSt}>Before</th>
                    <th style={thSt}>After</th>
                    <th style={thSt}>Change</th>
                  </tr></thead>
                  <tbody>
                    {Object.keys(METRIC_LABELS).filter(k => beforeData[k] !== undefined).map(key => {
                      const b = beforeData[key], a = afterData[key]
                      const diff = (a - b).toFixed(4)
                      const good = key === "balanced_accuracy" ? a >= b : Math.abs(a) < Math.abs(b)
                      return (
                        <tr key={key}>
                          <td style={{ ...tdSt, textAlign: "left", fontWeight: 600, color: TEXT }}>{METRIC_LABELS[key]}</td>
                          <td style={{ ...tdSt, color: CORAL, fontWeight: 700 }}>{b}</td>
                          <td style={{ ...tdSt, color: NAVY,  fontWeight: 700 }}>{a}</td>
                          <td style={{ ...tdSt, color: good ? NAVY : CORAL, fontWeight: 800 }}>{diff > 0 ? "+" : ""}{diff}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>

              {/* EXPLANATION */}
              <div style={{ ...card, background: `${NAVY}08`, border: `1.5px solid ${BORDER}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: showExplain ? 16 : 0 }}>
                  <div style={{ fontWeight: 700, color: TEXT, fontSize: 16 }}>Understand Your Results</div>
                  <button onClick={() => setShowExplain(o => !o)} style={{ background: "none", border: "none", color: CORAL, fontWeight: 700, fontSize: 13, cursor: "pointer" }}>{showExplain ? "Hide" : "Show"}</button>
                </div>
                {showExplain && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    {explanationLines.map((line, i) => (
                      <div key={i} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                        <div style={{ width: 24, height: 24, borderRadius: "50%", background: CORAL, color: WHITE, fontSize: 12, fontWeight: 800, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 1 }}>{i+1}</div>
                        <p style={{ color: TEXT, fontSize: 14, lineHeight: 1.7, margin: 0 }}>{line}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>}

            {/* Download Notebooks — always visible at bottom of lab */}
            <div style={card}>
              <div style={{ fontWeight: 700, color: TEXT, fontSize: 15, marginBottom: 4 }}>Download Jupyter Notebooks</div>
              <div style={{ color: MUTED, fontSize: 13, marginBottom: 16 }}>
                Download the original notebook for any algorithm, open it in Jupyter, and run it directly — no additional setup needed beyond installing AIF360.
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12 }}>
                {[
                  { label: "Reweighing",               file: "demo_reweighing_preproc.ipynb",              sub: "Pre-processing" },
                  { label: "Disparate Impact Remover", file: "demo_disparate_impact_remover.ipynb",         sub: "Pre-processing" },
                  { label: "Adversarial Debiasing",    file: "demo_adversarial_debiasing.ipynb",            sub: "In-processing" },
                  { label: "Exponentiated Gradient",   file: "demo_exponentiated_gradient_reduction.ipynb", sub: "In-processing" },
                ].map(nb => (
                  <a key={nb.file} href={`/${nb.file}`} download={nb.file} style={{ textDecoration: "none" }}>
                    <div style={{
                      background: BG, borderRadius: 10, padding: "14px 16px",
                      border: `1.5px solid ${BORDER}`, cursor: "pointer",
                      transition: "border-color 0.15s",
                    }}
                      onMouseEnter={e => e.currentTarget.style.borderColor = CORAL}
                      onMouseLeave={e => e.currentTarget.style.borderColor = BORDER}
                    >
                      <div style={{ fontSize: 13, fontWeight: 700, color: TEXT, marginBottom: 4 }}>{nb.label}</div>
                      <div style={{ fontSize: 11, color: nb.sub === "Pre-processing" ? CORAL : NAVY, fontWeight: 600, marginBottom: 8 }}>{nb.sub}</div>
                      <div style={{ fontSize: 11, color: MUTED, fontFamily: "monospace", wordBreak: "break-all" }}>{nb.file}</div>
                      <div style={{ marginTop: 10, background: CORAL, color: WHITE, borderRadius: 6, padding: "5px 0", textAlign: "center", fontSize: 12, fontWeight: 700 }}>
                        Download .ipynb
                      </div>
                    </div>
                  </a>
                ))}
              </div>
            </div>

          </>}

          {/* ══════════ DATASETS ══════════ */}
          {page === "Datasets" && (
            <DatasetExplorer />
          )}

          {/* ══════════ ALGORITHMS ══════════ */}
          {page === "Algorithms" && (
            <AlgorithmsPage />
          )}

          {/* ══════════ METRICS ══════════ */}
          {page === "Metrics" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              <div style={{ ...card, background: `${NAVY}08`, border: `1px solid ${BORDER}` }}>
                <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.7, margin: 0 }}>
                  Fairness metrics measure different aspects of how a model treats protected groups. No single metric captures all forms of fairness — this is why we report five complementary measures. In general, values closer to 0 indicate fairness for difference-based metrics, and values closer to 1 indicate fairness for ratio-based metrics.
                </p>
              </div>
              {METRICS_INFO.map(m => (
                <div key={m.key} style={card}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
                    <div style={{ fontWeight: 800, color: TEXT, fontSize: 18 }}>{m.name}</div>
                    <div style={{ background: `${CORAL2}55`, borderRadius: 8, padding: "4px 12px", fontSize: 12, fontWeight: 700, color: CORAL, fontFamily: "monospace" }}>{m.key}</div>
                  </div>
                  <div style={{ background: BG, borderRadius: 8, padding: "10px 16px", marginBottom: 14, fontFamily: "monospace", fontSize: 13, color: NAVY, fontWeight: 600 }}>
                    {m.formula}
                  </div>
                  <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.7, margin: "0 0 14px" }}>{m.interpret}</p>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                    <div style={{ background: BG, borderRadius: 8, padding: "10px 14px" }}>
                      <div style={{ fontSize: 11, fontWeight: 700, color: MUTED, letterSpacing: 1, marginBottom: 4, textTransform: "uppercase" }}>Ideal Value</div>
                      <div style={{ fontSize: 13, color: NAVY, fontWeight: 600 }}>{m.ideal}</div>
                    </div>
                    <div style={{ background: BG, borderRadius: 8, padding: "10px 14px" }}>
                      <div style={{ fontSize: 11, fontWeight: 700, color: MUTED, letterSpacing: 1, marginBottom: 4, textTransform: "uppercase" }}>Range & Interpretation</div>
                      <div style={{ fontSize: 13, color: NAVY, fontWeight: 600 }}>{m.range}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* ══════════ INTRODUCTION ══════════ */}
          {page === "Introduction" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

              {/* What is bias */}
              <div style={card}>
                <div style={{ fontWeight: 800, color: TEXT, fontSize: 18, marginBottom: 12 }}>What is Algorithmic Bias?</div>
                <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.8, margin: "0 0 14px" }}>
                  Imagine you apply for a bank loan. A computer program — not a human — decides whether you get it. This program was trained on historical data: past loan applications, who got approved, who didn't. The problem? If the historical data reflects human prejudices (e.g. banks historically gave fewer loans to women or minorities), the AI learns those same prejudices and repeats them — at massive scale, automatically, and often invisibly.
                </p>
                <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.8, margin: 0 }}>
                  This is algorithmic bias: when an AI system produces systematically unfair outcomes for certain groups of people based on characteristics like gender, race, or age. The scary part is that the AI isn't being "evil" — it's just doing exactly what it was trained to do.
                </p>
              </div>

              {/* Real world example */}
              <div style={{ ...card, borderLeft: `4px solid ${CORAL}` }}>
                <div style={{ fontWeight: 800, color: TEXT, fontSize: 16, marginBottom: 10 }}>A Real Example: The COMPAS Algorithm</div>
                <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.8, margin: "0 0 12px" }}>
                  In 2016, investigative journalists at ProPublica analyzed COMPAS — a commercial AI tool used by US courts to predict whether a criminal defendant would reoffend. Judges used its scores to decide bail, sentencing, and parole.
                </p>
                <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.8, margin: "0 0 12px" }}>
                  Their finding was alarming: <strong style={{ color: TEXT }}>Black defendants were nearly twice as likely as white defendants to be incorrectly flagged as high risk</strong>, even when controlling for criminal history. Meanwhile, white defendants who did go on to reoffend were more often incorrectly labeled low risk.
                </p>
                <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.8, margin: 0 }}>
                  This is not just a technical problem — it is a civil rights problem. Real people's freedom was affected by a biased algorithm. The COMPAS dataset used in this lab is the same data from that study.
                </p>
              </div>

              {/* Why hard */}
              <div style={card}>
                <div style={{ fontWeight: 800, color: TEXT, fontSize: 18, marginBottom: 12 }}>Why Is This Hard to Fix?</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
                  {[
                    { title: "Historical data is biased", body: "If past hiring, lending, or sentencing decisions were discriminatory, training data reflects that. The AI learns from — and perpetuates — those patterns." },
                    { title: "Removing protected attributes doesn't help", body: "Even if you remove gender or race from the data, the model can infer them from correlated features like occupation, zip code, or name. Experiments on the Adult dataset confirmed this — sex can be predicted with 77% accuracy from other features alone." },
                    { title: "Fairness vs accuracy tradeoff", body: "Making a model fairer often slightly reduces its overall accuracy. There is no free lunch — every bias mitigation technique involves a tradeoff that must be carefully chosen." },
                    { title: "Multiple definitions of fairness conflict", body: "Mathematically, it is impossible to satisfy all fairness criteria simultaneously. Different metrics measure different things, and optimizing one can worsen another." },
                  ].map(s => (
                    <div key={s.title} style={{ background: BG, borderRadius: 10, padding: "14px 16px" }}>
                      <div style={{ fontWeight: 700, color: TEXT, fontSize: 13, marginBottom: 6 }}>{s.title}</div>
                      <div style={{ color: MUTED, fontSize: 13, lineHeight: 1.6 }}>{s.body}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* What is AIF360 */}
              <div style={card}>
                <div style={{ fontWeight: 800, color: TEXT, fontSize: 18, marginBottom: 12 }}>What is IBM AIF360?</div>
                <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.8, margin: "0 0 12px" }}>
                  AI Fairness 360 (AIF360) is an open-source Python toolkit developed by IBM Research. It provides a comprehensive set of tools to detect and mitigate bias in machine learning models. Think of it as a medical kit for biased AI — it comes with diagnostic tools (fairness metrics) and treatments (bias mitigation algorithms).
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
                  {[
                    { label: "Fairness Metrics", value: "70+", sub: "ways to measure bias" },
                    { label: "Mitigation Algorithms", value: "10+", sub: "pre, in, and post-processing" },
                    { label: "Supported Datasets", value: "9", sub: "real-world benchmark datasets" },
                  ].map(s => (
                    <div key={s.label} style={{ background: BG, borderRadius: 10, padding: "14px 16px", textAlign: "center" }}>
                      <div style={{ fontSize: 28, fontWeight: 800, color: CORAL, marginBottom: 4 }}>{s.value}</div>
                      <div style={{ fontSize: 12, fontWeight: 700, color: TEXT }}>{s.label}</div>
                      <div style={{ fontSize: 11, color: MUTED, marginTop: 2 }}>{s.sub}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* How to use this lab */}
              <div style={card}>
                <div style={{ fontWeight: 800, color: TEXT, fontSize: 18, marginBottom: 16 }}>How to Use This Lab</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  {[
                    { step: "1", title: "Read the Glossary", body: "Before running experiments, visit the Glossary page to understand what terms like 'disparate impact', 'protected attribute', and 'pre-processing' mean. You'll encounter them everywhere." },
                    { step: "2", title: "Explore the Datasets", body: "Visit the Datasets page to understand what data we're working with — who the people in the data are, what the AI is trying to predict, and which groups are considered 'protected'." },
                    { step: "3", title: "Understand the Algorithms", body: "Visit the Algorithms page to learn how each bias mitigation technique works in plain English before you run it." },
                    { step: "4", title: "Run an Experiment in the Lab", body: "Go to the Lab page, pick a dataset, a protected attribute (like sex or race), and an algorithm. Hit Run and wait for results — it takes 30 seconds to 3 minutes depending on the algorithm. After results appear, scroll down to the 'Understand Your Results' section — it translates the numbers into plain English specifically for your chosen dataset and group." },
                    { step: "5", title: "Check the Security section", body: "See how adversarial attacks can manipulate fairness metrics and why fairness-aware AI systems need security considerations too." },
                  ].map(s => (
                    <div key={s.step} style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
                      <div style={{ width: 32, height: 32, borderRadius: "50%", background: `linear-gradient(135deg, ${BLUE}, ${CORAL})`, color: WHITE, fontSize: 14, fontWeight: 800, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>{s.step}</div>
                      <div>
                        <div style={{ fontWeight: 700, color: TEXT, fontSize: 14, marginBottom: 3 }}>{s.title}</div>
                        <div style={{ color: MUTED, fontSize: 13, lineHeight: 1.6 }}>{s.body}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* References & Credits */}
              <div style={card}>
                <div style={{ fontWeight: 800, color: TEXT, fontSize: 16, marginBottom: 16 }}>References & Credits</div>
                <div style={{ fontWeight: 700, color: CORAL, fontSize: 12, letterSpacing: 1, textTransform: "uppercase", marginBottom: 10 }}>Framework</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 20 }}>
                  {[{ label: "IBM AIF360 Toolkit", citation: 'R. K. E. Bellamy et al., "AI Fairness 360: An Extensible Toolkit for Detecting and Mitigating Algorithmic Bias," IBM Journal of Research and Development, vol. 63, no. 4/5, 2019.', url: "https://github.com/Trusted-AI/AIF360" }].map(r => (
                    <div key={r.label} style={{ background: BG, borderRadius: 8, padding: "12px 14px" }}>
                      <div style={{ fontWeight: 700, color: TEXT, fontSize: 13, marginBottom: 4 }}>{r.label}</div>
                      <div style={{ color: MUTED, fontSize: 12, lineHeight: 1.6, marginBottom: 6 }}>{r.citation}</div>
                      <a href={r.url} target="_blank" rel="noreferrer" style={{ fontSize: 12, color: BLUE, textDecoration: "none", fontWeight: 600 }}>{r.url}</a>
                    </div>
                  ))}
                </div>
                <div style={{ fontWeight: 700, color: CORAL, fontSize: 12, letterSpacing: 1, textTransform: "uppercase", marginBottom: 10 }}>Datasets</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 20 }}>
                  {[
                    { label: "Adult Census Income Dataset", citation: 'Becker, B. & Kohavi, R. (1996). Adult. UCI Machine Learning Repository. Extracted from the 1994 US Census database.', url: "https://archive.ics.uci.edu/ml/datasets/adult" },
                    { label: "German Credit Dataset", citation: 'Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. South German Credit dataset.', url: "https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)" },
                    { label: "COMPAS Recidivism Dataset", citation: 'Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016). Machine Bias. ProPublica. Data from Broward County, Florida.', url: "https://github.com/propublica/compas-analysis" },
                  ].map(r => (
                    <div key={r.label} style={{ background: BG, borderRadius: 8, padding: "12px 14px" }}>
                      <div style={{ fontWeight: 700, color: TEXT, fontSize: 13, marginBottom: 4 }}>{r.label}</div>
                      <div style={{ color: MUTED, fontSize: 12, lineHeight: 1.6, marginBottom: 6 }}>{r.citation}</div>
                      <a href={r.url} target="_blank" rel="noreferrer" style={{ fontSize: 12, color: BLUE, textDecoration: "none", fontWeight: 600 }}>{r.url}</a>
                    </div>
                  ))}
                </div>
                <div style={{ fontWeight: 700, color: CORAL, fontSize: 12, letterSpacing: 1, textTransform: "uppercase", marginBottom: 10 }}>Algorithm Papers</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {[
                    { label: "Reweighing", citation: 'Kamiran, F. & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. Knowledge and Information Systems, 33(1), 1–33.' },
                    { label: "Disparate Impact Remover", citation: 'Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2015). Certifying and removing disparate impact. KDD 2015.' },
                    { label: "Adversarial Debiasing", citation: 'Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). Mitigating unwanted biases with adversarial learning. AIES 2018.' },
                    { label: "Exponentiated Gradient Reduction", citation: 'Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., & Wallach, H. (2018). A reductions approach to fair classification. ICML 2018.' },
                  ].map(r => (
                    <div key={r.label} style={{ background: BG, borderRadius: 8, padding: "12px 14px" }}>
                      <div style={{ fontWeight: 700, color: TEXT, fontSize: 13, marginBottom: 4 }}>{r.label}</div>
                      <div style={{ color: MUTED, fontSize: 12, lineHeight: 1.6 }}>{r.citation}</div>
                    </div>
                  ))}
                </div>
              </div>

            </div>
          )}

          {/* ══════════ GLOSSARY ══════════ */}
          {page === "Glossary" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

              <div style={{ ...card, background: `${NAVY}08`, border: `1px solid ${BORDER}` }}>
                <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.7, margin: 0 }}>
                  This page explains every technical term you'll encounter in this lab — in plain English, without assuming any prior machine learning knowledge. If you're confused by something you see in the Lab or Security sections, look it up here first.
                </p>
              </div>

              {/* Core concepts */}
              <div style={card}>
                <div style={{ fontWeight: 800, color: TEXT, fontSize: 16, marginBottom: 16 }}>Core Concepts</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
                  {[
                    { term: "Machine Learning Model", def: "A computer program that learns patterns from data and uses those patterns to make predictions. For example, a model trained on loan applications learns which features (income, credit score, etc.) predict whether someone will repay — and then uses that to decide on new applications." },
                    { term: "Training Data", def: "The historical data used to teach the model. If this data contains biases from the past (e.g. fewer women were approved for loans historically), the model will learn those biases." },
                    { term: "Label / Outcome", def: "What the model is trying to predict. In the Adult dataset, the label is whether income exceeds $50K. In COMPAS, it's whether someone will reoffend. A 'favorable label' is the outcome considered positive (getting the loan, not reoffending)." },
                    { term: "Feature", def: "An input variable the model uses to make predictions — things like age, education, occupation, or hours worked per week. Features are the model's 'clues'." },
                    { term: "Protected Attribute", def: "A characteristic that should NOT influence decisions because it is ethically or legally protected — typically race, sex, age, or national origin. Bias occurs when a model's predictions unfairly depend on these attributes." },
                    { term: "Privileged Group", def: "The group historically receiving more favorable outcomes. In our datasets: Male (sex), White (race), Age ≥ 25 (age). Being 'privileged' here is a statistical observation, not a value judgment." },
                    { term: "Unprivileged Group", def: "The group historically receiving fewer favorable outcomes: Female, Non-white, Age < 25. Bias mitigation aims to close the gap between privileged and unprivileged groups." },
                    { term: "Classifier", def: "A type of model that sorts inputs into categories — e.g. 'will repay loan' vs 'won't repay loan'. All four algorithms in this lab produce classifiers." },
                  ].map((item, i) => (
                    <div key={item.term} style={{ padding: "14px 0", borderBottom: i < 7 ? `1px solid ${BORDER}` : "none" }}>
                      <div style={{ fontWeight: 700, color: TEXT, fontSize: 14, marginBottom: 4 }}>{item.term}</div>
                      <div style={{ color: MUTED, fontSize: 13, lineHeight: 1.7 }}>{item.def}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Fairness metrics */}
              <div style={card}>
                <div style={{ fontWeight: 800, color: TEXT, fontSize: 16, marginBottom: 4 }}>Fairness Metrics — What the Numbers Mean</div>
                <p style={{ color: MUTED, fontSize: 13, lineHeight: 1.7, margin: "0 0 16px" }}>These are the measurements shown in the Lab after you run an algorithm. Each one measures a different aspect of fairness.</p>
                <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
                  {[
                    {
                      term: "Disparate Impact",
                      simple: "How often does the unprivileged group get a good outcome, compared to the privileged group?",
                      detail: "Calculated as a ratio: (% of unprivileged group getting favorable outcome) ÷ (% of privileged group getting favorable outcome). A value of 1.0 means both groups are treated equally. A value of 0.5 means the unprivileged group is half as likely to get a favorable outcome. The legal standard (called the '80% rule') says this should be at least 0.8.",
                      example: "If 40% of female applicants get approved vs 80% of male applicants, disparate impact = 40/80 = 0.5. This would be illegal discrimination.",
                      ideal: "As close to 1.0 as possible. Must be ≥ 0.8.",
                    },
                    {
                      term: "Statistical Parity Difference",
                      simple: "What is the raw gap in favorable outcome rates between the two groups?",
                      detail: "Calculated as: (% unprivileged getting favorable outcome) − (% privileged getting favorable outcome). Unlike disparate impact which is a ratio, this is a simple subtraction. A value of -0.2 means the unprivileged group receives favorable outcomes 20 percentage points less often.",
                      example: "If 30% of non-white applicants and 50% of white applicants get loans, statistical parity difference = 30% − 50% = −0.2.",
                      ideal: "As close to 0.0 as possible.",
                    },
                    {
                      term: "Average Odds Difference",
                      simple: "Are both correct approvals AND correct rejections happening equally for both groups?",
                      detail: "This combines two rates: True Positive Rate (TPR — correctly identifying qualified people) and False Positive Rate (FPR — incorrectly approving unqualified people). It checks that both rates are equal across groups. A value of 0 means perfect equality on both dimensions.",
                      example: "If qualified women are approved less often than qualified men (low TPR gap) AND unqualified women are rejected more often than unqualified men (FPR gap), average odds difference will be negative.",
                      ideal: "As close to 0.0 as possible.",
                    },
                    {
                      term: "Equal Opportunity Difference",
                      simple: "Are qualified people from both groups equally likely to be correctly identified?",
                      detail: "Focuses only on True Positive Rate — among people who actually deserve a favorable outcome, are they getting it equally regardless of their protected attribute? This is especially important when false negatives are costly (e.g. denying a loan to someone who would definitely repay it).",
                      example: "If 70% of qualified men get approved but only 40% of equally qualified women do, equal opportunity difference = 40% − 70% = −0.3.",
                      ideal: "As close to 0.0 as possible.",
                    },
                    {
                      term: "Balanced Accuracy",
                      simple: "How often is the model correct overall, accounting for unequal group sizes?",
                      detail: "Regular accuracy can be misleading — a model that always predicts 'no' would be 90% accurate if only 10% of people should get approved. Balanced accuracy averages the correct rate for positive outcomes and the correct rate for negative outcomes, giving a fairer picture of model quality.",
                      example: "A model with balanced accuracy of 0.75 correctly identifies 75% of people who should get approved AND 75% of people who shouldn't — regardless of how many are in each category.",
                      ideal: "Closer to 1.0 is better. After bias mitigation, this typically drops slightly — that small accuracy cost is the fairness-accuracy tradeoff.",
                    },
                  ].map((item, i) => (
                    <div key={item.term} style={{ padding: "16px 0", borderBottom: i < 4 ? `1px solid ${BORDER}` : "none" }}>
                      <div style={{ fontWeight: 700, color: TEXT, fontSize: 14, marginBottom: 4 }}>{item.term}</div>
                      <div style={{ color: CORAL, fontSize: 12, fontWeight: 600, marginBottom: 6, fontStyle: "italic" }}>{item.simple}</div>
                      <div style={{ color: MUTED, fontSize: 13, lineHeight: 1.7, marginBottom: 8 }}>{item.detail}</div>
                      <div style={{ background: BG, borderRadius: 8, padding: "8px 12px", marginBottom: 6 }}>
                        <span style={{ fontSize: 12, fontWeight: 700, color: TEXT }}>Example: </span>
                        <span style={{ fontSize: 12, color: MUTED }}>{item.example}</span>
                      </div>
                      <div style={{ fontSize: 12, color: NAVY, fontWeight: 600 }}>Ideal value: {item.ideal}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Algorithm types */}
              <div style={card}>
                <div style={{ fontWeight: 800, color: TEXT, fontSize: 16, marginBottom: 16 }}>Algorithm Types for Bias Mitigation</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
                  {[
                    { term: "Pre-processing", def: "Bias mitigation applied to the training data before the model is trained. The data itself is modified — for example, by reweighting records or editing feature values — so that the model learns from a fairer version of the data. Like fixing a recipe before cooking, rather than trying to fix the dish afterwards." },
                    { term: "In-processing", def: "Bias mitigation built into the training process itself. The learning algorithm is modified with fairness constraints so that it actively avoids discriminatory patterns while learning. More powerful than pre-processing but also more complex." },
                    { term: "Post-processing (not covered here)", def: "Bias mitigation applied to the model's predictions after training. Thresholds are adjusted differently for different groups. The simplest approach but also the most limited." },
                    { term: "Repair Level", def: "A parameter specific to the Disparate Impact Remover algorithm, ranging from 0.0 to 1.0. At 0.0, no changes are made to the data. At 1.0, feature distributions are made completely identical across groups. Higher repair levels improve fairness but may reduce accuracy." },
                  ].map((item, i) => (
                    <div key={item.term} style={{ padding: "14px 0", borderBottom: i < 3 ? `1px solid ${BORDER}` : "none" }}>
                      <div style={{ fontWeight: 700, color: TEXT, fontSize: 14, marginBottom: 4 }}>{item.term}</div>
                      <div style={{ color: MUTED, fontSize: 13, lineHeight: 1.7 }}>{item.def}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Indian legal context */}
              <div style={{ ...card, borderLeft: `4px solid ${CORAL}`, marginBottom: 0 }}>
                <div style={{ fontWeight: 800, color: TEXT, fontSize: 16, marginBottom: 10 }}>Legal Context in India: DPDP Act 2023</div>
                <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.8, margin: "0 0 12px" }}>
                  Unlike the United States (which has the EEOC's 80% rule) or the European Union (which has GDPR provisions on automated decision-making), India currently has no single statute that explicitly mandates algorithmic fairness for all AI systems.
                </p>
                <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.8, margin: "0 0 12px" }}>
                  The Digital Personal Data Protection Act, 2023 (DPDP Act) is India's primary data protection law. It establishes individual data rights, fiduciary duties, and consent mechanisms for personal data processing. Notably, it designates certain organizations as <strong style={{ color: TEXT }}>"Significant Data Fiduciaries"</strong> — entities required to conduct periodic algorithmic audits and Data Protection Impact Assessments (DPIAs). However, the DPDP Act does not contain specific provisions for algorithmic fairness, bias detection, or automated decision-making transparency, and its enforcement rules remain pending notification.
                </p>
                <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.8, margin: "0 0 12px" }}>
                  India's constitutional framework does provide some protection: Articles 14 and 15 guarantee equality and prohibit discrimination on grounds of religion, race, caste, sex, or place of birth. These principles become relevant when AI systems are deployed by public authorities in ways that affect individuals' rights.
                </p>
                <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.8, margin: 0 }}>
                  NITI Aayog's Responsible AI guidelines and upcoming sector-specific norms from MeitY are expected to push India toward more comprehensive AI regulation. For now, organizations deploying AI in India must treat fairness as a cross-cutting compliance obligation across the IT Act 2000, DPDP Act 2023, and constitutional principles — rather than a single bright-line rule.
                </p>
              </div>

            </div>
          )}

          {/* ══════════ SECURITY ══════════ */}
          {page === "Security" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

              {/* Intro */}
              <div style={{ ...card, borderLeft: `4px solid ${CORAL}`, marginBottom: 0 }}>
                <div style={{ fontWeight: 800, color: TEXT, fontSize: 16, marginBottom: 10 }}>Overview</div>
                <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.7, margin: 0 }}>
                  Fairness-aware ML systems face unique security vulnerabilities. This section documents three experimentally verified attack vectors on the Adult Census dataset: data poisoning attacks targeting protected groups, sensitive attribute inference from non-protected features, and privacy risks from re-identification. All experiments were conducted using logistic regression classifiers with AIF360 fairness metrics.
                </p>
              </div>

              {/* Experiment 1 — poisoning sex */}
              <div style={card}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  <div style={{ fontWeight: 800, color: TEXT, fontSize: 16 }}>Experiment 1 — Data Poisoning Attack (Sex)</div>
                  <span style={{ background: `${CORAL2}66`, color: CORAL, fontSize: 11, fontWeight: 700, padding: "3px 10px", borderRadius: 6 }}>Verified</span>
                </div>
                <p style={{ color: MUTED, fontSize: 13, lineHeight: 1.7, margin: "0 0 8px" }}>
                  We systematically flipped the income labels of a random subset of female records in the Adult dataset and measured the effect on fairness metrics. The attacker's goal: make the model appear fairer while degrading its real-world reliability.
                </p>
                <div style={{ background: `${CORAL}11`, borderRadius: 8, padding: "10px 14px", marginBottom: 20, border: `1px solid ${CORAL2}` }}>
                  <span style={{ fontSize: 13, color: CORAL, fontWeight: 700 }}>Key finding: </span>
                  <span style={{ fontSize: 13, color: TEXT }}>At just 5% label poisoning, disparate impact increases from 0.18 → 0.26 while accuracy drops only 0.017 — the model appears to improve on fairness metrics while its actual quality quietly degrades. This makes the attack difficult to detect using standard monitoring.</span>
                </div>
                <div style={{ fontWeight: 700, color: TEXT, fontSize: 13, marginBottom: 12 }}>Disparate Impact vs Poison Percentage (Adult, Sex)</div>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={POISON_SEX} margin={{ top: 5, right: 20, left: 0, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={BG} />
                    <XAxis dataKey="pct" label={{ value: "% Labels Poisoned", position: "insideBottom", offset: -10, fontSize: 12 }} />
                    <YAxis yAxisId="left"  domain={[0, 0.8]} label={{ value: "Disparate Impact", angle: -90, position: "insideLeft", fontSize: 11 }} />
                    <YAxis yAxisId="right" orientation="right" domain={[0.6, 0.75]} label={{ value: "Accuracy", angle: 90, position: "insideRight", fontSize: 11 }} />
                    <Tooltip contentStyle={{ borderRadius: 8, border: "none", boxShadow: "0 4px 12px rgba(0,0,0,0.1)" }} />
                    <Legend verticalAlign="top" />
                    <Line yAxisId="left"  type="monotone" dataKey="disparate_impact" name="Disparate Impact" stroke={CORAL} strokeWidth={2.5} dot={{ r: 4, fill: CORAL }} />
                    <Line yAxisId="right" type="monotone" dataKey="accuracy"         name="Accuracy"         stroke={NAVY}  strokeWidth={2.5} dot={{ r: 4, fill: NAVY  }} strokeDasharray="5 5" />
                  </LineChart>
                </ResponsiveContainer>
                <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 16, fontSize: 13 }}>
                  <thead><tr style={{ background: BG }}>
                    <th style={thSt}>Poison %</th>
                    <th style={thSt}>Disparate Impact</th>
                    <th style={thSt}>Accuracy</th>
                    <th style={thSt}>Appears Fair? (≥0.8)</th>
                  </tr></thead>
                  <tbody>
                    {POISON_SEX.map(r => (
                      <tr key={r.pct}>
                        <td style={{ ...tdSt, fontWeight: 700 }}>{r.pct}</td>
                        <td style={{ ...tdSt, color: CORAL, fontWeight: 700 }}>{r.disparate_impact}</td>
                        <td style={{ ...tdSt, color: NAVY,  fontWeight: 700 }}>{r.accuracy}</td>
                        <td style={{ ...tdSt, color: r.disparate_impact >= 0.8 ? NAVY : MUTED, fontWeight: 700 }}>{r.disparate_impact >= 0.8 ? "Yes" : "No"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Experiment 2 — poisoning race */}
              <div style={card}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  <div style={{ fontWeight: 800, color: TEXT, fontSize: 16 }}>Experiment 2 — Data Poisoning Attack (Race)</div>
                  <span style={{ background: `${CORAL2}66`, color: CORAL, fontSize: 11, fontWeight: 700, padding: "3px 10px", borderRadius: 6 }}>Verified</span>
                </div>
                <p style={{ color: MUTED, fontSize: 13, lineHeight: 1.7, margin: "0 0 8px" }}>
                  Same experiment repeated for race as the protected attribute, flipping income labels for non-white individuals.
                </p>
                <div style={{ background: `${CORAL}11`, borderRadius: 8, padding: "10px 14px", marginBottom: 20, border: `1px solid ${CORAL2}` }}>
                  <span style={{ fontSize: 13, color: CORAL, fontWeight: 700 }}>Key finding: </span>
                  <span style={{ fontSize: 13, color: TEXT }}>At 10% poisoning, the model crosses the 0.8 legal fairness threshold (disparate impact = 0.84). By 15%, it exceeds 1.0, meaning the model now statistically favors non-white individuals — completely masking the original racial bias. This is a particularly deceptive attack: automated fairness audits would pass the model while the underlying data has been compromised.</span>
                </div>
                <div style={{ fontWeight: 700, color: TEXT, fontSize: 13, marginBottom: 12 }}>Disparate Impact vs Poison Percentage (Adult, Race)</div>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={POISON_RACE} margin={{ top: 5, right: 20, left: 0, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={BG} />
                    <XAxis dataKey="pct" label={{ value: "% Labels Poisoned", position: "insideBottom", offset: -10, fontSize: 12 }} />
                    <YAxis yAxisId="left"  domain={[0, 1.4]} label={{ value: "Disparate Impact", angle: -90, position: "insideLeft", fontSize: 11 }} />
                    <YAxis yAxisId="right" orientation="right" domain={[0.6, 0.75]} label={{ value: "Accuracy", angle: 90, position: "insideRight", fontSize: 11 }} />
                    <Tooltip contentStyle={{ borderRadius: 8, border: "none", boxShadow: "0 4px 12px rgba(0,0,0,0.1)" }} />
                    <Legend verticalAlign="top" />
                    <ReferenceLine yAxisId="left" y={0.8} stroke={CORAL} strokeDasharray="4 4" />
                    <ReferenceLine yAxisId="left" y={1.0} stroke={NAVY}  strokeDasharray="4 4" />
                    <Line yAxisId="left"  type="monotone" dataKey="disparate_impact" name="Disparate Impact" stroke={CORAL} strokeWidth={2.5} dot={{ r: 4, fill: CORAL }} />
                    <Line yAxisId="right" type="monotone" dataKey="accuracy"         name="Accuracy"         stroke={NAVY}  strokeWidth={2.5} dot={{ r: 4, fill: NAVY  }} strokeDasharray="5 5" />
                  </LineChart>
                </ResponsiveContainer>
                <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 16, fontSize: 13 }}>
                  <thead><tr style={{ background: BG }}>
                    <th style={thSt}>Poison %</th>
                    <th style={thSt}>Disparate Impact</th>
                    <th style={thSt}>Accuracy</th>
                    <th style={thSt}>Passes 0.8 Threshold?</th>
                  </tr></thead>
                  <tbody>
                    {POISON_RACE.map(r => (
                      <tr key={r.pct}>
                        <td style={{ ...tdSt, fontWeight: 700 }}>{r.pct}</td>
                        <td style={{ ...tdSt, color: r.disparate_impact >= 0.8 ? NAVY : CORAL, fontWeight: 700 }}>{r.disparate_impact}</td>
                        <td style={{ ...tdSt, color: NAVY, fontWeight: 700 }}>{r.accuracy}</td>
                        <td style={{ ...tdSt, color: r.disparate_impact >= 0.8 ? NAVY : MUTED, fontWeight: 700 }}>{r.disparate_impact >= 0.8 ? "Yes" : "No"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Experiment 3 — sensitive attribute inference */}
              <div style={card}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  <div style={{ fontWeight: 800, color: TEXT, fontSize: 16 }}>Experiment 3 — Sensitive Attribute Inference</div>
                  <span style={{ background: `${BLUE}33`, color: NAVY, fontSize: 11, fontWeight: 700, padding: "3px 10px", borderRadius: 6 }}>Verified</span>
                </div>
                <p style={{ color: MUTED, fontSize: 13, lineHeight: 1.7, margin: "0 0 8px" }}>
                  We trained a logistic regression classifier to predict the protected attribute (sex) using only the non-protected features — age, occupation, education, hours-per-week, capital gain, etc. If the model can predict sex from these features, then removing sex from training data provides no real privacy protection.
                </p>
                <div style={{ background: `${BLUE}22`, borderRadius: 8, padding: "10px 14px", marginBottom: 16, border: `1px solid ${BLUE}55` }}>
                  <span style={{ fontSize: 13, color: NAVY, fontWeight: 700 }}>Result: </span>
                  <span style={{ fontSize: 13, color: TEXT }}>5-fold cross-validation accuracy of <strong>77.31% (±0.54%)</strong> for predicting sex without using sex as a feature. This substantially exceeds the random baseline of 50%, demonstrating that sex is strongly encoded in correlated features like occupation and marital status.</span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
                  {[
                    { label: "Prediction Accuracy", value: "77.31%", sub: "predicting sex from other features", color: CORAL },
                    { label: "Random Baseline",     value: "50.00%", sub: "expected if sex were uninferable",   color: MUTED },
                    { label: "Uplift over Baseline", value: "+27.31%", sub: "information leaked by correlated features", color: NAVY },
                  ].map(s => (
                    <div key={s.label} style={{ background: BG, borderRadius: 10, padding: "14px 16px" }}>
                      <div style={{ fontSize: 11, fontWeight: 700, color: MUTED, letterSpacing: 1, textTransform: "uppercase", marginBottom: 4 }}>{s.label}</div>
                      <div style={{ fontSize: 22, fontWeight: 800, color: s.color, marginBottom: 2 }}>{s.value}</div>
                      <div style={{ fontSize: 11, color: MUTED }}>{s.sub}</div>
                    </div>
                  ))}
                </div>
                <p style={{ color: MUTED, fontSize: 13, lineHeight: 1.7, margin: "16px 0 0" }}>
                  This result demonstrates that <strong>"fairness through unawareness"</strong> — removing protected attributes from the feature set — is an insufficient defense. The model effectively re-learns the protected attribute from proxy features. True fairness requires explicit algorithmic intervention such as the bias mitigation techniques demonstrated in the Lab section.
                </p>
              </div>

              {/* Privacy risks */}
              <div style={{ ...card, borderLeft: `4px solid ${BLUE}`, marginBottom: 0 }}>
                <div style={{ fontWeight: 800, color: TEXT, fontSize: 16, marginBottom: 10 }}>Privacy Risks in the COMPAS Dataset</div>
                <p style={{ color: MUTED, fontSize: 14, lineHeight: 1.7, margin: 0 }}>
                  The COMPAS dataset contains sensitive criminal justice data used in real US courts. Beyond fairness concerns, this dataset poses significant privacy risks. Membership inference attacks — where an adversary determines whether a specific individual's record was used in model training — are particularly concerning given the sensitivity of recidivism data. Re-identification is feasible using quasi-identifiers: combining age, charge type, jurisdiction, and number of prior offenses with publicly available court records can uniquely identify individuals in the dataset, violating their privacy even when names are removed.
                </p>
              </div>

            </div>
          )}

        </div>
      </div>
    </>
  )
}