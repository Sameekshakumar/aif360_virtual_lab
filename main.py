from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from fastapi.responses import FileResponse
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunRequest(BaseModel):
    dataset: str
    protected_attr: str

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "data")

def load_dataset(dataset: str, protected_attr: str):
    from aif360.datasets import BinaryLabelDataset

    if dataset == "adult":
        cols = ["age","workclass","fnlwgt","education","education-num","marital-status",
                "occupation","relationship","race","sex","capital-gain","capital-loss",
                "hours-per-week","native-country","income"]
        df1 = pd.read_csv(f"{BASE}/adult/adult.data", header=None, names=cols, skipinitialspace=True, na_values="?")
        df2 = pd.read_csv(f"{BASE}/adult/adult.test", header=None, names=cols, skipinitialspace=True, na_values="?")
        df  = pd.concat([df1, df2], ignore_index=True).dropna()
        df["income"] = df["income"].str.strip().str.replace(".", "", regex=False)
        df["income"] = (df["income"] == ">50K").astype(float)
        df["sex"]    = (df["sex"] == "Male").astype(float)
        df["race"]   = (df["race"] == "White").astype(float)
        for c in ["workclass","education","marital-status","occupation","relationship","native-country"]:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        label = "income"

    elif dataset == "german":
        cols = ["status","duration","credit_history","purpose","credit_amount","savings",
                "employment","installment_rate","personal_status","other_debtors","residence",
                "property","age","other_installments","housing","existing_credits","job",
                "dependents","telephone","foreign_worker","credit"]
        df = pd.read_csv(f"{BASE}/german/german.data", sep=" ", header=None, names=cols)
        df["credit"] = (df["credit"] == 1).astype(float)
        df["sex"]    = df["personal_status"].apply(lambda x: 1.0 if x in ["A91","A93","A94"] else 0.0)
        df["age"]    = (df["age"] >= 25).astype(float)
        df = df.drop(columns=["personal_status"])
        for c in df.select_dtypes(include="object").columns:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        label = "credit"

    elif dataset == "compas":
        df = pd.read_csv(f"{BASE}/compas/compas-scores-two-years.csv")
        df = df[["sex","race","age","juv_fel_count","juv_misd_count","priors_count",
                 "c_charge_degree","two_year_recid"]].dropna()
        df["two_year_recid"]  = (df["two_year_recid"] == 0).astype(float)
        df["sex"]  = (df["sex"] == "Male").astype(float)
        df["race"] = (df["race"] == "Caucasian").astype(float)
        df["c_charge_degree"] = LabelEncoder().fit_transform(df["c_charge_degree"].astype(str))
        label = "two_year_recid"

    else:
        return None, None, None, None

    privileged_groups   = [{protected_attr: 1.0}]
    unprivileged_groups = [{protected_attr: 0.0}]

    dataset_obj = BinaryLabelDataset(
        df=df,
        label_names=[label],
        protected_attribute_names=[protected_attr],
        favorable_label=1.0,
        unfavorable_label=0.0,
        privileged_protected_attributes=[[1.0]],
        unprivileged_protected_attributes=[[0.0]]
    )
    return dataset_obj, privileged_groups, unprivileged_groups, label


@app.post("/run-reweighing")
def run_reweighing(req: RunRequest):
    dataset_orig, privileged_groups, unprivileged_groups, label = load_dataset(req.dataset, req.protected_attr)
    if dataset_orig is None:
        return {"error": "invalid dataset"}

    np.random.seed(1)
    train, vt   = dataset_orig.split([0.7], shuffle=True)
    valid, test = vt.split([0.5], shuffle=True)

    metric_before    = BinaryLabelDatasetMetric(train, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    mean_diff_before = metric_before.mean_difference()

    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    RW.fit(train)
    train_transf = RW.transform(train)

    metric_after  = BinaryLabelDatasetMetric(train_transf, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    mean_diff_after = metric_after.mean_difference()

    scaler_orig = StandardScaler()
    X_tr = scaler_orig.fit_transform(train.features)
    lmod_orig = LogisticRegression(max_iter=500)
    lmod_orig.fit(X_tr, train.labels.ravel(), sample_weight=train.instance_weights)

    pos_ind    = np.where(lmod_orig.classes_ == train.favorable_label)[0][0]
    X_val      = scaler_orig.transform(valid.features)
    scores_val = lmod_orig.predict_proba(X_val)[:, pos_ind]

    best_thresh, best_ba = 0.5, 0
    for t in np.linspace(0.01, 0.99, 100):
        preds = valid.copy(deepcopy=True)
        preds.labels = np.where(scores_val > t, valid.favorable_label, valid.unfavorable_label).reshape(-1,1)
        cm = ClassificationMetric(valid, preds, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        ba = 0.5 * (cm.true_positive_rate() + cm.true_negative_rate())
        if ba > best_ba:
            best_ba, best_thresh = ba, t

    X_test      = scaler_orig.transform(test.features)
    scores_orig = lmod_orig.predict_proba(X_test)[:, pos_ind]
    test_pred_orig = test.copy(deepcopy=True)
    test_pred_orig.labels = np.where(scores_orig > best_thresh, test.favorable_label, test.unfavorable_label).reshape(-1,1)
    cm_orig = ClassificationMetric(test, test_pred_orig, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    scaler_transf = StandardScaler()
    X_tr2 = scaler_transf.fit_transform(train_transf.features)
    lmod_transf = LogisticRegression(max_iter=500)
    lmod_transf.fit(X_tr2, train_transf.labels.ravel(), sample_weight=train_transf.instance_weights)

    X_test2       = scaler_transf.transform(test.features)
    scores_transf = lmod_transf.predict_proba(X_test2)[:, pos_ind]
    test_pred_transf = test.copy(deepcopy=True)
    test_pred_transf.labels = np.where(scores_transf > best_thresh, test.favorable_label, test.unfavorable_label).reshape(-1,1)
    cm_transf = ClassificationMetric(test, test_pred_transf, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    return {
        "dataset": req.dataset,
        "protected_attr": req.protected_attr,
        "dataset_bias": {
            "mean_diff_before": round(mean_diff_before, 4),
            "mean_diff_after":  round(mean_diff_after, 4),
        },
        "before_reweighing": {
            "balanced_accuracy":       round(0.5*(cm_orig.true_positive_rate()+cm_orig.true_negative_rate()), 4),
            "disparate_impact":        round(cm_orig.disparate_impact(), 4),
            "avg_odds_difference":     round(cm_orig.average_odds_difference(), 4),
            "equal_opp_difference":    round(cm_orig.equal_opportunity_difference(), 4),
            "statistical_parity_diff": round(cm_orig.statistical_parity_difference(), 4),
        },
        "after_reweighing": {
            "balanced_accuracy":       round(0.5*(cm_transf.true_positive_rate()+cm_transf.true_negative_rate()), 4),
            "disparate_impact":        round(cm_transf.disparate_impact(), 4),
            "avg_odds_difference":     round(cm_transf.average_odds_difference(), 4),
            "equal_opp_difference":    round(cm_transf.equal_opportunity_difference(), 4),
            "statistical_parity_diff": round(cm_transf.statistical_parity_difference(), 4),
        },
        "optimal_threshold": round(best_thresh, 4),
    }


@app.post("/run-adversarial-debiasing")
def run_adversarial_debiasing(req: RunRequest):
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
    from sklearn.preprocessing import MaxAbsScaler

    dataset_orig, privileged_groups, unprivileged_groups, label = load_dataset(req.dataset, req.protected_attr)
    if dataset_orig is None:
        return {"error": "invalid dataset"}

    np.random.seed(1)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

    scaler = MaxAbsScaler()
    dataset_orig_train.features = scaler.fit_transform(dataset_orig_train.features)
    dataset_orig_test.features  = scaler.transform(dataset_orig_test.features)

    metric_orig      = BinaryLabelDatasetMetric(dataset_orig_train, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    mean_diff_before = metric_orig.mean_difference()

    tf.reset_default_graph()
    sess = tf.Session()
    plain_model = AdversarialDebiasing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups,
                                       scope_name='plain_classifier', debias=False, sess=sess)
    plain_model.fit(dataset_orig_train)
    dataset_plain_test = plain_model.predict(dataset_orig_test)
    sess.close()

    cm_plain = ClassificationMetric(dataset_orig_test, dataset_plain_test,
                                    unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    TPR = cm_plain.true_positive_rate()
    TNR = cm_plain.true_negative_rate()

    tf.reset_default_graph()
    sess2 = tf.Session()
    debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups,
                                          scope_name='debiased_classifier', debias=True, sess=sess2)
    debiased_model.fit(dataset_orig_train)
    dataset_debiased_test = debiased_model.predict(dataset_orig_test)
    sess2.close()

    cm_debiased = ClassificationMetric(dataset_orig_test, dataset_debiased_test,
                                       unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    TPR2 = cm_debiased.true_positive_rate()
    TNR2 = cm_debiased.true_negative_rate()

    metric_debiased = BinaryLabelDatasetMetric(dataset_debiased_test,
                                               unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    return {
        "dataset": req.dataset,
        "protected_attr": req.protected_attr,
        "dataset_bias": {
            "mean_diff_before": round(mean_diff_before, 4),
            "mean_diff_after":  round(metric_debiased.mean_difference(), 4),
        },
        "before_debiasing": {
            "balanced_accuracy":       round(0.5*(TPR+TNR), 4),
            "disparate_impact":        round(cm_plain.disparate_impact(), 4),
            "avg_odds_difference":     round(cm_plain.average_odds_difference(), 4),
            "equal_opp_difference":    round(cm_plain.equal_opportunity_difference(), 4),
            "statistical_parity_diff": round(cm_plain.statistical_parity_difference(), 4),
            "theil_index":             round(cm_plain.theil_index(), 4),
        },
        "after_debiasing": {
            "balanced_accuracy":       round(0.5*(TPR2+TNR2), 4),
            "disparate_impact":        round(cm_debiased.disparate_impact(), 4),
            "avg_odds_difference":     round(cm_debiased.average_odds_difference(), 4),
            "equal_opp_difference":    round(cm_debiased.equal_opportunity_difference(), 4),
            "statistical_parity_diff": round(cm_debiased.statistical_parity_difference(), 4),
            "theil_index":             round(cm_debiased.theil_index(), 4),
        }
    }


@app.post("/run-disparate-impact-remover")
def run_disparate_impact_remover(req: RunRequest):
    from aif360.algorithms.preprocessing import DisparateImpactRemover
    from sklearn.preprocessing import MinMaxScaler

    protected = req.protected_attr
    dataset_orig, privileged_groups, unprivileged_groups, label = load_dataset(req.dataset, req.protected_attr)
    if dataset_orig is None:
        return {"error": "invalid dataset"}

    test, train = dataset_orig.split([0.5], shuffle=True)

    scaler = MinMaxScaler(copy=False)
    train.features = scaler.fit_transform(train.features)
    test.features  = scaler.fit_transform(test.features)

    index = train.feature_names.index(protected)

    results = []
    for level in np.linspace(0., 1., 11):
        di = DisparateImpactRemover(repair_level=level, sensitive_attribute=protected)
        train_repd = di.fit_transform(train)
        test_repd  = di.fit_transform(test)

        X_tr = np.delete(train_repd.features, index, axis=1)
        X_te = np.delete(test_repd.features,  index, axis=1)
        y_tr = train_repd.labels.ravel()

        lmod = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=500)
        lmod.fit(X_tr, y_tr)

        test_repd_pred = test_repd.copy(deepcopy=True)
        test_repd_pred.labels = lmod.predict(X_te).reshape(-1, 1)

        cm = BinaryLabelDatasetMetric(test_repd_pred, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
        results.append({
            "repair_level":     round(float(level), 2),
            "disparate_impact": round(cm.disparate_impact(), 4)
        })

    return {
        "dataset": req.dataset,
        "protected_attr": req.protected_attr,
        "repair_levels": results
    }


@app.post("/run-exponentiated-gradient")
def run_exponentiated_gradient(req: RunRequest):
    from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction
    from sklearn.preprocessing import MaxAbsScaler

    dataset_orig, privileged_groups, unprivileged_groups, label = load_dataset(req.dataset, req.protected_attr)
    if dataset_orig is None:
        return {"error": "invalid dataset"}

    np.random.seed(0)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

    scaler = MaxAbsScaler()
    dataset_orig_train.features = scaler.fit_transform(dataset_orig_train.features)
    dataset_orig_test.features  = scaler.transform(dataset_orig_test.features)

    metric_orig      = BinaryLabelDatasetMetric(dataset_orig_train, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    mean_diff_before = metric_orig.mean_difference()

    lmod = LogisticRegression(solver='lbfgs', max_iter=1000)
    lmod.fit(dataset_orig_train.features, dataset_orig_train.labels.ravel(),
             sample_weight=dataset_orig_train.instance_weights)
    y_pred = lmod.predict(dataset_orig_test.features)

    dataset_baseline_pred = dataset_orig_test.copy(deepcopy=True)
    dataset_baseline_pred.labels = y_pred.reshape(-1, 1)

    cm_baseline = ClassificationMetric(dataset_orig_test, dataset_baseline_pred,
                                       unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    estimator = LogisticRegression(solver='lbfgs', max_iter=1000)
    np.random.seed(0)
    egr = ExponentiatedGradientReduction(estimator=estimator, constraints="EqualizedOdds", drop_prot_attr=False)
    egr.fit(dataset_orig_train)
    egr_pred = egr.predict(dataset_orig_test)

    cm_egr = ClassificationMetric(dataset_orig_test, egr_pred,
                                  unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    TPR_b = cm_baseline.true_positive_rate()
    TNR_b = cm_baseline.true_negative_rate()
    TPR_e = cm_egr.true_positive_rate()
    TNR_e = cm_egr.true_negative_rate()

    return {
        "dataset": req.dataset,
        "protected_attr": req.protected_attr,
        "dataset_bias": {
            "mean_diff_before": round(mean_diff_before, 4),
            "mean_diff_after":  round(cm_egr.statistical_parity_difference(), 4),
        },
        "before_egr": {
            "balanced_accuracy":       round(0.5*(TPR_b+TNR_b), 4),
            "disparate_impact":        round(cm_baseline.disparate_impact(), 4),
            "avg_odds_difference":     round(cm_baseline.average_odds_difference(), 4),
            "equal_opp_difference":    round(cm_baseline.equal_opportunity_difference(), 4),
            "statistical_parity_diff": round(cm_baseline.statistical_parity_difference(), 4),
        },
        "after_egr": {
            "balanced_accuracy":       round(0.5*(TPR_e+TNR_e), 4),
            "disparate_impact":        round(cm_egr.disparate_impact(), 4),
            "avg_odds_difference":     round(cm_egr.average_odds_difference(), 4),
            "equal_opp_difference":    round(cm_egr.equal_opportunity_difference(), 4),
            "statistical_parity_diff": round(cm_egr.statistical_parity_difference(), 4),
        }
    }

DATA_PATHS = {
    "adult":  f"{BASE}/adult/adult.data",
    "german": f"{BASE}/german/german.data",
    "compas": f"{BASE}/compas/compas-scores-two-years.csv",
}

ADULT_COLS = ["age","workclass","fnlwgt","education","education-num","marital-status",
              "occupation","relationship","race","sex","capital-gain","capital-loss",
              "hours-per-week","native-country","income"]

@app.get("/dataset-preview/{dataset}")
def dataset_preview(dataset: str):
    try:
        if dataset == "adult":
            df = pd.read_csv(DATA_PATHS["adult"], header=None, names=ADULT_COLS,
                             skipinitialspace=True, na_values="?").dropna().head(8)
        elif dataset == "german":
            cols = ["status","duration","credit_history","purpose","credit_amount","savings",
                    "employment","installment_rate","personal_status","other_debtors","residence",
                    "property","age","other_installments","housing","existing_credits","job",
                    "dependents","telephone","foreign_worker","credit"]
            df = pd.read_csv(DATA_PATHS["german"], sep=" ", header=None, names=cols).head(8)
        elif dataset == "compas":
            df = pd.read_csv(DATA_PATHS["compas"])[
                ["name","sex","race","age","juv_fel_count","priors_count","c_charge_degree","two_year_recid"]
            ].head(8)
        else:
            return {"error": "unknown dataset"}
        return {"columns": df.columns.tolist(), "rows": df.values.tolist()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/dataset-download/{dataset}")
def dataset_download(dataset: str):
    paths = {
        "adult":  (DATA_PATHS["adult"],  "adult_census.csv"),
        "german": (DATA_PATHS["german"], "german_credit.csv"),
        "compas": (DATA_PATHS["compas"], "compas_recidivism.csv"),
    }
    if dataset not in paths:
        return {"error": "unknown dataset"}
    path, filename = paths[dataset]
    return FileResponse(
        path,
        media_type="text/csv",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/algo-data-diff/{dataset}/{protected_attr}")
def algo_data_diff(dataset: str, protected_attr: str):
    """Returns before/after rows for reweighing and disparate impact remover"""
    try:
        dataset_orig, privileged_groups, unprivileged_groups, label = load_dataset(dataset, protected_attr)
        if dataset_orig is None:
            return {"error": "invalid dataset"}

        np.random.seed(1)
        train, _ = dataset_orig.split([0.7], shuffle=True)

        # Before reweighing — show raw weights and features
        before_rows = []
        for i in range(6):
            row = {name: round(float(train.features[i][j]), 3)
                   for j, name in enumerate(train.feature_names)}
            row["label"] = train.labels[i][0]
            row["weight"] = round(float(train.instance_weights[i]), 4)
            before_rows.append(row)

        # After reweighing — same rows but with new weights
        RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        RW.fit(train)
        train_rw = RW.transform(train)

        after_rows_rw = []
        for i in range(6):
            row = {name: round(float(train_rw.features[i][j]), 3)
                   for j, name in enumerate(train_rw.feature_names)}
            row["label"] = train_rw.labels[i][0]
            row["weight"] = round(float(train_rw.instance_weights[i]), 4)
            after_rows_rw.append(row)

        # After disparate impact remover — show changed feature values
        from aif360.algorithms.preprocessing import DisparateImpactRemover
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(copy=False)
        train_di = dataset_orig.split([0.7], shuffle=True)[0]
        train_di.features = scaler.fit_transform(train_di.features)
        di = DisparateImpactRemover(repair_level=1.0, sensitive_attribute=protected_attr)
        train_di_repaired = di.fit_transform(train_di)

        after_rows_di = []
        for i in range(6):
            row = {name: round(float(train_di_repaired.features[i][j]), 3)
                   for j, name in enumerate(train_di_repaired.feature_names)}
            row["label"] = train_di_repaired.labels[i][0]
            after_rows_di.append(row)

        before_rows_di = []
        for i in range(6):
            row = {name: round(float(train_di.features[i][j]), 3)
                   for j, name in enumerate(train_di.feature_names)}
            row["label"] = train_di.labels[i][0]
            before_rows_di.append(row)

        # pick 4 readable columns to display
        display_cols = [protected_attr, "age", "education-num" if "education-num" in train.feature_names else train.feature_names[1], label]
        display_cols = [c for c in display_cols if c in (train.feature_names + [label])][:4]

        return {
            "feature_names": train.feature_names,
            "label": label,
            "protected_attr": protected_attr,
            "reweighing": {"before": before_rows, "after": after_rows_rw},
            "di_remover": {"before": before_rows_di, "after": after_rows_di},
            "display_cols": display_cols,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {"status": "AIF360 API running"}