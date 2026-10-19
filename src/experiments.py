"""Staged experiments: each stage adds one kind of information and is scored the same way.

Models are selected and early-stopped on the validation set only; the held-out test set is scored
once per stage and never used to make a choice. Results are cached in data/cache so the notebooks
re-run in seconds (delete data/cache/stage_*.pkl to retrain).
"""
import pickle
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

import features as F

ORIGINAL_PARAMS = dict(iterations=1000, depth=4, eval_metric="AUC", verbose=0, random_seed=0)
FINAL_PARAMS = dict(iterations=3000, depth=6, learning_rate=0.05, eval_metric="AUC",
                    early_stopping_rounds=200, verbose=0, random_seed=0, thread_count=8)

STAGES = {
    "S1": "Original 17 hand-picked features",
    "S2": "+ all application columns (no gender)",
    "S3": "+ engineered ratios and fixes",
    "S4": "+ bureau, installment and credit-card history",
}


def load_splits():
    df = F.load_application()
    return F.make_splits(df)


def build_stage(frame: pd.DataFrame, stage: str, drop_sensitive: bool = True):
    """Feature matrix and categorical column names for one stage."""
    frame = frame.reset_index(drop=True)
    if stage == "S1":
        out = F.baseline_features(frame, F.bureau_max_overdue())
        return out[F.BASELINE_FEATURES], []
    X, cats = F.application_columns(frame, drop_sensitive=drop_sensitive)
    if stage in ("S3", "S4"):
        X = pd.concat([X, F.engineered_features(frame).drop(columns="SK_ID_CURR")], axis=1)
    if stage == "S4":
        X = X.merge(F.history_features(frame["SK_ID_CURR"]), on="SK_ID_CURR", how="left")
    return X.drop(columns="SK_ID_CURR"), cats


def run_stage(stage: str, params: dict, tag: str, drop_sensitive: bool = True, force: bool = False):
    path = F.CACHE / f"stage_{tag}.pkl"
    if path.exists() and not force:
        return pickle.loads(path.read_bytes())
    F.CACHE.mkdir(parents=True, exist_ok=True)
    train, valid, test = load_splits()
    Xtr, cats = build_stage(train, stage, drop_sensitive)
    Xva, _ = build_stage(valid, stage, drop_sensitive)
    Xte, _ = build_stage(test, stage, drop_sensitive)
    ytr, yva, yte = train["TARGET"].values, valid["TARGET"].values, test["TARGET"].values

    start = time.time()
    model = CatBoostClassifier(**params)
    model.fit(Pool(Xtr, ytr, cat_features=cats), eval_set=Pool(Xva, yva, cat_features=cats), use_best_model=True)
    pred_va = model.predict_proba(Pool(Xva, cat_features=cats))[:, 1]
    pred_te = model.predict_proba(Pool(Xte, cat_features=cats))[:, 1]
    result = {
        "stage": stage, "tag": tag, "n_features": Xtr.shape[1],
        "val_auc": roc_auc_score(yva, pred_va), "test_auc": roc_auc_score(yte, pred_te),
        "best_iteration": int(model.get_best_iteration()), "seconds": time.time() - start,
        "y_test": yte, "pred_test": pred_te,
        "importance": dict(zip(Xtr.columns, model.get_feature_importance())),
    }
    path.write_bytes(pickle.dumps(result))
    return result


def run_decision_tree():
    path = F.CACHE / "stage_tree.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    F.CACHE.mkdir(parents=True, exist_ok=True)
    train, valid, test = load_splits()
    Xtr, _ = build_stage(train, "S1")
    Xva, _ = build_stage(valid, "S1")
    Xte, _ = build_stage(test, "S1")
    tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xtr, train["TARGET"])
    result = {
        "val_auc": roc_auc_score(valid["TARGET"], tree.predict_proba(Xva)[:, 1]),
        "test_auc": roc_auc_score(test["TARGET"], tree.predict_proba(Xte)[:, 1]),
    }
    path.write_bytes(pickle.dumps(result))
    return result


def run_all():
    rows = [("Decision tree, depth 4 (original baseline)", "S1", run_decision_tree())]
    rows.append(("CatBoost, original settings (original baseline)", "S1",
                 run_stage("S1", ORIGINAL_PARAMS, "S1_original", drop_sensitive=False)))
    for stage, label in STAGES.items():
        rows.append((label, stage, run_stage(stage, FINAL_PARAMS, f"{stage}_final", drop_sensitive=(stage != "S1"))))
    return rows


if __name__ == "__main__":
    for label, stage, r in run_all():
        print(f"{stage}  val {r['val_auc']:.4f}  test {r['test_auc']:.4f}  {label}", flush=True)
    r = run_stage("S4", FINAL_PARAMS, "S4_with_gender", drop_sensitive=False)
    print(f"S4 with gender: val {r['val_auc']:.4f} test {r['test_auc']:.4f}", flush=True)
