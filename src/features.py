"""Data loading, the train/validation/test split and feature builders for the Home Credit project.

Everything reads from data/raw (the Kaggle files, see data/README.md) and caches the slow
aggregations in data/cache, so re-running is quick.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"

# Columns the original baseline used (17 features chosen by hand during exploration).
BASELINE_FEATURES = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE_COM", "NAME_INCOME_TYPE_PEN", "NAME_INCOME_TYPE_STA", "NAME_INCOME_TYPE_WOR",
    "DAYS_BIRTH", "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "FLAG_EMP_PHONE", "REG_CITY_NOT_LIVE_CITY", "REGION_RATING_CLIENT_W_CITY", "CREDIT_DAY_OVERDUE",
]
EDUCATION_ORDER = {
    "Lower secondary": 0, "Secondary / secondary special": 1, "Incomplete higher": 2,
    "Higher education": 3, "Academic degree": 4,
}
# Protected attribute: dropped from the main models (see the README).
SENSITIVE = ["CODE_GENDER"]


def load_application() -> pd.DataFrame:
    return pd.read_csv(RAW / "application_train.csv")


def make_splits(df: pd.DataFrame):
    """Same split as the original exploration: the 10% highest SK_ID_CURR are the held-out test set,
    10% of all rows (random, from the rest) are validation, and the remainder is training."""
    n = int(0.1 * len(df))
    ordered = df.sort_values("SK_ID_CURR")
    test = ordered.iloc[-n:]
    rest = shuffle(ordered.iloc[:-n], random_state=0)
    return rest.iloc[:-n].copy(), rest.iloc[-n:].copy(), test.copy()


# --- Stage 1: the original hand-picked features -------------------------------------------------

def baseline_features(df: pd.DataFrame, bureau_overdue: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CODE_GENDER"] = (out["CODE_GENDER"] == "F").astype(int)
    out["FLAG_OWN_CAR"] = (out["FLAG_OWN_CAR"] == "Y").astype(int)
    out["FLAG_OWN_REALTY"] = (out["FLAG_OWN_REALTY"] == "Y").astype(int)
    out["NAME_EDUCATION_TYPE"] = out["NAME_EDUCATION_TYPE"].map(EDUCATION_ORDER)
    for tag, value in [("COM", "Commercial associate"), ("PEN", "Pensioner"),
                       ("STA", "State servant"), ("WOR", "Working")]:
        out[f"NAME_INCOME_TYPE_{tag}"] = (out["NAME_INCOME_TYPE"] == value).astype(int)
    out["DAYS_BIRTH"] = (out["DAYS_BIRTH"] / 365).astype(int).abs()  # age in years
    out = out.merge(bureau_overdue, on="SK_ID_CURR", how="left")
    out["CREDIT_DAY_OVERDUE"] = out["CREDIT_DAY_OVERDUE"].fillna(0)
    out["AMT_ANNUITY"] = out["AMT_ANNUITY"].fillna(0)
    return out[["SK_ID_CURR", "TARGET"] + BASELINE_FEATURES]


def bureau_max_overdue() -> pd.DataFrame:
    bureau = pd.read_csv(RAW / "bureau.csv", usecols=["SK_ID_CURR", "CREDIT_DAY_OVERDUE"])
    return bureau.groupby("SK_ID_CURR", as_index=False).agg({"CREDIT_DAY_OVERDUE": "max"})


# --- Stage 2: every application column ---------------------------------------------------------

def application_columns(df: pd.DataFrame, drop_sensitive: bool = True):
    """All application columns as-is (categoricals stay strings for CatBoost). Returns (frame, categorical names)."""
    out = df.drop(columns=["TARGET"])
    if drop_sensitive:
        out = out.drop(columns=SENSITIVE)
    cats = [c for c in out.columns if out[c].dtype == object or str(out[c].dtype) in ("str", "string")]
    out[cats] = out[cats].fillna("missing").astype(str)
    return out, cats


# --- Stage 3: engineered application features --------------------------------------------------

def engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios and fixes on top of the raw application columns."""
    out = pd.DataFrame({"SK_ID_CURR": df["SK_ID_CURR"].values})
    income, credit, annuity = df["AMT_INCOME_TOTAL"], df["AMT_CREDIT"], df["AMT_ANNUITY"]
    out["CREDIT_INCOME_RATIO"] = (credit / income).values
    out["ANNUITY_INCOME_RATIO"] = (annuity / income).values
    out["CREDIT_TERM"] = (credit / annuity).values  # roughly: number of payments
    out["GOODS_CREDIT_RATIO"] = (df["AMT_GOODS_PRICE"] / credit).values
    out["INCOME_PER_PERSON"] = (income / df["CNT_FAM_MEMBERS"].replace(0, np.nan)).values
    # 365243 days (~1000 years) is a placeholder for "not employed", mostly pensioners.
    employed = df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"] != 365243)
    out["DAYS_EMPLOYED_CLEAN"] = employed.values
    out["NOT_EMPLOYED_FLAG"] = (df["DAYS_EMPLOYED"] == 365243).astype(int).values
    out["EMPLOYED_TO_AGE"] = (employed / df["DAYS_BIRTH"]).values
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    out["EXT_SOURCE_MEAN"] = ext.mean(axis=1).values
    out["EXT_SOURCE_MIN"] = ext.min(axis=1).values
    out["EXT_SOURCE_PROD"] = ext.prod(axis=1, skipna=False).values
    out["EXT_SOURCE_COUNT"] = ext.notna().sum(axis=1).values
    out["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).values
    out["PHONE_CHANGE_RECENCY"] = (df["DAYS_LAST_PHONE_CHANGE"]).values
    return out


# --- Stage 4: the customer's other loans -------------------------------------------------------

def _cached(name, build):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frame = build()
    frame.to_parquet(path)
    return frame


def bureau_aggregates() -> pd.DataFrame:
    def build():
        b = pd.read_csv(RAW / "bureau.csv", usecols=[
            "SK_ID_CURR", "CREDIT_ACTIVE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "CNT_CREDIT_PROLONG",
            "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE"])
        b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
        g = b.groupby("SK_ID_CURR")
        out = pd.DataFrame({
            "BUREAU_N_CREDITS": g.size(),
            "BUREAU_N_ACTIVE": g["IS_ACTIVE"].sum(),
            "BUREAU_DAYS_CREDIT_MEAN": g["DAYS_CREDIT"].mean(),
            "BUREAU_DAYS_CREDIT_MIN": g["DAYS_CREDIT"].min(),
            "BUREAU_MAX_OVERDUE_DAYS": g["CREDIT_DAY_OVERDUE"].max(),
            "BUREAU_PROLONGED_SUM": g["CNT_CREDIT_PROLONG"].sum(),
            "BUREAU_CREDIT_SUM": g["AMT_CREDIT_SUM"].sum(),
            "BUREAU_DEBT_SUM": g["AMT_CREDIT_SUM_DEBT"].sum(),
            "BUREAU_OVERDUE_SUM": g["AMT_CREDIT_SUM_OVERDUE"].sum(),
        })
        out["BUREAU_DEBT_RATIO"] = out["BUREAU_DEBT_SUM"] / out["BUREAU_CREDIT_SUM"].replace(0, np.nan)
        return out.reset_index()
    return _cached("bureau_agg", build)


def installments_aggregates() -> pd.DataFrame:
    def build():
        p = pd.read_csv(RAW / "installments_payments.csv", usecols=[
            "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"],
            dtype={"SK_ID_CURR": "int32", "DAYS_INSTALMENT": "float32", "DAYS_ENTRY_PAYMENT": "float32",
                   "AMT_INSTALMENT": "float32", "AMT_PAYMENT": "float32"})
        p["DAYS_LATE"] = (p["DAYS_ENTRY_PAYMENT"] - p["DAYS_INSTALMENT"]).clip(lower=0)
        p["IS_LATE"] = (p["DAYS_LATE"] > 0).astype("float32")
        p["PAID_RATIO"] = p["AMT_PAYMENT"] / p["AMT_INSTALMENT"].replace(0, np.nan)
        p["UNDERPAID"] = (p["AMT_INSTALMENT"] - p["AMT_PAYMENT"]).clip(lower=0)
        recent = p[p["DAYS_INSTALMENT"] > -365]
        g = p.groupby("SK_ID_CURR")
        out = pd.DataFrame({
            "INST_COUNT": g.size(),
            "INST_DAYS_LATE_MEAN": g["DAYS_LATE"].mean(),
            "INST_DAYS_LATE_MAX": g["DAYS_LATE"].max(),
            "INST_LATE_SHARE": g["IS_LATE"].mean(),
            "INST_PAID_RATIO_MEAN": g["PAID_RATIO"].mean(),
            "INST_UNDERPAID_SUM": g["UNDERPAID"].sum(),
        })
        out["INST_LATE_SHARE_LAST_YEAR"] = recent.groupby("SK_ID_CURR")["IS_LATE"].mean()
        return out.reset_index()
    return _cached("installments_agg", build)


def credit_card_aggregates() -> pd.DataFrame:
    def build():
        c = pd.read_csv(RAW / "credit_card_balance.csv", usecols=[
            "SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"])
        c["UTILIZATION"] = c["AMT_BALANCE"] / c["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
        c["HAS_DPD"] = (c["SK_DPD"] > 0).astype(int)
        g = c.groupby("SK_ID_CURR")
        out = pd.DataFrame({
            "CC_MONTHS": g.size(),
            "CC_BALANCE_MEAN": g["AMT_BALANCE"].mean(),
            "CC_UTILIZATION_MEAN": g["UTILIZATION"].mean(),
            "CC_UTILIZATION_MAX": g["UTILIZATION"].max(),
            "CC_DPD_MAX": g["SK_DPD"].max(),
            "CC_DPD_SHARE": g["HAS_DPD"].mean(),
        })
        return out.reset_index()
    return _cached("credit_card_agg", build)


def history_features(ids: pd.Series) -> pd.DataFrame:
    """All other-loan aggregates, one row per requested SK_ID_CURR (missing history stays NaN)."""
    base = pd.DataFrame({"SK_ID_CURR": ids.values})
    for table in (bureau_aggregates(), installments_aggregates(), credit_card_aggregates()):
        base = base.merge(table, on="SK_ID_CURR", how="left")
    return base
