import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, silhouette_score
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    XGB_AVAILABLE = False

RANDOM_STATE = 42

st.set_page_config(
    page_title="Rajasthan Climate Risk & Heatwave Simulator",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Rajasthan Climate Risk Analysis")
st.subheader("Machine Learning-Based Heatwave Prediction & 2035 Scenario Simulation")
st.markdown(
    "Explore historical heatwave patterns (2006–2025), estimate next-day heatwave "
    "risk, discover climate regimes, and compare hypothetical 2035 climate scenarios."
)

with st.sidebar:
    st.header("Project controls")
    uploaded_file = st.file_uploader(
        "Upload Rajasthan climate CSV",
        type=["csv"],
        help="Upload Rajasthan_Heatwave_2006_2025.csv"
    )
    st.caption("Expected columns include YEAR, MONTH, DAY, DISTRICT, HEATWAVE, "
               "LAT, LON and the climate variables used in the notebook.")
    page = st.radio(
        "Navigate",
        ["Climate Overview", "Next-Day Prediction", "Climate Regimes",
         "2035 Scenario Simulator", "District Risk"]
    )

DEFAULT_PATHS = [
    "Rajasthan_Heatwave_2006_2025.csv",
    "data/Rajasthan_Heatwave_2006_2025.csv",
    "data/rajasthan_heatwave_2006_2025.csv"
]

@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):
    if file_bytes is not None:
        import io
        return pd.read_csv(io.BytesIO(file_bytes))
    for path in DEFAULT_PATHS:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

import os
file_bytes = uploaded_file.getvalue() if uploaded_file else None
file_name = uploaded_file.name if uploaded_file else None
raw = load_data(file_bytes, file_name)

if raw is None:
    st.info("Upload your Rajasthan climate CSV using the sidebar to start.")
    st.markdown("**Expected dataset:** `Rajasthan_Heatwave_2006_2025.csv`")
    st.markdown("You can also place the CSV in a `data/` folder beside `app.py`.")
    st.stop()

df = raw.copy()
required = ["YEAR", "MONTH", "DAY", "DISTRICT", "HEATWAVE"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {', '.join(missing)}")
    st.stop()

df["DATE"] = pd.to_datetime(
    dict(year=df["YEAR"], month=df["MONTH"], day=df["DAY"]),
    errors="coerce"
)
df = df.dropna(subset=["DATE", "DISTRICT", "HEATWAVE"]).copy()
df["DISTRICT"] = df["DISTRICT"].astype(str)
df["HEATWAVE"] = pd.to_numeric(df["HEATWAVE"], errors="coerce")
df = df.dropna(subset=["HEATWAVE"])
df["HEATWAVE"] = df["HEATWAVE"].astype(int)
df = df.sort_values(["DISTRICT", "DATE"]).reset_index(drop=True)

for col in ["TEMP2M", "TMAX", "TMIN", "DEW2M", "SOILT1"]:
    if col in df.columns:
        df[f"{col}_C"] = pd.to_numeric(df[col], errors="coerce") - 273.15

if "LAT" not in df.columns:
    df["LAT"] = 0.0
if "LON" not in df.columns:
    df["LON"] = 0.0

st.sidebar.success(f"Loaded {len(df):,} rows · {df['DISTRICT'].nunique()} districts")

def make_model_data(source):
    work = source.sort_values(["DISTRICT", "DATE"]).copy()
    base_numeric = [
        "WIND_U10", "WIND_V10", "MSLP", "BLH", "GEOP",
        "TEMP2M_C", "TMAX_C", "TMIN_C", "DEW2M_C",
        "CLOUD", "RAIN", "SRAD", "EVAP", "SOILT1_C", "SOILM1", "LAI"
    ]
    absent = [c for c in base_numeric if c not in work.columns]
    if absent:
        raise ValueError("Missing climate columns needed by the notebook model: " + ", ".join(absent))

    work["HEATWAVE_NEXT_DAY"] = work.groupby("DISTRICT")["HEATWAVE"].shift(-1)
    for col in base_numeric:
        work[f"{col}_LAG1"] = work.groupby("DISTRICT")[col].shift(1)
        work[f"{col}_ROLL3"] = work.groupby("DISTRICT")[col].transform(
            lambda s: s.shift(1).rolling(3, min_periods=3).mean()
        )
        work[f"{col}_ROLL7"] = work.groupby("DISTRICT")[col].transform(
            lambda s: s.shift(1).rolling(7, min_periods=7).mean()
        )

    work["HEATWAVE_LAG1"] = work.groupby("DISTRICT")["HEATWAVE"].shift(1)
    work["HEATWAVE_ROLL7"] = work.groupby("DISTRICT")["HEATWAVE"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=7).mean()
    )
    doy = work["DATE"].dt.dayofyear
    work["SIN_DOY"] = np.sin(2 * np.pi * doy / 365.25)
    work["COS_DOY"] = np.cos(2 * np.pi * doy / 365.25)

    lag_features = [
        c for c in work.columns
        if c.endswith("_LAG1") or c.endswith("_ROLL3") or c.endswith("_ROLL7")
    ]
    features = list(dict.fromkeys(
        lag_features + ["HEATWAVE_LAG1", "HEATWAVE_ROLL7",
                        "SIN_DOY", "COS_DOY", "LAT", "LON", "DISTRICT"]
    ))
    work = work.dropna(subset=["HEATWAVE_NEXT_DAY"] + features).copy()
    work["HEATWAVE_NEXT_DAY"] = work["HEATWAVE_NEXT_DAY"].astype(int)
    return work, features

@st.cache_resource(show_spinner="Training the leakage-aware model...")
def train_model(data):
    model_df, features = make_model_data(data)
    train = model_df[model_df["DATE"] < "2022-01-01"].copy()
    test = model_df[model_df["DATE"] >= "2022-01-01"].copy()
    if train.empty or test.empty:
        raise ValueError("The dataset must contain observations both before 2022 and from 2022 onward.")
    if train["HEATWAVE_NEXT_DAY"].nunique() < 2:
        raise ValueError("Training data must contain both heatwave and non-heatwave target classes.")

    X_train, y_train = train[features], train["HEATWAVE_NEXT_DAY"]
    X_test, y_test = test[features], test["HEATWAVE_NEXT_DAY"]
    categorical = ["DISTRICT"]
    numeric = [c for c in features if c not in categorical]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric)
    ])
    weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    if XGB_AVAILABLE:
        estimator = XGBClassifier(
            n_estimators=220, max_depth=3, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            reg_lambda=2.0, eval_metric="logloss", random_state=RANDOM_STATE,
            scale_pos_weight=weight, n_jobs=2
        )
        model_label = "XGBoost"
    else:
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
            random_state=RANDOM_STATE, class_weight="balanced"
        )
        model_label = "HistGradientBoosting (fallback)"
    pipe = Pipeline([("preprocess", pre), ("model", estimator)])
    pipe.fit(X_train, y_train)
    probabilities = pipe.predict_proba(X_test)[:, 1]
    return model_df, train, test, features, pipe, probabilities, model_label

try:
    model_df, train_df, test_df, model_features, clf, test_proba, model_label = train_model(df)
except Exception as e:
    st.error(f"Could not prepare/train the model: {e}")
    st.warning("Check that your CSV contains the notebook's climate columns and enough consecutive daily records per district.")
    st.stop()

y_test = test_df["HEATWAVE_NEXT_DAY"].to_numpy()
if len(np.unique(y_test)) == 2:
    model_roc = roc_auc_score(y_test, test_proba)
else:
    model_roc = np.nan
model_pr = average_precision_score(y_test, test_proba)
default_pred = (test_proba >= 0.5).astype(int)

scenario_names = [
    "Baseline 2022-2025",
    "Moderate Warming (+1C)",
    "Extreme Warming (+2C)",
    "Dry Future (-20% rain, -10% soil moisture)",
    "Hot and Dry (+2C, -20% rain, -10% soil moisture)",
    "Adaptation (+10% LAI, +10% soil moisture)"
]

def apply_scenario(X, scenario):
    Xs = X.copy()
    temp_patterns = ["TEMP2M_C", "TMAX_C", "TMIN_C", "DEW2M_C", "SOILT1_C"]
    for col in Xs.columns:
        if scenario == "Moderate Warming (+1C)" and any(p in col for p in temp_patterns):
            Xs[col] = Xs[col] + 1.0
        elif scenario == "Extreme Warming (+2C)" and any(p in col for p in temp_patterns):
            Xs[col] = Xs[col] + 2.0
        elif scenario == "Dry Future (-20% rain, -10% soil moisture)":
            if "RAIN" in col: Xs[col] = Xs[col] * 0.80
            if "SOILM1" in col: Xs[col] = Xs[col] * 0.90
        elif scenario == "Hot and Dry (+2C, -20% rain, -10% soil moisture)":
            if any(p in col for p in temp_patterns): Xs[col] = Xs[col] + 2.0
            if "RAIN" in col: Xs[col] = Xs[col] * 0.80
            if "SOILM1" in col: Xs[col] = Xs[col] * 0.90
        elif scenario == "Adaptation (+10% LAI, +10% soil moisture)":
            if "LAI" in col: Xs[col] = Xs[col] * 1.10
            if "SOILM1" in col: Xs[col] = Xs[col] * 1.10
    return Xs

def scenario_results():
    base = test_df[model_features].copy()
    rows, district_frames = [], []
    for scenario in scenario_names:
        xs = base.copy() if scenario == scenario_names[0] else apply_scenario(base, scenario)
        p = clf.predict_proba(xs)[:, 1]
        rows.append({
            "Scenario": scenario,
            "Mean probability": float(np.mean(p)),
            "75th percentile": float(np.quantile(p, .75)),
            "90th percentile": float(np.quantile(p, .90)),
            "Expected heatwave district-days": float(np.sum(p))
        })
        tmp = test_df[["DATE", "DISTRICT"]].copy()
        tmp["probability"] = p
        tmp["scenario"] = scenario
        district_frames.append(tmp)
    return pd.DataFrame(rows), pd.concat(district_frames, ignore_index=True)

@st.cache_data(show_spinner=False)
def historical_summaries(data):
    annual = data.groupby("YEAR").agg(
        heatwave_days=("HEATWAVE", "sum"),
        observations=("HEATWAVE", "size"),
        mean_tmax_c=("TMAX_C", "mean") if "TMAX_C" in data else ("HEATWAVE", "mean")
    ).reset_index()
    annual["heatwave_rate"] = annual["heatwave_days"] / annual["observations"]
    district = data.groupby("DISTRICT").agg(
        heatwave_days=("HEATWAVE", "sum"),
        observations=("HEATWAVE", "size"),
        mean_tmax_c=("TMAX_C", "mean") if "TMAX_C" in data else ("HEATWAVE", "mean"),
        lat=("LAT", "mean"), lon=("LON", "mean")
    ).reset_index()
    district["heatwave_rate"] = district["heatwave_days"] / district["observations"]
    return annual, district.sort_values("heatwave_rate", ascending=False)

annual, district_summary = historical_summaries(df)

if page == "Climate Overview":
    st.header("Historical Climate Overview")
    a, b, c, d = st.columns(4)
    a.metric("Observations", f"{len(df):,}")
    b.metric("Districts", f"{df['DISTRICT'].nunique():,}")
    c.metric("Years covered", f"{df['YEAR'].min()}–{df['YEAR'].max()}")
    d.metric("Historical heatwave rate", f"{df['HEATWAVE'].mean()*100:.2f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Annual heatwave-labeled district-days")
        fig, ax = plt.subplots()
        ax.plot(annual["YEAR"], annual["heatwave_days"], marker="o")
        ax.set_xlabel("Year"); ax.set_ylabel("Heatwave district-days"); ax.grid(alpha=.3)
        st.pyplot(fig); plt.close(fig)
    with right:
        st.subheader("Annual mean maximum temperature")
        if "TMAX_C" in df:
            fig, ax = plt.subplots()
            ax.plot(annual["YEAR"], annual["mean_tmax_c"], marker="o")
            ax.set_xlabel("Year"); ax.set_ylabel("Mean TMAX (°C)"); ax.grid(alpha=.3)
            st.pyplot(fig); plt.close(fig)
        else:
            st.info("TMAX is unavailable in this file.")
    st.subheader("District-wise historical heatwave rate")
    fig, ax = plt.subplots(figsize=(10, max(4, len(district_summary)*.25)))
    view = district_summary.head(25).sort_values("heatwave_rate")
    ax.barh(view["DISTRICT"], view["heatwave_rate"])
    ax.set_xlabel("Heatwave rate"); ax.grid(axis="x", alpha=.3)
    st.pyplot(fig); plt.close(fig)
    st.dataframe(district_summary, use_container_width=True)

elif page == "Next-Day Prediction":
    st.header("Next-Day Heatwave Prediction")
    st.caption(f"Model: {model_label} · Train: before 2022 · Test: 2022 onward")
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{model_roc:.3f}" if np.isfinite(model_roc) else "N/A")
    c2.metric("PR-AUC", f"{model_pr:.3f}")
    c3.metric("Test observations", f"{len(test_df):,}")

    threshold = st.slider("Prediction probability threshold", 0.05, 0.95, 0.50, 0.05)
    pred = (test_proba >= threshold).astype(int)
    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
    m2.metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.3f}")
    m3.metric("F1-score", f"{f1_score(y_test, pred, zero_division=0):.3f}")
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No heatwave", "Heatwave"],
                yticklabels=["No heatwave", "Heatwave"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig); plt.close(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("ROC curve")
        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, test_proba)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {model_roc:.3f}")
            ax.plot([0, 1], [0, 1], "--"); ax.legend()
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            st.pyplot(fig); plt.close(fig)
    with right:
        st.subheader("Precision-recall curve")
        precision, recall, _ = precision_recall_curve(y_test, test_proba)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f"AP = {model_pr:.3f}")
        ax.axhline(y_test.mean(), linestyle="--", label="Positive rate")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()
        st.pyplot(fig); plt.close(fig)

    st.subheader("Prediction sample")
    results = test_df[["DATE", "DISTRICT", "HEATWAVE_NEXT_DAY"]].copy()
    results["Predicted probability"] = test_proba
    results["Predicted class"] = pred
    st.dataframe(results.sort_values("Predicted probability", ascending=False).head(200),
                 use_container_width=True)
    st.download_button("Download test predictions",
        results.to_csv(index=False).encode("utf-8"),
        "rajasthan_next_day_predictions.csv", "text/csv")

elif page == "Climate Regimes":
    st.header("Climate Regime Discovery")
    regime_features = [
        "TEMP2M_C", "TMAX_C", "TMIN_C", "DEW2M_C", "RAIN", "SRAD",
        "EVAP", "SOILT1_C", "SOILM1", "LAI", "CLOUD", "MSLP", "BLH"
    ]
    absent = [c for c in regime_features if c not in df.columns]
    if absent:
        st.warning("Missing columns for climate regime clustering: " + ", ".join(absent))
    else:
        regime_df = df.dropna(subset=regime_features).copy()
        k = st.slider("Number of climate regimes (K)", 3, 7, 4)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(regime_df[regime_features])
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        regime_df["CLIMATE_REGIME"] = km.fit_predict(scaled)
        summary = regime_df.groupby("CLIMATE_REGIME").agg(
            observations=("HEATWAVE", "size"),
            heatwave_rate=("HEATWAVE", "mean"),
            mean_tmax_c=("TMAX_C", "mean"),
            mean_rain=("RAIN", "mean"),
            mean_soilm1=("SOILM1", "mean"),
            mean_srad=("SRAD", "mean"),
            mean_lai=("LAI", "mean")
        ).reset_index().sort_values("heatwave_rate", ascending=False)
        st.dataframe(summary, use_container_width=True)
        fig, ax = plt.subplots()
        ax.bar(summary["CLIMATE_REGIME"].astype(str), summary["heatwave_rate"])
        ax.set_xlabel("Climate regime"); ax.set_ylabel("Historical heatwave rate")
        st.pyplot(fig); plt.close(fig)
        st.caption("Clusters are statistical groupings, not official meteorological categories.")

elif page == "2035 Scenario Simulator":
    st.header("2035 Climate Scenario Simulator")
    st.warning("These are controlled what-if stress tests applied to 2022–2025 test features. "
               "They are not official IPCC projections or deterministic forecasts.")
    scenario_df, district_scenarios = scenario_results()
    st.dataframe(scenario_df.style.format({
        "Mean probability": "{:.3f}", "75th percentile": "{:.3f}",
        "90th percentile": "{:.3f}", "Expected heatwave district-days": "{:.1f}"
    }), use_container_width=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    view = scenario_df.sort_values("Mean probability")
    ax.barh(view["Scenario"], view["Mean probability"])
    ax.set_xlabel("Mean predicted next-day heatwave probability")
    ax.grid(axis="x", alpha=.3)
    st.pyplot(fig); plt.close(fig)
    st.download_button("Download scenario comparison",
        scenario_df.to_csv(index=False).encode("utf-8"),
        "rajasthan_2035_scenario_comparison.csv", "text/csv")
    st.markdown("**Scenario assumptions follow the notebook:** +1°C, +2°C, −20% rainfall, "
                "−10% soil moisture, combined hot-and-dry stress, and +10% LAI/soil moisture adaptation.")

elif page == "District Risk":
    st.header("District-Level Scenario Sensitivity")
    scenario_df, district_scenarios = scenario_results()
    baseline = (district_scenarios[district_scenarios["scenario"] == scenario_names[0]]
                .groupby("DISTRICT")["probability"].mean().rename("baseline_probability"))
    summary = (district_scenarios.groupby(["scenario", "DISTRICT"])["probability"]
               .mean().reset_index(name="scenario_probability")
               .join(baseline, on="DISTRICT"))
    summary["probability_lift"] = summary["scenario_probability"] - summary["baseline_probability"]
    selected = st.selectbox("Select climate scenario", scenario_names[1:])
    view = summary[summary["scenario"] == selected].sort_values("probability_lift", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, len(view)*.25)))
    ax.barh(view.tail(25)["DISTRICT"], view.tail(25)["probability_lift"])
    ax.set_xlabel("Change in predicted probability vs baseline")
    ax.set_title(f"District sensitivity: {selected}")
    ax.grid(axis="x", alpha=.3)
    st.pyplot(fig); plt.close(fig)
    st.dataframe(view.sort_values("probability_lift", ascending=False), use_container_width=True)
    st.download_button("Download district sensitivity",
        view.to_csv(index=False).encode("utf-8"),
        "rajasthan_district_sensitivity.csv", "text/csv")

st.divider()
st.caption("Research prototype · Historical data-driven analysis · Scenario outputs are exploratory, "
           "not official weather warnings or validated climate projections.")
