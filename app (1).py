import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)

st.set_page_config(page_title="Tree Model Pipelines Dashboard", layout="wide")

RANDOM_STATE = 42
DEFAULT_PATH = "simulated_dataset.csv"


def generate_dataset(n_samples=500, n_predictors=50, random_state=42):
    rng = np.random.default_rng(random_state)

    X = rng.normal(0, 1, size=(n_samples, n_predictors))
    predictor_names = [f"X{i+1}" for i in range(n_predictors)]

    y_cont = (
        3 * X[:, 0]
        - 2 * X[:, 1]
        + 1.5 * X[:, 2]
        + 0.8 * X[:, 3] * X[:, 4]
        - 1.2 * (X[:, 5] ** 2)
        + rng.normal(0, 2, n_samples)
    )

    logit = (
        1.2 * X[:, 0]
        - 1.5 * X[:, 1]
        + 1.0 * X[:, 6]
        - 0.7 * X[:, 7] * X[:, 8]
        + 0.5 * (X[:, 9] > 0).astype(int)
    )
    prob = 1 / (1 + np.exp(-logit))
    y_bin = rng.binomial(1, prob, n_samples)

    df = pd.DataFrame(X, columns=predictor_names)
    df["y_continuous"] = y_cont
    df["y_binary"] = y_bin
    return df


def load_data(uploaded_file=None, use_generated=True, random_state=42):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    if os.path.exists(DEFAULT_PATH):
        return pd.read_csv(DEFAULT_PATH)

    if use_generated:
        df = generate_dataset(n_samples=500, n_predictors=50, random_state=random_state)
        df.to_csv(DEFAULT_PATH, index=False)
        return df

    return None


def prepare_features(df):
    required_cols = {"y_continuous", "y_binary"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Dataset must contain y_continuous and y_binary columns.")

    feature_cols = [c for c in df.columns if c not in ["y_continuous", "y_binary"]]
    X = df[feature_cols].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    return X_imputed, df["y_continuous"], df["y_binary"], feature_cols


def get_regression_models(random_state=42):
    return {
        "Decision Tree": DecisionTreeRegressor(
            max_depth=6, min_samples_leaf=5, random_state=random_state
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=random_state
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=3, random_state=random_state
        ),
        "HistGradient Boosting": HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.05, max_depth=6, random_state=random_state
        ),
    }


def get_classification_models(random_state=42):
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=random_state
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.05, max_depth=3, random_state=random_state
        ),
        "HistGradient Boosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_depth=6, random_state=random_state
        ),
    }


def evaluate_regression(df, test_size=0.2, random_state=42):
    X, y_reg, _, _ = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=test_size, random_state=random_state
    )

    rows = []
    for name, model in get_regression_models(random_state).items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rows.append(
            {
                "Pipeline": name,
                "MAE": mean_absolute_error(y_test, preds),
                "RMSE": mean_squared_error(y_test, preds) ** 0.5,
                "R2": r2_score(y_test, preds),
            }
        )

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def evaluate_classification(df, test_size=0.2, random_state=42):
    X, _, y_clf, _ = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
    )

    rows = []
    for name, model in get_classification_models(random_state).items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            proba = 1 / (1 + np.exp(-scores))
        else:
            proba = preds

        rows.append(
            {
                "Pipeline": name,
                "Accuracy": accuracy_score(y_test, preds),
                "F1": f1_score(y_test, preds),
                "ROC_AUC": roc_auc_score(y_test, proba),
            }
        )

    return pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)


st.title("Tree Model Pipelines Dashboard")
st.write(
    "This dashboard compares five tree-model pipeline families for both "
    "regression and classification."
)

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    use_generated = st.checkbox("Generate sample data if CSV is unavailable", value=True)
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

df = load_data(uploaded_file=uploaded_file, use_generated=use_generated, random_state=int(random_state))

if df is None:
    st.error("No dataset found. Upload a CSV or enable sample data generation.")
    st.stop()

required_cols = {"y_continuous", "y_binary"}
if not required_cols.issubset(df.columns):
    st.error("Dataset must contain y_continuous and y_binary columns.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Rows", df.shape[0])
with c2:
    st.metric("Columns", df.shape[1])
with c3:
    st.metric("Predictors", df.shape[1] - 2)

reg_results = evaluate_regression(df, test_size=float(test_size), random_state=int(random_state))
clf_results = evaluate_classification(df, test_size=float(test_size), random_state=int(random_state))

tab1, tab2, tab3 = st.tabs(["Regression Pipelines", "Classification Pipelines", "Summary"])

with tab1:
    st.subheader("Regression Tree Pipelines")
    st.dataframe(reg_results, use_container_width=True)
    st.write("RMSE comparison")
    st.bar_chart(reg_results.set_index("Pipeline")[["RMSE"]])

with tab2:
    st.subheader("Classification Tree Pipelines")
    st.dataframe(clf_results, use_container_width=True)
    st.write("ROC-AUC comparison")
    st.bar_chart(clf_results.set_index("Pipeline")[["ROC_AUC"]])

with tab3:
    st.subheader("Best Models at a Glance")
    best_reg = reg_results.iloc[0]
    best_clf = clf_results.iloc[0]

    left, right = st.columns(2)
    with left:
        st.success(
            f"Best regression pipeline: {best_reg['Pipeline']} | "
            f"RMSE = {best_reg['RMSE']:.3f}, R² = {best_reg['R2']:.3f}"
        )
    with right:
        st.success(
            f"Best classification pipeline: {best_clf['Pipeline']} | "
            f"ROC-AUC = {best_clf['ROC_AUC']:.3f}, Accuracy = {best_clf['Accuracy']:.3f}"
        )

    merged = reg_results[["Pipeline", "MAE", "RMSE", "R2"]].merge(
        clf_results[["Pipeline", "Accuracy", "F1", "ROC_AUC"]],
        on="Pipeline",
        how="inner",
    )
    st.dataframe(merged, use_container_width=True)

st.caption(
    "Single-file Streamlit app with built-in data generation and five tree-model "
    "pipeline families for both regression and classification."
)
