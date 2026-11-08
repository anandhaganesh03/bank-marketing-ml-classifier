from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

HERE = Path(__file__).parent
TARGET = "y"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

st.set_page_config(page_title="Bank Marketing Classifier", layout="wide")
st.title("will the Bank customer subscribe or not?")
st.write(
    "Upload a test CSV, choose a model and see how it works. "
    "The models are loaded from the model folder."
)


@st.cache_resource
def load_model(name):
    return joblib.load(HERE / "model" / MODEL_FILES[name])


def evaluate(pipe, X, y):
    pred = pipe.predict(X)
    if hasattr(pipe, "predict_proba"):
        score = pipe.predict_proba(X)[:, 1]
    else:
        score = pipe.decision_function(X)
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, score),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return pred, metrics


# ---- sidebar: data and model choice ----
st.sidebar.header("1. Test data")
uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
st.sidebar.header("2. Model")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES))
st.sidebar.caption(
    "The CSV needs the 16 original input columns plus the target column y "
    "(1 = subscribed, 0 = not subscribed)."
)

# ---- load the data ----
if uploaded is not None:
    data = pd.read_csv(uploaded, sep=None, engine="python")
    source = f"uploaded file: {uploaded.name}"
else:
    data = pd.read_csv(HERE / "test_data.csv")
    source = "the bundled test_data.csv (upload your own file in the sidebar)"

if TARGET not in data.columns:
    st.error(f"The file must contain a target column named '{TARGET}'.")
    st.stop()
if not pd.api.types.is_numeric_dtype(data[TARGET]):
    data[TARGET] = data[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
if data[TARGET].isna().any() or data[TARGET].nunique() < 2:
    st.error("The target column must contain both classes (yes/no or 1/0) and no blanks.")
    st.stop()

pipe = load_model(model_name)
required = list(pipe.feature_names_in_)
missing = [c for c in required if c not in data.columns]
if missing:
    st.error("These required columns are missing from the file: " + ", ".join(missing))
    st.stop()

X = data[required]
y = data[TARGET].astype(int)
st.caption(f"Evaluating on {source}: {len(data)} rows.")
with st.expander("Preview the data"):
    st.dataframe(data.head(10))

# ---- results for the chosen model ----
pred, scores = evaluate(pipe, X, y)
st.subheader(f"Results for {model_name}")
cols = st.columns(6)
for col, (label, value) in zip(cols, scores.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["No", "Yes"], ax=ax, colorbar=False
    )
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("**Classification report**")
    report = classification_report(y, pred, target_names=["No", "Yes"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3))

# ---- all five models on the same data ----
st.subheader("All five models on this data")
rows = {}
for name in MODEL_FILES:
    _, rows[name] = evaluate(load_model(name), X, y)
st.dataframe(pd.DataFrame(rows).T.round(4))
