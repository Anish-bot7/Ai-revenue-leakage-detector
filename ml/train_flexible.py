"""
Flexible Revenue Leakage Model Training

Purpose:
Train an XGBoost model using billing features that can
realistically be obtained from different CSV/Excel formats.

Important:
Target leakage columns are NEVER used as model features.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

from ml.flexible_features import (
    FLEXIBLE_NUMERIC_FEATURES,
    OPTIONAL_NUMERIC_FEATURES,
    FLEXIBLE_CATEGORICAL_FEATURES,
    FLEXIBLE_DATE_FEATURES,
)

from ml.preprocessing.feature_builder import build_features
from ml.preprocessing.date_features import build_date_features


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = (
    "datasets/training/revenue_leakage_dataset.csv"
)

MODEL_DIR = (
    "ml/models"
)

MODEL_PATH = (
    f"{MODEL_DIR}/flexible_model.pkl"
)

PREPROCESSOR_PATH = (
    f"{MODEL_DIR}/flexible_preprocessor.pkl"
)

THRESHOLD_PATH = (
    f"{MODEL_DIR}/flexible_threshold.pkl"
)

FEATURES_PATH = (
    f"{MODEL_DIR}/flexible_features.pkl"
)

METRICS_PATH = (
    f"{MODEL_DIR}/flexible_metrics.pkl"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)

print(
    "FLEXIBLE REVENUE LEAKAGE MODEL"
)

print("=" * 70)


print(
    "\nLoading dataset..."
)

df = pd.read_csv(
    DATASET_PATH,
    keep_default_na=False
)


print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# TARGET CHECK
# ============================================================

if "Leakage" not in df.columns:

    raise ValueError(
        "Leakage target column not found."
    )


print(
    "\nTarget distribution:"
)

print(
    df["Leakage"].value_counts()
)


# ============================================================
# CLEAN TARGET
# ============================================================

df["Leakage"] = pd.to_numeric(
    df["Leakage"],
    errors="coerce"
)

df = df.dropna(
    subset=["Leakage"]
)

df["Leakage"] = (
    df["Leakage"]
    .astype(int)
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "BUILDING FLEXIBLE FEATURES"
)

print(
    "=" * 70
)


# ------------------------------------------------------------
# General feature engineering
# ------------------------------------------------------------

df = build_features(
    df
)


# ------------------------------------------------------------
# Date feature engineering
# ------------------------------------------------------------

df = build_date_features(
    df
)


# ============================================================
# FEATURE LIST
# ============================================================

NUMERIC_FEATURES = (
    FLEXIBLE_NUMERIC_FEATURES
    + OPTIONAL_NUMERIC_FEATURES
    + FLEXIBLE_DATE_FEATURES
)

CATEGORICAL_FEATURES = (
    FLEXIBLE_CATEGORICAL_FEATURES
)

ALL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# Remove accidental duplicates
NUMERIC_FEATURES = list(
    dict.fromkeys(
        NUMERIC_FEATURES
    )
)

CATEGORICAL_FEATURES = list(
    dict.fromkeys(
        CATEGORICAL_FEATURES
    )
)

ALL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# CREATE MISSING OPTIONAL COLUMNS
# ============================================================

for feature in ALL_FEATURES:

    if feature not in df.columns:

        df[feature] = np.nan


# ============================================================
# DISPLAY FEATURES
# ============================================================

print(
    "\nFlexible numerical features:"
)

for feature in NUMERIC_FEATURES:

    print(
        f"  ✓ {feature}"
    )


print(
    "\nFlexible categorical features:"
)

for feature in CATEGORICAL_FEATURES:

    print(
        f"  ✓ {feature}"
    )


print(
    f"\nTotal model features: "
    f"{len(ALL_FEATURES)}"
)


# ============================================================
# CREATE X AND y
# ============================================================

X = df[
    ALL_FEATURES
].copy()

y = df[
    "Leakage"
].copy()


# ============================================================
# CLEAN NUMERICAL VALUES
# ============================================================

for column in NUMERIC_FEATURES:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# ============================================================
# CLEAN CATEGORICAL VALUES
# ============================================================

for column in CATEGORICAL_FEATURES:

    X[column] = (
        X[column]
        .astype(str)
        .replace(
            [
                "",
                "nan",
                "None",
                "NaN"
            ],
            np.nan
        )
    )


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TRAIN / TEST SPLIT"
)

print(
    "=" * 70
)


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y
)


print(
    f"\nTraining samples: "
    f"{len(X_train)}"
)

print(
    f"Testing samples: "
    f"{len(X_test)}"
)


# ============================================================
# NUMERICAL PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        )

    ]
)


# ============================================================
# CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",

            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numerical",

            numeric_pipeline,

            NUMERIC_FEATURES
        ),

        (
            "categorical",

            categorical_pipeline,

            CATEGORICAL_FEATURES
        ),

    ],

    remainder="drop"

)


# ============================================================
# FIT PREPROCESSOR
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FITTING PREPROCESSOR"
)

print(
    "=" * 70
)


X_train_processed = (
    preprocessor.fit_transform(
        X_train
    )
)

X_test_processed = (
    preprocessor.transform(
        X_test
    )
)


print(
    "\nProcessed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ============================================================
# CREATE XGBOOST MODEL
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "CREATING FLEXIBLE XGBOOST MODEL"
)

print(
    "=" * 70
)


model = XGBClassifier(

    n_estimators=400,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.85,

    colsample_bytree=0.85,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=RANDOM_STATE,

    n_jobs=-1,

)


# ============================================================
# TRAIN
# ============================================================

print(
    "\nTraining flexible model..."
)


model.fit(

    X_train_processed,

    y_train

)


print(
    "Training completed."
)


# ============================================================
# PREDICT PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# ROC AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print(
    "\nROC-AUC:",
    round(
        roc_auc,
        4
    )
)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "THRESHOLD OPTIMIZATION"
)

print(
    "=" * 70
)


threshold_results = []


for threshold in np.arange(
    0.20,
    0.81,
    0.05
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    threshold_results.append({

        "Threshold":
            round(
                float(threshold),
                2
            ),

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1_Score":
            f1,

    })

    print(
        f"{threshold:.2f}  "
        f"Accuracy={accuracy:.4f}  "
        f"Precision={precision:.4f}  "
        f"Recall={recall:.4f}  "
        f"F1={f1:.4f}"
    )


# ============================================================
# BEST THRESHOLD
# ============================================================

threshold_df = pd.DataFrame(
    threshold_results
)


best_row = threshold_df.loc[
    threshold_df["F1_Score"].idxmax()
]


best_threshold = float(
    best_row["Threshold"]
)


print(
    "\n"
    + "=" * 70
)

print(
    "BEST THRESHOLD"
)

print(
    "=" * 70
)

print(
    f"\nThreshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : "
    f"{best_row['Accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{best_row['Precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best_row['Recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{best_row['F1_Score']:.4f}"
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

final_predictions = (
    probabilities >= best_threshold
).astype(int)


# ============================================================
# FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    final_predictions
)

precision = precision_score(
    y_test,
    final_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    final_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    final_predictions,
    zero_division=0
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FINAL MODEL PERFORMANCE"
)

print(
    "=" * 70
)


print(
    "\nAccuracy:",
    round(
        accuracy,
        4
    )
)

print(
    "Precision:",
    round(
        precision,
        4
    )
)

print(
    "Recall:",
    round(
        recall,
        4
    )
)

print(
    "F1 Score:",
    round(
        f1,
        4
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc,
        4
    )
)


print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        final_predictions,
        target_names=[
            "No Leakage",
            "Leakage"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    final_predictions
)


print(
    "\nConfusion Matrix:"
)

print(
    cm
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)


joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)


joblib.dump(
    best_threshold,
    THRESHOLD_PATH
)


joblib.dump(
    ALL_FEATURES,
    FEATURES_PATH
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "accuracy":
        float(accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1_score":
        float(f1),

    "roc_auc":
        float(roc_auc),

    "threshold":
        float(best_threshold),

    "training_samples":
        int(len(X_train)),

    "testing_samples":
        int(len(X_test)),

    "feature_count":
        int(len(ALL_FEATURES)),

}


joblib.dump(
    metrics,
    METRICS_PATH
)


# ============================================================
# SAVE THRESHOLD TABLE
# ============================================================

threshold_df.to_csv(
    f"{MODEL_DIR}/flexible_threshold_results.csv",
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FLEXIBLE MODEL TRAINING COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nSaved files:"
)

print(
    f"Model        : {MODEL_PATH}"
)

print(
    f"Preprocessor : {PREPROCESSOR_PATH}"
)

print(
    f"Threshold    : {THRESHOLD_PATH}"
)

print(
    f"Features     : {FEATURES_PATH}"
)

print(
    f"Metrics      : {METRICS_PATH}"
)

print(
    "\nThe flexible model is ready for evaluation."
)