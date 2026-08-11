import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocess import prepare_data


# ============================================================
# CONFIGURATION
# ============================================================

RF_MODEL_PATH = "ml/model.pkl"
XGB_MODEL_PATH = "ml/xgboost_model.pkl"

RESULT_PATH = "ml/model_comparison.csv"


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(name, model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    # Default threshold for model comparison
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {

        "Model": name,

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1_Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_test,
            probabilities
        )
    }


# ============================================================
# MAIN
# ============================================================

def compare_models():

    print("\n")
    print("=" * 70)
    print(" RANDOM FOREST VS XGBOOST")
    print("=" * 70)

    # --------------------------------------------------------
    # Prepare exactly the same validation data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data()

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    random_forest = joblib.load(
        RF_MODEL_PATH
    )

    xgboost = joblib.load(
        XGB_MODEL_PATH
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    rf_result = evaluate_model(
        "Random Forest",
        random_forest,
        X_test,
        y_test
    )

    xgb_result = evaluate_model(
        "XGBoost",
        xgboost,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Create comparison table
    # --------------------------------------------------------

    results = pd.DataFrame([
        rf_result,
        xgb_result
    ])

    # Convert to percentages for display

    display_results = results.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC"
    ]

    for column in metric_columns:

        display_results[column] = (
            display_results[column] * 100
        ).round(2)

    print("\n")
    print("=" * 70)
    print(" MODEL COMPARISON")
    print("=" * 70)

    print(
        display_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    results.to_csv(
        RESULT_PATH,
        index=False
    )

    print(
        f"\nComparison saved to:"
        f" {RESULT_PATH}"
    )

    # --------------------------------------------------------
    # Determine best model
    # --------------------------------------------------------

    best_f1_index = results[
        "F1_Score"
    ].idxmax()

    best_model = results.loc[
        best_f1_index,
        "Model"
    ]

    print("\n")
    print(
        f"Best model based on F1:"
        f" {best_model}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    compare_models()