import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ml.models.predict_robust_v3 import predict_file


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Revenue Leakage Detector",
    description="AI-powered revenue leakage detection system",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "AI Revenue Leakage Detector",
        "version": "1.0.0"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    filename = file.filename.lower()

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    if not filename.endswith(
        (".csv", ".xlsx", ".xls")
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file format. "
                "Please upload CSV or Excel."
            )
        )

    temp_path = None

    try:

        # ----------------------------------------------------
        # Create temporary file
        # ----------------------------------------------------

        suffix = os.path.splitext(
            filename
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            content = await file.read()

            temp_file.write(
                content
            )

            temp_path = temp_file.name

        # ----------------------------------------------------
        # Run your existing ML pipeline
        # ----------------------------------------------------

        results = predict_file(
            temp_path
        )

        if results is None:

            raise HTTPException(
                status_code=422,
                detail=(
                    "The uploaded file "
                    "could not be analyzed."
                )
            )

        # ----------------------------------------------------
        # Convert DataFrame -> JSON records
        # ----------------------------------------------------

        records = results.to_dict(
            orient="records"
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        total_invoices = len(
            results
        )

        leakage_count = int(
            (
                results["Prediction"]
                == "Revenue Leakage Detected"
            ).sum()
        )

        review_count = int(
            (
                results["Prediction"]
                == "Potential Anomaly - Review"
            ).sum()
        )

        high_risk_count = int(
            (
                results["Risk_Level"]
                == "HIGH"
            ).sum()
        )

        medium_risk_count = int(
            (
                results["Risk_Level"]
                == "MEDIUM"
            ).sum()
        )

        low_risk_count = int(
            (
                results["Risk_Level"]
                == "LOW"
            ).sum()
        )

        total_potential_leakage = float(
            results[
                "Potential_Leakage"
            ]
            .fillna(0)
            .sum()
        )

        # ----------------------------------------------------
        # Return API response
        # ----------------------------------------------------

        return {

            "success": True,

            "filename": file.filename,

            "summary": {

                "total_invoices":
                    total_invoices,

                "leakage_detected":
                    leakage_count,

                "potential_anomalies":
                    review_count,

                "high_risk":
                    high_risk_count,

                "medium_risk":
                    medium_risk_count,

                "low_risk":
                    low_risk_count,

                "total_potential_leakage":
                    round(
                        total_potential_leakage,
                        2
                    )
            },

            "results": records
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # ----------------------------------------------------
        # Remove temporary uploaded file
        # ----------------------------------------------------

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):

            os.remove(
                temp_path
            )