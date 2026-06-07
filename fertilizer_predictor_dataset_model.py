import os
import joblib
import pandas as pd


class FertilizerDatasetPredictor:
    """Predict NPK_Distance_Score and recommend fertilizer from Fertility_Dataset.csv.

    Model targets only NPK_Distance_Score (continuous). For recommendation, we sort
    candidate rows with same crop/soil/temp/humidity/moisture etc. and pick best.

    Note: As requested, EC/B/S/Cu are NOT used as features.
    """

    def __init__(
        self,
        model_path="fertility_dataset_npk_distance_rf.joblib",
        dataset_path="Fertility_Dataset.csv",
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        self.model = joblib.load(model_path)
        self.df = pd.read_csv(dataset_path)

        # Keep only columns used for feature construction (model pipeline decides)
        # We just need candidate generation.
        self.required_cols = {
            "Soil Type",
            "Crop Type",
            "Temparature",
            "Humidity ",
            "Moisture",
            "Fertilizer Name",
            "NPK_Distance_Score",
        }
        missing = self.required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    def _filter_candidates(
        self,
        soil_type: str,
        crop_type: str,
        temperature: float,
        humidity: float,
        moisture: float,
        top_k: int = 50,
    ):
        # Lightweight similarity filter: exact match for categorical + nearest for numeric.
        sub = self.df.copy()
        sub = sub[(sub["Soil Type"] == soil_type) & (sub["Crop Type"] == crop_type)]

        if sub.empty:
            # Fallback: use any soil/crop with closest numeric values.
            sub = self.df.copy()

        # Numeric distance on the three sensor features.
        sub["_dist"] = (
            (sub["Temparature"].astype(float) - float(temperature)) ** 2
            + (sub["Humidity "].astype(float) - float(humidity)) ** 2
            + (sub["Moisture"].astype(float) - float(moisture)) ** 2
        )

        sub = sub.sort_values("_dist").head(top_k)
        return sub

    def predict_best_fertilizer(
        self,
        soil_type: str,
        crop_type: str,
        temperature: float,
        humidity: float,
        moisture: float,
        top_k_candidates: int = 80,
    ):
        """Return best fertilizer name + predicted score.

        Output: dict
        - fertilizer_name
        - predicted_score
        - candidates_used
        """
        candidates = self._filter_candidates(
            soil_type=soil_type,
            crop_type=crop_type,
            temperature=temperature,
            humidity=humidity,
            moisture=moisture,
            top_k=top_k_candidates,
        )

        # Features: we use the same row-wise columns except that the pipeline will
        # drop everything it doesn't know. The model pipeline was trained on
        # all non-dropped columns; we therefore supply the full row subset.
        feature_df = candidates.drop(columns=["NPK_Distance_Score"], errors="ignore")

        # Model expects feature columns consistent with training pipeline.
        preds = self.model.predict(feature_df)
        candidates = candidates.copy()
        candidates["_pred"] = preds

        # For multiple rows per fertilizer, take best (lowest distance -> best)
        # Dataset seems to use higher score for better (observed range 40..150),
        # but to be safe we treat higher as better.
        best_idx = candidates["_pred"].idxmax()
        best = candidates.loc[best_idx]

        return {
            "fertilizer_name": str(best["Fertilizer Name"]),
            "predicted_score": float(best["_pred"]),
            "candidates_used": int(len(candidates)),
        }

