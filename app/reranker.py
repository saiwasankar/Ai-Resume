"""Optional supervised criterion/evidence matcher.

The production model is loaded only from a locally trained artifact. Uploaded
resumes are never used for training automatically.
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "criterion_matcher.joblib"


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def trained_score(requirement: str, evidence: str) -> Optional[float]:
    """Return a calibrated-ish positive probability, or None if untrained."""
    model = _load_model()
    if model is None:
        return None
    try:
        return round(float(model.predict_proba([f"{requirement} [SEP] {evidence}"])[0][1]), 4)
    except Exception:
        return None


def model_status() -> dict:
    model = _load_model()
    return {
        "trained_model_available": model is not None,
        "model_path": str(MODEL_PATH) if model is not None else None,
        "analysis_method": "Supervised evidence matcher + semantic checks" if model is not None else "Semantic checks + conservative lexical evidence",
    }
