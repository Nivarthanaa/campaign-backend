import os
import json
import joblib # type: ignore
import gdown
import shap # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from typing import Any, Optional, Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
def download_if_missing(file_id, filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        gdown.download(url, path, quiet=False)

_pre_cluster: Any = None
_kmeans: Any = None
_pre_predict: Any = None
_rf: Any = None
_schema: Optional[Dict] = None
_explainer: Any = None

def load_all():
    global _pre_cluster, _kmeans, _pre_predict, _rf, _schema

    # DOWNLOAD FIRST
    download_if_missing("1pQnnnOkPCXqs9NuYkxPF29ojfSSGlv1z", "preprocess_cluster.joblib")
    download_if_missing("1NTlxpVu6aF03ewcpPuba8rxRFoMz6fJz", "kmeans.joblib")
    download_if_missing("1SYvW_b0B3gWY18KxYU2U126g5xfDjB1-", "preprocess_predict.joblib")
    download_if_missing("1-IdB7PWijnHNpBGwFa53pkUUc6wvS7Fq", "rf_model.joblib")
    download_if_missing("1cXIAUv0LeI4K2SRsf8UZAq-zU_4tSFHs", "schema.json")

    try:
        models_to_load = {
            "preprocess_cluster.joblib": "_pre_cluster",
            "kmeans.joblib": "_kmeans",
            "preprocess_predict.joblib": "_pre_predict",
            "rf_model.joblib": "_rf"
        }
        for filename, var_name in models_to_load.items():
            path = os.path.join(MODEL_DIR, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file missing: {filename}")
            globals()[var_name] = joblib.load(path)
            
        schema_path = os.path.join(MODEL_DIR, "schema.json")
        if not os.path.exists(schema_path):
            raise FileNotFoundError("Schema file missing: schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            _schema = json.load(f)
            
        print(f"Successfully loaded all {len(models_to_load)} models and schema.")
    except Exception as e:
        print(f"CRITICAL ML ERROR: {str(e)}")
        raise RuntimeError(f"Failed to load ML models or schema: {str(e)}")

def ensure_loaded():
    if any(x is None for x in [_pre_cluster, _kmeans, _pre_predict, _rf, _schema]):
        load_all()

def get_schema():
    ensure_loaded()
    return _schema

def validate_payload(payload: dict):
    ensure_loaded()
    if _schema is None:
        raise RuntimeError("Schema not loaded.")
    required = _schema["feature_cols"]
    missing = [c for c in required if c not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

def to_dataframe(payload: dict) -> pd.DataFrame:
    ensure_loaded()
    assert _schema is not None
    df = pd.DataFrame([payload], columns=_schema["feature_cols"])
    return df

def run_clustering(payload: dict):
    ensure_loaded()
    validate_payload(payload)
    df = to_dataframe(payload)
    if _pre_cluster is None or _kmeans is None:
        raise RuntimeError("Clustering models not loaded.")
    Xt_raw = _pre_cluster.transform(df) # type: ignore
    Xt = Xt_raw.toarray() if hasattr(Xt_raw, "toarray") else Xt_raw
    
    # Apply feature weights if present in schema
    weights = _schema.get("feature_weights") if _schema else None
    if weights:
        Xt = Xt * np.array(weights)

    cluster_id = int(_kmeans.predict(Xt)[0]) # type: ignore
    return cluster_id

def run_prediction(payload: dict, cluster_id: int):
    ensure_loaded()
    df = to_dataframe(payload).copy()
    df["cluster_id"] = str(cluster_id)
    assert _pre_predict is not None
    assert _rf is not None
    Xt = _pre_predict.transform(df) # type: ignore
    pred = int(_rf.predict(Xt)[0]) # type: ignore
    probs = _rf.predict_proba(Xt)[0].tolist() # type: ignore
    return pred, probs

def segment_profile(cluster_id: int):
    ensure_loaded()
    assert _schema is not None
    profiles = _schema.get("cluster_profiles", [])
    for p in profiles:
        if int(p["cluster_id"]) == int(cluster_id):
            return p
    return None

def get_feature_importance():
    ensure_loaded()
    assert _rf is not None
    assert _schema is not None
    # Scikit-learn Random Forest feature importances
    importances = _rf.feature_importances_.tolist()
    feature_names = _schema["feature_cols"] + ["cluster_id"]
    return dict(zip(feature_names, importances))

def explain_prediction(payload: dict, cluster_id: int):
    global _explainer
    ensure_loaded()
    assert _rf is not None
    assert _pre_predict is not None
    assert _schema is not None
    
    if _explainer is None:
        # TreeExplainer is efficient for Random Forests
        _explainer = shap.TreeExplainer(_rf)
    
    df = to_dataframe(payload).copy()
    df["cluster_id"] = str(cluster_id)
    Xt = _pre_predict.transform(df) # type: ignore
    
    # Xt is likely a numpy array or CSR matrix from the preprocessor pipeline
    shap_values = _explainer.shap_values(Xt) # type: ignore
    
    # For multi-class (0, 1, 2) where 2 is "High", we ALWAYS explain the "High" class
    # This ensures that Positive (Blue) always means "Success Driver"
    # and Negative (Red) always means "Performance Detractor" for the user.
    target_class = 2 # High
    
    if isinstance(shap_values, list):
        # shap_values[class][instance]
        instance_shap = shap_values[target_class][0].tolist() # type: ignore
    elif len(shap_values.shape) == 3: # (instances, features, classes)
        instance_shap = shap_values[0, :, target_class].tolist() # type: ignore
    else: # (instances, features)
        instance_shap = shap_values[0].tolist() # type: ignore
        
    feature_names = _schema["feature_cols"] + ["cluster_id"]
    return dict(zip(feature_names, instance_shap))
