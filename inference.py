"""
VitaFusion AI — Core Inference Engine
========================================
Ties together all encoders and fusion model for end-to-end prediction.
Runs in simulation mode by default (no GPU required).
"""

import numpy as np
from typing import Dict, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.retinal_encoder   import RetinalEncoderSimulated
from models.tongue_encoder    import TongueEncoderSimulated
from models.ecg_encoder       import ECGEncoderSimulated
from models.ecg_image_encoder import ECGImageEncoderSimulated
from models.clinical_encoder  import ClinicalEncoderSimulated
from models.fusion_model      import VitaFusionSimulated
from preprocessing.retinal_preprocess  import preprocess_retinal_image
from preprocessing.tongue_preprocess   import preprocess_tongue_image
from preprocessing.ecg_preprocess      import preprocess_ecg, load_ecg_from_csv
from preprocessing.clinical_preprocess import preprocess_clinical_features


class VitaFusionInference:
    """
    End-to-end inference pipeline in simulation mode.
    Each modality is optional; gracefully handles missing inputs.
    """

    def __init__(self):
        self.retinal_model   = RetinalEncoderSimulated()
        self.tongue_model    = TongueEncoderSimulated()
        self.ecg_model       = ECGEncoderSimulated()
        self.ecg_image_model = ECGImageEncoderSimulated()   # NEW: image-based ECG
        self.clinical_model  = ClinicalEncoderSimulated()
        self.fusion          = VitaFusionSimulated()

    def run(
        self,
        retinal_image   = None,   # PIL Image, numpy array, or bytes
        tongue_image    = None,
        ecg_data        = None,   # np.ndarray [leads, timesteps] or bytes (CSV)
        ecg_image       = None,   # bytes / PIL Image of ECG paper photo
        clinical_data   : Optional[Dict] = None,
        patient_info    : Optional[Dict] = None,
    ) -> Dict:
        """
        Run full multimodal inference.

        Returns comprehensive results dict with risk scores, per-modality
        findings, and XAI-ready data.
        """
        results = {
            "patient_info":     patient_info or {},
            "retinal_preds":    None,
            "tongue_preds":     None,
            "ecg_preds":        None,
            "ecg_image_preds":  None,
            "clinical_preds":   None,
            "fusion_results":   None,
            "preprocessing":    {},
        }

        # ── Clinical risk hint (used to correlate modality predictions) ────
        clinical_risk_hint = 0.3  # population average prior

        # ── 1. Clinical Data ────────────────────────────────────────────────
        if clinical_data:
            try:
                preproc = preprocess_clinical_features(clinical_data)
                clinical_preds = self.clinical_model.predict(
                    preproc["filled_features"]
                )
                results["clinical_preds"]  = clinical_preds
                results["preprocessing"]["clinical"] = preproc
                # Update risk hint for other modalities
                clinical_risk_hint = float(np.mean([
                    clinical_preds["cardiovascular_risk"],
                    clinical_preds["diabetes_risk"],
                    clinical_preds["hypertension_risk"],
                ]))
            except Exception as e:
                results["errors"] = results.get("errors", []) + [f"Clinical: {e}"]

        # ── 2. Retinal Image ─────────────────────────────────────────────────
        if retinal_image is not None:
            try:
                preproc = preprocess_retinal_image(retinal_image)
                viz_arr = preproc["viz_array"].astype(np.float32) / 255.0
                retinal_preds = self.retinal_model.predict(viz_arr, clinical_risk_hint)
                results["retinal_preds"]  = retinal_preds
                results["preprocessing"]["retinal"] = preproc
            except Exception as e:
                results["errors"] = results.get("errors", []) + [f"Retinal: {e}"]

        # ── 3. Tongue Image ───────────────────────────────────────────────────
        if tongue_image is not None:
            try:
                preproc = preprocess_tongue_image(tongue_image)
                viz_arr = preproc["viz_array"].astype(np.float32) / 255.0
                tongue_preds = self.tongue_model.predict(viz_arr, clinical_risk_hint)
                results["tongue_preds"]  = tongue_preds
                results["preprocessing"]["tongue"] = preproc
            except Exception as e:
                results["errors"] = results.get("errors", []) + [f"Tongue: {e}"]

        # ── 4. ECG Signal ──────────────────────────────────────────────────────────
        if ecg_data is not None:
            try:
                if isinstance(ecg_data, bytes):
                    raw_signal, fs = load_ecg_from_csv(ecg_data)
                elif isinstance(ecg_data, np.ndarray):
                    raw_signal, fs = ecg_data, 250.0
                else:
                    raw_signal, fs = ecg_data, 250.0

                preproc = preprocess_ecg(raw_signal, fs=fs)
                ecg_preds = self.ecg_model.predict(preproc["signal_norm"], clinical_risk_hint)
                ecg_preds["hrv_features"] = preproc["hrv_features"]
                ecg_preds["r_peaks"]      = preproc["r_peaks"]
                ecg_preds["signal_norm"]  = preproc["signal_norm"]
                ecg_preds["sample_rate"]  = preproc["sample_rate"]
                ecg_preds["lead_names"]   = preproc["lead_names"]
                ecg_preds["quality"]      = preproc["quality"]
                results["ecg_preds"]     = ecg_preds
                results["preprocessing"]["ecg"] = preproc
            except Exception as e:
                results["errors"] = results.get("errors", []) + [f"ECG Signal: {e}"]

        # ── 4b. ECG Image (photo of ECG paper) ────────────────────────────────
        if ecg_image is not None:
            try:
                ecg_img_preds = self.ecg_image_model.predict(ecg_image, clinical_risk_hint)
                results["ecg_image_preds"] = ecg_img_preds
                # If no CSV ECG was provided, also use image preds as the primary ECG result
                if results["ecg_preds"] is None:
                    results["ecg_preds"] = ecg_img_preds
            except Exception as e:
                results["errors"] = results.get("errors", []) + [f"ECG Image: {e}"]

        # ── 5. Fusion ───────────────────────────────────────────────────────────
        fusion = self.fusion.fuse(
            retinal_preds  = results["retinal_preds"],
            tongue_preds   = results["tongue_preds"],
            ecg_preds      = results["ecg_preds"],
            clinical_preds = results["clinical_preds"],
        )
        results["fusion_results"] = fusion

        return results
