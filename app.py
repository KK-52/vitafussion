"""
VitaFusion AI — FastAPI Application Entry Point (Consumer Flow)
"""

import os, sys, json, base64, io, traceback, tempfile, pathlib, random, string
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

app = FastAPI(title="VitaFusion AI", version="2.0.0")

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SECRET_KEY", "vitafusion-secret-key-2025"))
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Static files & templates ──────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Server-side results cache ─────────────────────────────────────────────────
_CACHE_FILE = pathlib.Path(tempfile.gettempdir()) / "vitafusion_results.json"


def _save_results(data: dict):
    with open(_CACHE_FILE, "w") as f:
        json.dump(data, f)


def _load_results() -> dict | None:
    if _CACHE_FILE.exists():
        with open(_CACHE_FILE) as f:
            return json.load(f)
    return None


# ── Inference engine (lazy init) ──────────────────────────────────────────────
from inference import VitaFusionInference
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = VitaFusionInference()
    return _engine


# ── Session helper ────────────────────────────────────────────────────────────
def current_user_phone(request: Request) -> str | None:
    return request.session.get("user_phone")


def require_session(request: Request):
    """Returns phone if logged in, else None (caller should redirect)."""
    return request.session.get("user_phone")


# ══════════════════════════════════════════════════════════════════════════════
#  Page Routes — Consumer Flow
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root(request: Request):
    if current_user_phone(request):
        return RedirectResponse("/home", status_code=302)
    return RedirectResponse("/register", status_code=302)


@app.get("/register")
async def register_page(request: Request):
    from data.hospitals import SUPPORTED_CITIES
    return templates.TemplateResponse("register.html", {
        "request": request,
        "cities": SUPPORTED_CITIES,
    })


@app.get("/home")
async def home_page(request: Request):
    phone = require_session(request)
    if not phone:
        return RedirectResponse("/register", status_code=302)
    from data.store import get_user, get_user_bookings
    user = get_user(phone) or {}
    analyses = user.get("analyses", [])
    bookings = get_user_bookings(phone)
    return templates.TemplateResponse("home.html", {
        "request": request,
        "user": user,
        "total_analyses": len(analyses),
        "active_bookings": len(bookings),
        "last_category": analyses[-1].get("category", "—") if analyses else "—",
    })


@app.get("/analyze")
async def analyze_page(request: Request):
    if not require_session(request):
        return RedirectResponse("/register", status_code=302)
    return templates.TemplateResponse("analyze.html", {"request": request})


@app.get("/results")
async def results_page(request: Request):
    phone = require_session(request)
    if not phone:
        return RedirectResponse("/register", status_code=302)
    from data.store import get_user
    user = get_user(phone) or {}
    return templates.TemplateResponse("results.html", {"request": request, "user": user})


@app.get("/booking")
async def booking_page(request: Request):
    phone = require_session(request)
    if not phone:
        return RedirectResponse("/register", status_code=302)
    from data.store import get_user
    from data.hospitals import get_hospitals_for_city
    user = get_user(phone) or {}
    hospitals = get_hospitals_for_city(user.get("city", "chennai"))
    return templates.TemplateResponse("booking.html", {
        "request": request,
        "user": user,
        "hospitals": hospitals,
    })


@app.get("/dashboard")
async def dashboard_page(request: Request):
    phone = require_session(request)
    if not phone:
        return RedirectResponse("/register", status_code=302)
    from data.store import get_user, get_user_bookings
    user = get_user(phone) or {}
    bookings = get_user_bookings(phone)
    analyses = user.get("analyses", [])
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "bookings": bookings,
        "analyses": analyses,
    })


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/register", status_code=302)


# ══════════════════════════════════════════════════════════════════════════════
#  API Routes — Consumer
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/register")
async def api_register(
    request: Request,
    name:    str = Form(),
    phone:   str = Form(),
    city:    str = Form(),
    address: str = Form(default=""),
):
    from data.store import save_user, get_user
    # Normalise phone
    phone = phone.strip().replace(" ", "").replace("-", "")
    user = {"name": name.strip(), "phone": phone, "city": city.lower().strip(), "address": address.strip()}
    save_user(user)
    request.session["user_phone"] = phone
    return JSONResponse({"status": "ok", "redirect": "/home"})


@app.post("/api/login")
async def api_login(request: Request, phone: str = Form()):
    from data.store import get_user
    phone = phone.strip().replace(" ", "").replace("-", "")
    user = get_user(phone)
    if not user:
        return JSONResponse({"status": "error", "message": "Phone not registered."}, status_code=404)
    request.session["user_phone"] = phone
    return JSONResponse({"status": "ok", "redirect": "/home"})


@app.get("/api/user/me")
async def api_user_me(request: Request):
    phone = current_user_phone(request)
    if not phone:
        return JSONResponse({"status": "error", "message": "Not logged in"}, status_code=401)
    from data.store import get_user
    return JSONResponse(get_user(phone) or {})


@app.get("/api/user/history")
async def api_user_history(request: Request):
    phone = current_user_phone(request)
    if not phone:
        return JSONResponse({"status": "error"}, status_code=401)
    from data.store import get_user
    user = get_user(phone) or {}
    return JSONResponse({"analyses": user.get("analyses", [])})


@app.post("/api/book-hospital")
async def api_book_hospital(request: Request):
    phone = current_user_phone(request)
    if not phone:
        return JSONResponse({"status": "error"}, status_code=401)
    data = await request.json()
    hospital_id = data.get("hospital_id")

    from data.hospitals import get_hospital_by_id
    from data.store import save_booking, add_analysis_to_user

    hospital = get_hospital_by_id(hospital_id)
    if not hospital:
        return JSONResponse({"status": "error", "message": "Hospital not found"}, status_code=404)

    # Generate unique token
    token = "VF-" + "".join(random.choices(string.digits, k=6))
    booking = {
        "token": token,
        "user_phone": phone,
        "hospital_id": hospital_id,
        "hospital_name": hospital["name"],
        "hospital_address": hospital["address"],
        "hospital_phone": hospital["phone"],
        "booked_at": datetime.now().isoformat(),
        "status": "Confirmed",
    }
    save_booking(booking)

    # Attach token to the latest analysis
    from data.store import get_user, _read, _write, USERS_FILE
    users = _read(USERS_FILE)
    if phone in users and users[phone].get("analyses"):
        users[phone]["analyses"][-1]["booking_token"] = token
        _write(USERS_FILE, users)

    # Simulate SMS
    print(f"[SMS SIM] To: +91{phone} — Your VitaFusion appointment token is {token} at {hospital['name']}. Show this at reception.")

    return JSONResponse({
        "status": "ok",
        "token": token,
        "hospital": hospital["name"],
        "address": hospital["address"],
        "phone": hospital["phone"],
        "message": f"Token {token} sent to +91{phone[-4:].zfill(10)}",
    })


# ══════════════════════════════════════════════════════════════════════════════
#  API Routes — Analysis
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/analyze")
async def api_analyze(
    request:        Request,
    clinical_data:  str  = Form(default="{}"),
    patient_info:   str  = Form(default="{}"),
    ecg_type:       str  = Form(default="normal"),
    demo_patient:   str  = Form(default=""),
    retinal_image:  Optional[UploadFile] = File(default=None),
    tongue_image:   Optional[UploadFile] = File(default=None),
    ecg_file:       Optional[UploadFile] = File(default=None),   # CSV signal
    ecg_image_file: Optional[UploadFile] = File(default=None),  # PNG/JPG image of ECG paper
):
    try:
        engine = get_engine()
        phone  = current_user_phone(request)

        # ── Parse clinical data ──────────────────────────────────────────────
        clinical = json.loads(clinical_data) if clinical_data and clinical_data != "{}" else {}
        patient  = json.loads(patient_info)  if patient_info  and patient_info  != "{}" else {}

        # Fill patient info from session user if not provided
        if phone and not patient:
            from data.store import get_user
            u = get_user(phone) or {}
            patient = {"name": u.get("name", ""), "phone": phone, "city": u.get("city", "")}

        # ── Images ───────────────────────────────────────────────────────────
        retinal_bytes   = await retinal_image.read() if (retinal_image and retinal_image.filename) else None
        tongue_bytes    = await tongue_image.read()  if (tongue_image  and tongue_image.filename)  else None
        ecg_bytes       = await ecg_file.read()      if (ecg_file      and ecg_file.filename)      else None
        ecg_image_bytes = await ecg_image_file.read() if (ecg_image_file and ecg_image_file.filename) else None

        # Auto-detect: if user uploads a PNG/JPG to the ecg_file slot, treat it as ecg_image
        if ecg_bytes and ecg_file and ecg_file.content_type and ecg_file.content_type.startswith("image/"):
            ecg_image_bytes = ecg_bytes
            ecg_bytes = None

        # Track which modalities were actually provided by the user
        user_provided = {
            "retinal":   retinal_bytes   is not None,
            "tongue":    tongue_bytes    is not None,
            "ecg":       ecg_bytes       is not None,
            "ecg_image": ecg_image_bytes is not None,
        }

        # ── Demo mode ─────────────────────────────────────────────────────────
        if demo_patient:
            from utils.synthetic_data import (generate_synthetic_retinal_image,
                                               generate_synthetic_tongue_image)
            seed_val = int(demo_patient[-1]) if demo_patient[-1].isdigit() else 1
            if retinal_bytes is None:
                retinal_bytes = generate_synthetic_retinal_image(dr_grade=1, seed=seed_val)
                user_provided["retinal"] = True
            if tongue_bytes is None:
                tongue_bytes  = generate_synthetic_tongue_image("Pale", seed=seed_val + 1)
                user_provided["tongue"] = True
            if ecg_bytes is None and ecg_image_bytes is None:
                from utils.synthetic_data import generate_synthetic_ecg
                ecg_bytes = generate_synthetic_ecg("normal", seed=42)
                user_provided["ecg"] = True

        # ── If absolutely no input, generate retinal+tongue so analysis works ─
        if retinal_bytes is None and tongue_bytes is None and ecg_bytes is None and ecg_image_bytes is None and not demo_patient:
            from utils.synthetic_data import (generate_synthetic_retinal_image,
                                               generate_synthetic_tongue_image)
            retinal_bytes = generate_synthetic_retinal_image(dr_grade=0, seed=10)
            tongue_bytes  = generate_synthetic_tongue_image("Normal", seed=11)
            user_provided["retinal"] = True
            user_provided["tongue"]  = True

        ecg_data = ecg_bytes  # None if not uploaded (ECG analysis skipped)

        # ── Run inference ─────────────────────────────────────────────────────
        results = engine.run(
            retinal_image = retinal_bytes,
            tongue_image  = tongue_bytes,
            ecg_data      = ecg_data,
            ecg_image     = ecg_image_bytes,   # NEW
            clinical_data = clinical,
            patient_info  = patient,
        )

        # ── Serialize & add health category ──────────────────────────────────
        response = _serialize_results(results)

        # risk_scores are 0-100 scale (already converted in VitaFusionSimulated)
        risk_scores = response.get("risk_scores", {})
        max_risk = max(risk_scores.values(), default=0.0) if risk_scores else 0.0

        # Thresholds on 0-100 scale
        if max_risk >= 70:
            category = "Immediate"
        elif max_risk >= 40:
            category = "Medium"
        else:
            category = "Good"

        # Mark which modality findings are real vs. not provided
        response["user_provided"] = user_provided
        # Remove ECG section from response if user didn't upload one (CSV or image)
        if not user_provided.get("ecg") and not user_provided.get("ecg_image") and not demo_patient:
            response.pop("ecg", None)

        response["health_category"] = category
        response["max_risk"] = round(float(max_risk), 3)
        response["analysis_id"] = "AN-" + "".join(random.choices(string.digits, k=8))
        response["timestamp"] = datetime.now().isoformat()

        _save_results(response)

        # ── Save to user history ──────────────────────────────────────────────
        if phone:
            from data.store import add_analysis_to_user
            snapshot = {
                "id": response["analysis_id"],
                "timestamp": response["timestamp"],
                "category": category,
                "max_risk": response["max_risk"],
                "risk_scores": risk_scores,
                "booking_token": None,
            }
            add_analysis_to_user(phone, snapshot)

        return JSONResponse({"status": "ok", "data": response})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/results")
async def api_results():
    """Return cached results from last analysis."""
    results = _load_results()
    if not results:
        return JSONResponse({"status": "empty", "data": None})
    return JSONResponse({"status": "ok", "data": results})


@app.post("/api/gradcam")
async def api_gradcam(request: Request):
    try:
        data       = await request.json()
        modality   = data.get("modality", "retinal")
        risk_score = float(data.get("risk_score", 0.5))
        img_b64    = data.get("image_b64")
        import numpy as np
        if img_b64:
            from PIL import Image as PILImage
            img_bytes = base64.b64decode(img_b64)
            pil_img   = PILImage.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
            img_arr   = np.array(pil_img)
        else:
            from utils.synthetic_data import generate_synthetic_retinal_image, generate_synthetic_tongue_image
            img_arr = (generate_synthetic_retinal_image(dr_grade=1, seed=1)
                       if modality == "retinal" else
                       generate_synthetic_tongue_image("Pale", seed=2))
        from xai.gradcam import simulate_gradcam, overlay_heatmap
        heatmap = simulate_gradcam(img_arr, risk_score, modality)
        overlay = overlay_heatmap(img_arr, heatmap, alpha=0.5)
        def arr_to_b64(arr):
            from PIL import Image as PILImage
            buf = io.BytesIO()
            PILImage.fromarray(arr.astype("uint8")).save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
        return JSONResponse({"original_b64": arr_to_b64(img_arr), "heatmap_b64": arr_to_b64((heatmap * 255).astype("uint8")), "overlay_b64": arr_to_b64(overlay)})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/ecg-plot")
async def api_ecg_plot(request: Request):
    try:
        data       = await request.json()
        ecg_preds  = data.get("ecg_preds", {})
        import numpy as np
        signal     = np.array(ecg_preds.get("signal_norm", []))
        r_peaks    = np.array(ecg_preds.get("r_peaks", []))
        arr_class  = ecg_preds.get("arrhythmia_class", "Normal Sinus Rhythm")
        arr_prob   = float(ecg_preds.get("arrhythmia_prob", 0.8))
        fs         = float(ecg_preds.get("sample_rate", 250))
        lead_names = ecg_preds.get("lead_names", ["II"])
        ref = signal[min(1, signal.shape[0] - 1)] if signal.ndim == 2 else signal
        from xai.ecg_attribution import compute_ecg_attribution_simulated, plot_ecg_with_attribution, plot_ecg_12_lead
        attribution = compute_ecg_attribution_simulated(ref, r_peaks, arr_class, arr_prob, fs)
        plot_png    = plot_ecg_with_attribution(ref, attribution, r_peaks, lead_name=lead_names[min(1, len(lead_names) - 1)], fs=fs, n_seconds=6, arrhythmia_class=arr_class)
        plot12_png  = plot_ecg_12_lead(signal if signal.ndim == 2 else signal[np.newaxis, :], lead_names, r_peaks, fs=fs, n_seconds=2.5)
        return JSONResponse({"attribution_plot": base64.b64encode(plot_png).decode(), "lead12_plot": base64.b64encode(plot12_png).decode()})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/shap")
async def api_shap(request: Request):
    try:
        data         = await request.json()
        importances  = data.get("feature_importances", {})
        base_val     = float(data.get("base_value", 0.25))
        output_val   = float(data.get("output_value", 0.55))
        feature_vals = data.get("feature_values", {})
        from xai.shap_explainer import compute_shap_values_simulated, plot_shap_waterfall
        shap_data = compute_shap_values_simulated(importances, base_val, output_val)
        png_bytes = plot_shap_waterfall(shap_data, feature_vals)
        return JSONResponse({"shap_plot": base64.b64encode(png_bytes).decode()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/generate-report")
async def api_generate_report(request: Request):
    try:
        data = await request.json()
        from llm.report_generator import ReportGenerator
        from llm.prompt_templates import build_clinical_report_prompt
        prompt = build_clinical_report_prompt(
            patient_info=data.get("patient_info", {}), risk_scores=data.get("risk_scores", {}),
            confidence_scores=data.get("confidence_scores", {}), retinal_findings=data.get("retinal_findings"),
            tongue_findings=data.get("tongue_findings"), ecg_findings=data.get("ecg_findings"),
            clinical_features=data.get("clinical_features"), feature_importances=data.get("feature_importances"),
            report_style=data.get("report_style", "clinical"),
        )
        generator = ReportGenerator()
        report    = generator.generate_report(prompt, patient_info=data.get("patient_info", {}), risk_scores=data.get("risk_scores", {}))
        return JSONResponse({"status": "ok", "report": report})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ── Legacy demo & PTB-XL routes ───────────────────────────────────────────────

@app.get("/api/demo-patients")
async def api_demo_patients():
    from utils.synthetic_data import get_all_demo_patients
    return JSONResponse(get_all_demo_patients())


@app.get("/api/demo/{patient_id}")
async def api_demo_patient(patient_id: str):
    from utils.synthetic_data import get_demo_patient
    return JSONResponse(get_demo_patient(patient_id))


@app.get("/api/ptbxl/stats")
async def api_ptbxl_stats():
    try:
        from preprocessing.ptbxl_loader import compute_dataset_stats
        import config
        return JSONResponse(compute_dataset_stats(config.PTBXL_DIR))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"available": False, "error": str(e)})


@app.get("/api/ptbxl/sample")
async def api_ptbxl_sample(ecg_id: Optional[int] = None, fs: int = 100):
    try:
        from preprocessing.ptbxl_loader import load_sample_ecg
        import config
        return JSONResponse(load_sample_ecg(config.PTBXL_DIR, ecg_id=ecg_id, sampling_rate=fs))
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e), "available": False}, status_code=404)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/ptbxl/search")
async def api_ptbxl_search(scp_code: str = "", superclass: str = "", limit: int = 20):
    try:
        from preprocessing.ptbxl_loader import load_ptbxl_metadata, VITAFUSION_CLASSES
        import config
        df, _ = load_ptbxl_metadata(config.PTBXL_DIR)
        if scp_code:
            df = df[df["scp_codes"].apply(lambda d: any(k.upper() == scp_code.upper() for k in d))]
        if superclass:
            df = df[df["diagnostic_superclass"] == superclass.upper()]
        results = [{"ecg_id": int(i), "vitafusion_class": VITAFUSION_CLASSES[int(r.get("vitafusion_label", 0))],
                    "diagnostic_superclass": str(r.get("diagnostic_superclass", "NORM")),
                    "age": float(r.get("age", 0)), "sex": "Female" if int(r.get("sex", 0)) == 1 else "Male",
                    "strat_fold": int(r.get("strat_fold", 1))} for i, r in df.head(limit).iterrows()]
        return JSONResponse({"results": results, "total": len(results)})
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e), "available": False}, status_code=404)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: Serialize numpy results
# ══════════════════════════════════════════════════════════════════════════════

def _serialize_results(results: dict) -> dict:
    import numpy as np

    def _convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict):        return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_convert(i) for i in obj]
        return obj

    out    = {}
    fusion = results.get("fusion_results", {})
    out["risk_scores"]       = _convert(fusion.get("risk_scores", {}))
    out["confidence_scores"] = _convert(fusion.get("confidence_scores", {}))
    out["n_modalities"]      = fusion.get("n_modalities_used", 0)
    out["patient_info"]      = _convert(results.get("patient_info", {}))

    ret = results.get("retinal_preds")
    if ret:
        out["retinal"] = {"dr_grade_label": ret.get("dr_grade_label"), "dr_probs": _convert(ret.get("dr_probs", [])),
                          "vessel_score": float(ret.get("vessel_score", 0)), "optic_score": float(ret.get("optic_score", 0)),
                          "macula_score": float(ret.get("macula_score", 0))}
    tng = results.get("tongue_preds")
    if tng:
        out["tongue"] = {"tongue_class": tng.get("tongue_class"), "tongue_probs": _convert(tng.get("tongue_probs", [])),
                         "coating_score": float(tng.get("coating_score", 0)), "moisture_score": float(tng.get("moisture_score", 0))}
    ecg = results.get("ecg_preds")
    if ecg:
        out["ecg"] = {"arrhythmia_class": ecg.get("arrhythmia_class"), "arrhythmia_idx": int(ecg.get("arrhythmia_idx", 0)),
                      "arrhythmia_prob": float(ecg.get("arrhythmia_probs", [0])[ecg.get("arrhythmia_idx", 0)]),
                      "arrhythmia_probs": _convert(ecg.get("arrhythmia_probs", [])), "heart_rate": float(ecg.get("heart_rate", 75)),
                      "hrv_ms": float(ecg.get("hrv_ms", 35)), "pr_interval": float(ecg.get("pr_interval", 160)),
                      "qrs_duration": float(ecg.get("qrs_duration", 90)), "qtc_interval": float(ecg.get("qtc_interval", 420)),
                      "signal_norm": _convert(ecg.get("signal_norm", [])), "r_peaks": _convert(ecg.get("r_peaks", [])),
                      "sample_rate": float(ecg.get("sample_rate", 250)), "lead_names": ecg.get("lead_names", []),
                      "quality": float(ecg.get("quality", 0.8)), "hrv_features": _convert(ecg.get("hrv_features", {}))}
    cli     = results.get("clinical_preds")
    preproc = results.get("preprocessing", {}).get("clinical", {})
    if cli:
        out["clinical"] = {"cardiovascular_risk": float(cli.get("cardiovascular_risk", 0)),
                           "diabetes_risk": float(cli.get("diabetes_risk", 0)),
                           "hypertension_risk": float(cli.get("hypertension_risk", 0)),
                           "arrhythmia_risk": float(cli.get("arrhythmia_risk", 0)),
                           "metabolic_risk": float(cli.get("metabolic_risk", 0)),
                           "feature_importances": _convert(cli.get("feature_importances", {})),
                           "derived": _convert(preproc.get("derived", {})) if preproc else {}}
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
