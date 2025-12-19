# app.py
import io
import base64
import os
import time
from typing import Tuple, Optional

import requests
from PIL import Image
import streamlit as st

st.set_page_config("Illegal Waste (light) — Classifier", layout="wide")

st.title("Illegal Waste — lightweight deploy (no YOLO)")

st.markdown(
    """
This app avoids heavy YOLO/ultralytics installs.  
**Preferred**: use Hugging Face Inference API (no heavy native libs).  
**Fallback**: local `torchvision` model if `torch` is installed in your environment.
"""
)

# Helper: read secrets or user input token
HF_TOKEN_FROM_SECRETS = st.secrets.get("HF_API_TOKEN") if hasattr(st, "secrets") else None

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Settings")
    st.text_input("Model filename (ignored for HF mode)", value="best.pt", key="model_filename")
    st.write("Mode:")
    use_hf_default = True
    mode = st.radio(
        "Run inference via",
        ("Hugging Face API (recommended)", "Local torchvision (if available)", "Heuristic fallback"),
        index=0 if use_hf_default else 1,
    )

    hf_token = HF_TOKEN_FROM_SECRETS or st.text_input(
        "Hugging Face API token (paste or add to Streamlit Secrets as HF_API_TOKEN)",
        value="",
        placeholder="hf_xxx...",
        type="password",
    )
    hf_model = st.text_input(
        "Hugging Face model id (image-classification)",
        value="google/vit-base-patch16-224",
        help="Examples: google/vit-base-patch16-224, microsoft/resnet-50",
    )

    st.markdown("---")
    st.write("Confidence threshold (UI only)")
    conf_thr = st.slider("Confidence threshold", 0.0, 1.0, 0.25)

with col2:
    st.header("Upload image")
    uploaded = st.file_uploader("Image (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image", use_column_width=True)
        except Exception as e:
            st.error(f"Could not open uploaded image: {e}")
            uploaded = None

# --- Inference helpers ---

def call_huggingface_classify(image_bytes: bytes, model: str, token: str, timeout=30):
    """
    Calls Hugging Face Inference API to get image classification.
    Sends raw image bytes to the model endpoint.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://api-inference.huggingface.co/models/{model}"
    try:
        resp = requests.post(url, headers=headers, data=image_bytes, timeout=timeout)
    except Exception as e:
        return {"error": f"Request error: {e}"}
    if resp.status_code == 503:
        return {"error": "Model is loading on HF (503). Try again in a bit."}
    if resp.status_code >= 400:
        return {"error": f"HF API error {resp.status_code}: {resp.text}"}
    try:
        return resp.json()
    except Exception as e:
        return {"error": f"Invalid JSON from HF: {e}"}

# Local torchvision fallback (only used if torch is installed)
def local_torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except Exception:
        return False

@st.cache_resource
def load_local_model():
    """
    Load a small torchvision model (ResNet18 pre-trained).
    This will only be called if torch is available in the environment.
    """
    import torch
    from torchvision import models, transforms

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace last fc with 2-class head if you have a custom trained model.
    # Here we keep original imagenet classes for demo.
    model.eval()
    preproc = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return model, preproc

def classify_local(img: Image.Image) -> Tuple[str, float]:
    """
    Run local torchvision model and return best label + prob.
    Uses ImageNet labels (for demo). If you have a custom local model,
    replace this loading and mapping accordingly.
    """
    import torch
    import json
    from torchvision import transforms

    model, preproc = load_local_model()
    x = preproc(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        top_idx = int(probs.argmax().item())
        top_score = float(probs[top_idx].item())

    # Try to load ImageNet labels (optional local file or remote)
    imagenet_labels = None
    labels_path = os.path.join(os.path.dirname(__file__), "imagenet_labels.txt")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            imagenet_labels = [l.strip() for l in f.readlines()]
    if imagenet_labels and top_idx < len(imagenet_labels):
        label = imagenet_labels[top_idx]
    else:
        label = f"ImageNet class #{top_idx}"
    return label, top_score

def heuristic_classifier(img: Image.Image) -> Tuple[str, float]:
    """
    Dummy heuristic classifier that checks 'trash-like' colors / shapes by simple
    heuristics. USE ONLY as placeholder so UI remains functional.
    """
    import numpy as np
    arr = np.array(img.resize((128, 128))).astype("int32")
    # Quick heuristic: lots of gray/white pixels may indicate cloth/bag
    grayness = np.mean(np.abs(arr - arr.mean(axis=2, keepdims=True)))
    brightness = arr.mean()
    score = max(0.0, min(1.0, (200 - grayness) / 200.0))  # arbitrary
    label = "Illegal Waste (heuristic)" if score > 0.5 else "No Illegal Waste (heuristic)"
    return label, float(score)

# --- Run inference based on selected mode ---
if uploaded:
    img_bytes = uploaded.getvalue()
    result_msg = None
    if mode.startswith("Hugging Face"):
        if not hf_token:
            st.warning("No HF token provided. Add HF_API_TOKEN in Streamlit Secrets or paste above to use HF API.")
            # allow user to continue to local/heuristic fallback
        else:
            with st.spinner("Sending image to Hugging Face Inference API..."):
                hf_resp = call_huggingface_classify(img_bytes, hf_model, hf_token)
            if isinstance(hf_resp, dict) and hf_resp.get("error"):
                st.error(f"Hugging Face error: {hf_resp['error']}")
                result_msg = None
            else:
                # HF returns list of {label,score}
                preds = hf_resp
                if isinstance(preds, dict) and "error" in preds:
                    st.error(f"Hugging Face error: {preds['error']}")
                else:
                    if len(preds) == 0:
                        st.warning("No predictions returned from HF model.")
                    else:
                        top = preds[0]
                        label = top.get("label", "unknown")
                        score = float(top.get("score", 0.0))
                        st.success(f"Prediction (HF): **{label}** — {score:.2f}")
                        st.progress(min(1.0, score))
                        st.json(preds[:5])
                        result_msg = (label, score)

    if result_msg is None and mode.startswith("Local"):
        if local_torch_available():
            try:
                with st.spinner("Running local torchvision model..."):
                    label, score = classify_local(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                    st.success(f"Prediction (local): **{label}** — {score:.2f}")
                    st.progress(min(1.0, score))
                    result_msg = (label, score)
            except Exception as e:
                st.error(f"Local model failed: {e}")
        else:
            st.error("Local torchvision/torch not available in this environment.")

    if result_msg is None and mode.startswith("Heuristic"):
        with st.spinner("Running heuristic fallback..."):
            label, score = heuristic_classifier(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            st.info(f"Prediction (heuristic): **{label}** — {score:.2f}")
            st.progress(min(1.0, score))
            result_msg = (label, score)

    # If no mode produced a result yet, attempt fallback order: HF -> local -> heuristic
    if result_msg is None and mode.startswith("Hugging Face"):
        # try local
        if local_torch_available():
            try:
                with st.spinner("Trying local model as fallback..."):
                    label, score = classify_local(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                    st.success(f"Fallback local prediction: **{label}** — {score:.2f}")
                    result_msg = (label, score)
            except Exception as e:
                st.error(f"Fallback local model failed: {e}")
        if result_msg is None:
            with st.spinner("Using heuristic fallback..."):
                label, score = heuristic_classifier(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                st.info(f"Fallback heuristic: **{label}** — {score:.2f}")
                result_msg = (label, score)

    # final result block
    if result_msg:
        label, score = result_msg
        st.write("---")
        st.markdown(f"**Final:** {label} — confidence {score:.2f}")
        st.write("Confidence threshold (UI):", conf_thr)
