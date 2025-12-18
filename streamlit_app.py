#############################################################
# Streamlit app for Illegal Waste Detection (Ultralytics YOLO)
# - Lazy YOLO imports (prevents build-time crashes)
# - PIL-based drawing (no cv2 required)
# - Optional Hugging Face Hub model download
# - Robust error handling for Spaces
#############################################################

import os
import traceback
import importlib
from typing import Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Optional Hugging Face Hub downloader
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


# -------------------- Streamlit config --------------------
st.set_page_config(page_title="Illegal Waste Detection", layout="wide")
st.title("🚮 Illegal Waste Detection (YOLO)")
st.write(
    "Upload an image to detect illegal dumping regions. "
    "Model loading is lazy and safe for Hugging Face Spaces."
)

# -------------------- Sidebar --------------------
st.sidebar.header("Settings")
model_filename = st.sidebar.text_input(
    "Model filename (repo root)", "best.pt"
)


# -------------------- YOLO import helper --------------------
def try_import_yolo() -> Tuple[Optional[type], list]:
    """
    Try multiple ultralytics import paths.
    Returns (YOLO_class, error_list)
    """
    errors = []
    YOLO = None

    # Attempt 1: official API
    try:
        mod = importlib.import_module("ultralytics")
        YOLO = getattr(mod, "YOLO", None)
        if YOLO is not None:
            return YOLO, errors
        errors.append(("ultralytics", "YOLO attribute missing"))
    except Exception as e:
        errors.append(("from ultralytics import YOLO", repr(e)))

    # Attempt 2: older submodule path
    try:
        mod2 = importlib.import_module("ultralytics.yolo.engine.model")
        YOLO = getattr(mod2, "YOLO", None)
        if YOLO is not None:
            return YOLO, errors
        errors.append(("ultralytics.yolo.engine.model", "YOLO missing"))
    except Exception as e:
        errors.append(("ultralytics.yolo.engine.model", repr(e)))

    return None, errors


# -------------------- Lazy model loader --------------------
@st.cache_resource
def load_model(path: str):
    """
    Returns (model, error_message)
    """
    YOLO_class, import_errors = try_import_yolo()

    if YOLO_class is None:
        msg = ["❌ Unable to import YOLO class:"]
        for name, err in import_errors:
            msg.append(f"- {name}: {err}")
        return None, "\n".join(msg)

    # If model not found locally, try HF Hub download
    if not os.path.exists(path):
        hf_repo = os.getenv("MODEL_HF_REPO")
        hf_filename = os.getenv("MODEL_HF_FILENAME", path)

        if hf_repo and hf_hub_download is not None:
            try:
                st.sidebar.info(f"Downloading model from HF repo: {hf_repo}")
                path = hf_hub_download(
                    repo_id=hf_repo,
                    filename=hf_filename
                )
            except Exception as e:
                return None, f"HF model download failed: {repr(e)}"
        else:
            return None, f"Model file '{path}' not found."

    try:
        model = YOLO_class(path)
        return model, None
    except Exception as e:
        return None, traceback.format_exc()


# -------------------- Sidebar model status --------------------
st.sidebar.markdown("## Model Loader")

if st.sidebar.button("Load Model Now"):
    model, err = load_model(model_filename)
    if model is None:
        st.sidebar.error("Model NOT loaded")
        st.sidebar.text_area("Error details", err, height=220)
    else:
        st.sidebar.success("Model loaded successfully")

# Quick YOLO availability check
yo, yo_errs = try_import_yolo()
if yo is None:
    st.sidebar.error("YOLO not available in environment")
    st.sidebar.text_area(
        "YOLO import debug",
        "\n".join([f"{n}: {e}" for n, e in yo_errs]),
        height=200
    )
else:
    st.sidebar.success("YOLO class detected")


# -------------------- Main UI --------------------
uploaded_file = st.file_uploader(
    "Upload image (jpg / jpeg / png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Upload an image to run detection.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded image", use_column_width=True)

# -------------------- Run detection --------------------
if st.button("🔍 Run Detection"):
    model, err = load_model(model_filename)

    if model is None:
        st.error("Model not loaded")
        st.text(err)
        st.stop()

    try:
        np_img = np.array(image)

        # Ultralytics inference
        try:
            results = model.predict(np_img)
        except Exception:
            results = model(np_img)

        result = results[0]
        boxes = getattr(result, "boxes", None)

        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        count = 0

        if boxes is not None and hasattr(boxes, "xyxy"):
            xyxy = boxes.xyxy
            confs = boxes.conf

            if hasattr(xyxy, "cpu"):
                xyxy = xyxy.cpu().numpy()
            if hasattr(confs, "cpu"):
                confs = confs.cpu().numpy()

            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = map(float, box)
                conf = float(confs[i]) if confs is not None else 0.0
                count += 1

                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=3
                )
                draw.text(
                    (x1 + 3, y1 + 3),
                    f"{conf * 100:.1f}%",
                    fill="red",
                    font=font
                )

        st.image(
            draw_img,
            caption=f"Detections: {count}",
            use_column_width=True
        )

        if count == 0:
            st.success("✅ No illegal waste detected.")
        else:
            st.error(f"🚨 Illegal waste detected in {count} region(s).")

    except Exception as e:
        st.error("Detection failed")
        st.text(repr(e))
        with st.expander("Full traceback"):
            st.text(traceback.format_exc())

# -------------------- End of app --------------------
