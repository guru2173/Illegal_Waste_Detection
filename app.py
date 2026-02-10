##############################################################
# Streamlit app: Illegal Waste Detection (YOLO)
# - avoids top-level `from ultralytics import YOLO`
# - lazy-imports Ultralytics inside load_model / try_import_yolo
# - does not import cv2 for inference or drawing (uses PIL)
# - prints helpful debug messages when imports fail
##############################################################
import os
import importlib
import traceback
from typing import Tuple, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# small env tweak (keeps OpenEXR disabled)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

st.set_page_config(page_title="Illegal Waste Detection", layout="wide")
st.title("🚮 Illegal Waste Detection (YOLO)")
st.write("Upload an image to detect illegal dumping regions. Model loading is lazy — see sidebar for status.")

# -------------------------
# Sidebar: settings & status
# -------------------------
st.sidebar.header("Settings")
model_filename = st.sidebar.text_input("Model filename (in repo root)", "best.pt")

# Add confidence slider to sidebar
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

st.sidebar.markdown("If YOLO class cannot be imported, follow instructions shown below.")

# -------------------------
# Utility: detect OpenCV debug info (no heavy import-time crash)
# -------------------------
def opencv_debug() -> str:
    """
    Try to import cv2 and return short diagnostic text.
    If import raises an ImportError (libGL missing etc), return that message.
    """
    try:
        import cv2
        ver = getattr(cv2, "__version__", "unknown")
        path = getattr(cv2, "__file__", "unknown")
        # check whether imshow exists (sometimes headless packages don't expose GUI functions)
        has_imshow = hasattr(cv2, "imshow")
        return f"cv2 version: {ver}\ncv2 file: {path}\nimshow available: {has_imshow}"
    except Exception as e:
        return f"cv2 import error: {repr(e)}"

# show OpenCV diagnostic to developer
st.sidebar.text_area("OpenCV debug", value=opencv_debug(), height=120)

# -------------------------
# Attempt to import YOLO (lazy)
# -------------------------
def try_import_yolo():
    """
    Attempt several import paths for the YOLO class.
    Returns (YOLO_class_or_None, list_of_errors)
    """
    errors = []

    # Attempt 1: public API
    try:
        ultralytics = importlib.import_module("ultralytics")
        YOLO_cls = getattr(ultralytics, "YOLO", None)
        if YOLO_cls is not None:
            return YOLO_cls, errors
        errors.append(("ultralytics module; YOLO attr missing", "ultralytics imported but YOLO attribute not present"))
    except Exception as e:
        errors.append(("from ultralytics import YOLO", repr(e)))

    # Attempt 2: submodule used by some versions
    try:
        mod = importlib.import_module("ultralytics.yolo.engine.model")
        YOLO_cls = getattr(mod, "YOLO", None)
        if YOLO_cls is not None:
            return YOLO_cls, errors
        errors.append(("ultralytics.yolo.engine.model; YOLO attr missing", "module loaded but YOLO attribute missing"))
    except Exception as e:
        errors.append(("import ultralytics.yolo.engine.model", repr(e)))

    # Attempt 3: other submodule path
    try:
        mod2 = importlib.import_module("ultralytics.yolo")
        YOLO_cls = getattr(mod2, "YOLO", None)
        if YOLO_cls is not None:
            return YOLO_cls, errors
        errors.append(("ultralytics.yolo; YOLO not found", "module loaded but YOLO attribute missing"))
    except Exception as e:
        errors.append(("import ultralytics.yolo", repr(e)))

    return None, errors

# -------------------------
# Lazy model loader (cached)
# -------------------------
@st.cache_resource
def load_model(path: str) -> Tuple[Optional[object], Optional[str]]:
    """
    Returns (model_object_or_None, error_message_or_None)
    """
    YOLO_cls, import_errors = try_import_yolo()
    if YOLO_cls is None:
        # build error message summarizing import attempts
        msg_lines = ["Unable to import YOLO class. Import attempts:"]
        for name, err in import_errors:
            msg_lines.append(f"- {name}: {err}")
        return None, "\n".join(msg_lines)

    # YOLO class present: try to load the .pt file
    if not os.path.exists(path):
        return None, f"Model file '{path}' not found in repo root. Upload it to repository root or change filename."

    try:
        model = YOLO_cls(path)  # instantiate model with weights
        return model, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"YOLO import ok but failed to load model '{path}': {repr(e)}\nFull traceback:\n{tb}"

# -------------------------
# Sidebar Model Loader UI
# -------------------------
st.sidebar.markdown("## Model Loader")
if st.sidebar.button("Load Model Now"):
    model_obj, model_err = load_model(model_filename)
    if model_obj is None:
        st.sidebar.error("Model NOT loaded.")
        st.sidebar.text_area("Load errors (copy for debugging)", value=model_err or "unknown", height=260)
    else:
        st.sidebar.success("Model loaded successfully. You can run detection from main UI.")

# quick check: indicate whether YOLO class is importable (without loading weights)
yo_cls, yo_errs = try_import_yolo()
if yo_cls is None:
    st.sidebar.error("⚠️ YOLO class NOT available.")
    st.sidebar.markdown("Try:\n- Ensure `ultralytics` is in `requirements.txt`.\n- Pin `ultralytics==8.3.235` in `requirements.txt`.\n- Delete any `cv2.py` file in repo root (it shadows real OpenCV).")
    # show debug lines
    st.sidebar.text_area("YOLO import attempts (debug)", "\n".join([f"{n} : {e}" for n, e in yo_errs]), height=200)
else:
    st.sidebar.success("YOLO class appears importable.")

# -------------------------
# Main UI: image upload + detection
# -------------------------
uploaded_file = st.file_uploader("Upload image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.info("Upload an image to run detection (model must be loaded first).")
else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error("Could not open uploaded file as image.")
        st.exception(e)
        st.stop()

    st.image(image, caption="Uploaded image", width=min(700, image.width))

    # Run detection button
    if st.button("🔍 Run Detection (lazy load model if not loaded)"):
        model_obj, model_err = load_model(model_filename)
        if model_obj is None:
            st.error("Cannot run detection: model not loaded.")
            st.text(model_err)
        else:
            # Run prediction (wrap in try/except)
            try:
                # ultralytics model.predict accepts numpy arrays
                # Passing conf=conf_threshold from sidebar slider
                results = model_obj.predict(np.array(image), conf=conf_threshold)
                
                if results is None or len(results) == 0:
                    st.warning("Model returned no results object.")
                else:
                    res0 = results[0]
                    boxes = getattr(res0, "boxes", None)
                    draw_img = image.copy()
                    draw = ImageDraw.Draw(draw_img)
                    count = 0

                    if boxes is None:
                        st.warning("No boxes attribute found on results — cannot draw detections.")
                        st.image(draw_img, caption="No detections", width=min(700, draw_img.width))
                    else:
                        # Try to extract coordinates and confidences robustly
                        try:
                            coords = getattr(boxes, "xyxy", None)
                            confs = getattr(boxes, "conf", None)
                            # convert to numpy arrays when needed
                            if coords is not None:
                                try:
                                    coords_arr = np.array(coords)
                                except Exception:
                                    # coords might be a torch tensor-like object
                                    coords_arr = np.array([list(map(float, c)) for c in coords])
                            else:
                                coords_arr = np.array([])

                            if confs is not None:
                                try:
                                    confs_arr = np.array(confs)
                                except Exception:
                                    confs_arr = np.array([float(c) for c in confs])
                            else:
                                confs_arr = None

                            # Draw rectangles
                            for i, xy in enumerate(coords_arr):
                                if len(xy) < 4:
                                    continue
                                x1, y1, x2, y2 = map(float, xy[:4])
                                conf = float(confs_arr[i]) if (confs_arr is not None and i < len(confs_arr)) else 0.0
                                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                                # choose a readable font size (fallback if not available)
                                try:
                                    font = ImageFont.truetype("DejaVuSans.ttf", size=16)
                                except Exception:
                                    font = None
                                txt = f"{conf*100:.1f}%"
                                draw.text((x1 + 4, y1 + 4), txt, fill="red", font=font)
                                count += 1

                        except Exception as inner_e:
                            st.error("Error processing detection boxes.")
                            st.exception(inner_e)

                        st.image(draw_img, caption=f"Detections ({count})", width=min(700, draw_img.width))
                        if count == 0:
                            st.success(f"No illegal waste detected at {conf_threshold*100:.0f}% confidence.")
                        else:
                            st.error(f"Illegal waste detected in {count} region(s) (threshold: {conf_threshold}).")
            except Exception as e:
                st.error("Error during model prediction. See debug below.")
                st.text(repr(e))
                with st.expander("Full traceback"):
                    st.text(traceback.format_exc())

# -------------------------
# Footer notes
# -------------------------
st.markdown("---")
st.caption("Note: This app lazy-loads the Ultralytics YOLO class. If you see import errors around `libGL.so.1`, you must install only headless OpenCV (opencv-python-headless) and remove opencv-python from requirements, and delete any local cv2.py file in the repo root.")
