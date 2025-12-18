# streamlit_app.py
"""
Streamlit app for Illegal Waste Detection (YOLOv8)
- Lazy-imports ultralytics to avoid import-time failures
- Uses PIL for drawing (no opencv cv2 required)
- Safe for Streamlit Cloud: model is loaded lazily with st.cache_resource
"""
import os
import traceback
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

st.set_page_config(page_title="Illegal Waste Detection", layout="wide")
st.title("🚮 Illegal Waste Detection (YOLO)")

# -------------------------
# Helper: try to import YOLO lazily
# -------------------------
def try_import_yolo():
    """
    Try several import paths for ultralytics.YOLO without raising at import time.
    Return (YOLO_class_or_None, list_of_error_strings)
    """
    errors = []
    YOLO = None

    # Try top-level ultralytics
    try:
        import importlib
        mod = importlib.import_module("ultralytics")
        YOLO = getattr(mod, "YOLO", None)
        if YOLO:
            return YOLO, errors
        errors.append("ultralytics found but YOLO attribute missing")
    except Exception as e:
        errors.append(f"ultralytics import failed: {repr(e)}")

    # Try common internal path
    try:
        import importlib
        mod = importlib.import_module("ultralytics.yolo")
        if hasattr(mod, "YOLO"):
            return getattr(mod, "YOLO"), errors
        # try deeper
        try:
            mod2 = importlib.import_module("ultralytics.yolo.engine.model")
            if hasattr(mod2, "YOLO"):
                return getattr(mod2, "YOLO"), errors
        except Exception as ex2:
            pass
        errors.append("ultralytics.yolo present but YOLO not found")
    except Exception as e:
        errors.append(f"ultralytics.yolo import failed: {repr(e)}")

    return None, errors


# -------------------------
# Model loader (cached)
# -------------------------
@st.cache_resource
def load_model_from_path(path: str) -> Tuple[Optional[object], Optional[str]]:
    """
    Try to import YOLO and load the model file from `path`.
    Returns (model_object_or_None, error_message_or_None)
    """
    YOLO_class, import_errors = try_import_yolo()
    if YOLO_class is None:
        msg = "Unable to import YOLO. Import attempts:\n" + "\n".join(import_errors)
        return None, msg

    # Check file exists
    if not os.path.exists(path):
        return None, f"Model file '{path}' not found in repo root. Upload 'best.pt' or change the filename in the sidebar."

    try:
        model = YOLO_class(path)
        return model, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"Failed to initialize YOLO model from '{path}': {repr(e)}\n\nTraceback:\n{tb}"


# -------------------------
# Utility: draw predictions with PIL
# -------------------------
def draw_detections(image: Image.Image, boxes: List[dict], show_conf: bool = True, show_label: bool = True) -> Image.Image:
    """
    boxes: list of dicts with keys: x1,y1,x2,y2,cls (int or str), conf (float)
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()

    for b in boxes:
        x1, y1, x2, y2 = map(float, (b["x1"], b["y1"], b["x2"], b["y2"]))
        label = ""
        if show_label and ("name" in b and b["name"]):
            label = str(b["name"])
        if show_conf and ("conf" in b):
            label = f"{label} {b['conf']*100:.1f}%".strip()
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        # Draw label background
        if label:
            text_size = draw.textsize(label, font=font)
            text_bg = [x1, y1 - text_size[1] - 4, x1 + text_size[0] + 6, y1]
            draw.rectangle(text_bg, fill="red")
            draw.text((x1 + 3, y1 - text_size[1] - 2), label, fill="white", font=font)
    return image


# -------------------------
# Streamlit UI
# -------------------------
with st.sidebar:
    st.header("Settings")
    model_filename = st.text_input("Model filename (repo root)", value="best.pt")
    st.markdown("Model is loaded lazily. If ultralytics cannot be imported, add it to `requirements.txt`.")
    if st.button("Load Model Now"):
        model_obj, model_err = load_model_from_path(model_filename)
        if model_obj is None:
            st.error("Model NOT loaded.")
            st.text_area("Load errors (copy for debugging)", value=str(model_err), height=220)
        else:
            st.success("Model loaded successfully.")

# Quick check: can we import ultralytics?
YOLO_class, yo_errs = try_import_yolo()
if YOLO_class is None:
    st.sidebar.error("YOLO not available in environment")
    st.sidebar.text_area("YOLO import debug", value="\n".join(yo_errs), height=220)
else:
    st.sidebar.success("YOLO class available")

st.write("Upload an image (jpg / jpeg / png). Model loading is lazy — click **Load Model Now** in the sidebar to preload it.")
uploaded = st.file_uploader("Upload image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

# Confidence slider
st.sidebar.header("Detection")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
show_conf = st.sidebar.checkbox("Show confidence", value=True)
show_label = st.sidebar.checkbox("Show class labels", value=True)

# Main detection flow
if uploaded is None:
    st.info("Upload an image to run detection.")
else:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        # Load model lazily
        model_obj, model_err = load_model_from_path(model_filename)
        if model_obj is None:
            st.error("Cannot run detection: model not loaded.")
            st.text(model_err)
        else:
            # Run prediction
            try:
                # ultralytics YOLO .predict prefers either source path or numpy array
                # use predict with a confidence param if available
                kwargs = {}
                # older/newer ultralytics versions accept 'conf' as a param
                try:
                    results = model_obj.predict(np.array(image), conf=conf_threshold, verbose=False)
                except TypeError:
                    # fallback
                    results = model_obj.predict(np.array(image))
                # results may be a list-like; take first
                res = results[0] if isinstance(results, (list, tuple)) else results

                # Parse detections
                boxes_out = []
                # Different ultralytics versions: res.boxes may be available
                if hasattr(res, "boxes") and res.boxes is not None:
                    # res.boxes.xyxy, res.boxes.conf, res.boxes.cls
                    for i in range(len(res.boxes)):
                        try:
                            xyxy = res.boxes.xyxy[i].tolist()  # tensor -> list
                        except Exception:
                            # fallback if indexing differs
                            xyxy = [float(v) for v in res.boxes.xyxy[i]]
                        conf = float(res.boxes.conf[i].item()) if hasattr(res.boxes.conf[i], "item") else float(res.boxes.conf[i])
                        cls_id = int(res.boxes.cls[i].item()) if hasattr(res.boxes.cls[i], "item") else int(res.boxes.cls[i])
                        name = None
                        # model may expose .names or .model.names
                        if hasattr(model_obj, "names"):
                            names = getattr(model_obj, "names")
                            name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else (names[cls_id] if cls_id < len(names) else str(cls_id))
                        boxes_out.append({
                            "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3],
                            "conf": conf, "cls": cls_id, "name": name
                        })
                else:
                    # fallback: try res.boxes.xyxy as attribute or res.xyxyn / res.boxes.data
                    # if no detections
                    pass

                # Filter by confidence threshold again (in case predict didn't accept conf)
                boxes_out = [b for b in boxes_out if b.get("conf", 0) >= conf_threshold]

                # Draw and show
                if len(boxes_out) == 0:
                    st.success("No illegal waste detected.")
                else:
                    draw_img = image.copy()
                    draw_img = draw_detections(draw_img, boxes_out, show_conf=show_conf, show_label=show_label)
                    st.image(draw_img, caption=f"Detections ({len(boxes_out)})", use_column_width=True)

            except Exception as e:
                st.error("Error during detection. See debug below.")
                st.text(repr(e))
                with st.expander("Full traceback"):
                    st.text(traceback.format_exc())

    except Exception as e:
        st.error("Failed to read uploaded image.")
        st.text(repr(e))
        with st.expander("Full traceback"):
            st.text(traceback.format_exc())
