##############################################################
# Robust Streamlit app for Illegal Waste Detection (YOLO)
# - NO top-level 'from ultralytics import YOLO'
# - Attempts lazy YOLO imports
# - Uses PIL (no cv2 required)
# - Debug messages visible in sidebar
##############################################################

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"   # Prevent OpenCV GL errors

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import traceback
import importlib


##############################################################
# STREAMLIT UI HEADER
##############################################################
st.set_page_config(page_title="Illegal Waste Detection", layout="wide")
st.title("🚮 Illegal Waste Detection (YOLO)")
st.write("Upload an image to detect illegal dumping regions. Model loading is lazy — see sidebar for status.")

##############################################################
# SIDEBAR SETTINGS
##############################################################
st.sidebar.header("Settings")
model_filename = st.sidebar.text_input("Model filename (in repo root)", "best.pt")


##############################################################
# TRY IMPORT YOLO IN DIFFERENT WAYS
##############################################################
def try_import_yolo():
    """
    Return (YOLO_class, error_list)
    YOLO_class = None if import failed.
    """
    errors = []
    YOLO = None

    # Try main ultralytics import
    try:
        mod = importlib.import_module("ultralytics")
        YOLO = getattr(mod, "YOLO", None)
        if YOLO is not None:
            return YOLO, errors
        errors.append(("ultralytics: YOLO missing", "ultralytics loaded but no YOLO attr"))
    except Exception as e:
        errors.append(("from ultralytics import YOLO", repr(e)))

    # Try engine path
    try:
        mod2 = importlib.import_module("ultralytics.yolo.engine.model")
        YOLO = getattr(mod2, "YOLO", None)
        if YOLO is not None:
            return YOLO, errors
        errors.append(("ultralytics.yolo.engine.model", "loaded but no YOLO attr"))
    except Exception as e:
        errors.append(("import ultralytics.yolo.engine.model", repr(e)))

    return None, errors


##############################################################
# LAZY MODEL LOADING
##############################################################
@st.cache_resource
def load_model(path):
    YOLO_class, import_errors = try_import_yolo()
    if YOLO_class is None:
        msg = ["Unable to import YOLO:"]
        for name, err in import_errors:
            msg.append(f"- {name}: {err}")
        return None, "\n".join(msg)

    if not os.path.exists(path):
        return None, f"Model '{path}' not found in repo root!"

    try:
        model = YOLO_class(path)
        return model, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"Model load error: {repr(e)}\n{tb}"


##############################################################
# MODEL LOADER BUTTON
##############################################################
st.sidebar.markdown("## Model Loader")
if st.sidebar.button("Load Model Now"):
    model_obj, model_err = load_model(model_filename)
    if model_obj is None:
        st.sidebar.error("❌ Model NOT loaded.")
        st.sidebar.text_area("Load errors", value=model_err, height=220)
        st.warning("Model not loaded.")
    else:
        st.sidebar.success("✅ Model loaded successfully.")


##############################################################
# YOLO AVAILABILITY CHECK (debug)
##############################################################
yo, yo_errs = try_import_yolo()
if yo is None:
    st.sidebar.error("⚠️ YOLO class NOT available.")
    st.sidebar.markdown("Try:")
    st.sidebar.markdown("- Make sure `ultralytics` is in requirements.txt")
    st.sidebar.markdown("- Pin: `ultralytics==8.3.235` ")
    st.sidebar.text_area("Debug", "\n".join([f"{n}: {e}" for n, e in yo_errs]), height=200)
else:
    st.sidebar.success("YOLO detected.")


##############################################################
# MAIN FILE UPLOAD
##############################################################
uploaded_file = st.file_uploader("Upload image (jpg,jpeg,png)", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.info("Upload an image to run detection.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded image", use_column_width=True)


##############################################################
# RUN DETECTION
##############################################################
if st.button("🔍 Run Detection"):
    model_obj, model_err = load_model(model_filename)
    if model_obj is None:
        st.error("Cannot run detection: model not loaded")
        st.text(model_err)
        st.stop()

    try:
        results = model_obj.predict(np.array(image))
        detections = results[0]

        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)
        count = 0

        for box in detections.boxes:
            count += 1
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"{conf*100:.1f}%", fill="red")

        st.image(draw_img, caption=f"Detections: {count}", use_column_width=True)

        if count == 0:
            st.success("No illegal waste detected.")
        else:
            st.error(f"Detected {count} illegal waste region(s).")

    except Exception as e:
        st.error("Error during detection!")
        st.text(repr(e))
        st.text(traceback.format_exc())
