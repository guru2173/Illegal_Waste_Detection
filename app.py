##############################################################
# Robust Streamlit app for Illegal Waste Detection (YOLO)
# - Patches cv2 BEFORE ultralytics import
# - Tries multiple import paths for YOLO
# - Uses PIL for drawing (no cv2 GUI)
##############################################################

# ------------------ 1) Fake / patch cv2 BEFORE anything else ------------------
import sys, types

cv2_fake = types.SimpleNamespace()
# constants and methods Ultralytics expects
cv2_fake.IMREAD_COLOR = 1
cv2_fake.setNumThreads = lambda *a, **k: None
cv2_fake.getNumThreads = lambda *a, **k: 1
cv2_fake.imread = lambda *a, **k: None
cv2_fake.imwrite = lambda *a, **k: None
cv2_fake.imshow = lambda *a, **k: None
cv2_fake.waitKey = lambda *a, **k: None

# Register fake module so any 'import cv2' returns this object
sys.modules['cv2'] = cv2_fake

# ------------------ 2) Environment safe-guards (no GUI / single-thread) -----
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FORCE_CPU"] = "1"                 # encourage CPU usage for ultralytics
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["DISPLAY"] = "0"

# ------------------ 3) Imports (after cv2 patch) ----------------------------
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import importlib
import traceback
import os.path

st.set_page_config(page_title="Illegal Waste Detection", layout="wide")
st.title("🚮 Illegal Waste Detection (YOLO)")
st.write("Upload an image to detect illegal dumping regions. This app attempts a robust import of ultralytics YOLO.")

# ------------------ 4) Robust attempt to import YOLO ------------------------
YOLO = None
_import_errors = []

try:
    # primary API (most common)
    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
except Exception as e:
    _import_errors.append(("from ultralytics import YOLO", str(e)))
    # try fallback path used in some installs
    try:
        mod = importlib.import_module("ultralytics.yolo.engine.model")
        YOLO = getattr(mod, "YOLO", None)
    except Exception as e2:
        _import_errors.append(("ultralytics.yolo.engine.model import", str(e2)))
        # try loading ultralytics module then attribute
        try:
            ul = importlib.import_module("ultralytics")
            YOLO = getattr(ul, "YOLO", None)
            if YOLO is None:
                _import_errors.append(("ultralytics module loaded but YOLO attr missing", "YOLO attribute not found"))
        except Exception as e3:
            _import_errors.append(("import ultralytics module", str(e3)))

# If YOLO still None, show error and debug info
if YOLO is None:
    st.error("Critical: Could not import YOLO class from the installed ultralytics package.")
    st.markdown("**What I tried:**")
    for name, err in _import_errors:
        st.text(f"- {name}: {err}")
    st.markdown(
        """
        **How to fix (choose one):**
        - Ensure `ultralytics` is installed in the environment: `pip install ultralytics`
        - Try a known-good version: `pip install ultralytics==8.3.235` (or a stable release)
        - If using Streamlit Cloud, make sure `requirements.txt` contains only:
          ```
          streamlit
          ultralytics
          numpy
          Pillow
          ```
        - If you previously added a `cv2.py` file in repo delete it (we patch cv2 in runtime).
        - Check server logs for more detailed Python tracebacks.
        """
    )
    # show full traceback to help debugging (collapsible)
    with st.expander("Full traceback and debug (developer)"):
        st.text(traceback.format_exc())
    st.stop()  # halt app because YOLO is required to proceed

# ------------------ 5) Model loading helper --------------------------------
@st.cache_resource
def load_yolo_model(path="best.pt"):
    if not os.path.exists(path):
        return None, f"Model file '{path}' not found in repo root."
    try:
        model = YOLO(path)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {e}"

# ------------------ 6) UI: upload + detect ---------------------------------
st.sidebar.header("Model & Status")
model_path = st.sidebar.text_input("Model filename (in repo root)", value="best.pt")
st.sidebar.write("YOLO import OK")

model, load_err = load_yolo_model(model_path)
if model is None:
    st.sidebar.warning(f"Model not loaded: {load_err}")
    st.warning("Model file not found or failed to load. Upload or place `best.pt` in repo root and restart.")
else:
    st.sidebar.success("Model loaded")

uploaded_file = st.file_uploader("Upload image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload an image to start detection.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("🔍 Run Detection"):
        try:
            # run prediction
            results = model.predict(np.array(image))
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

            st.image(draw_img, caption=f"Detections ({count})", use_column_width=True)
            if count == 0:
                st.success("No illegal waste detected.")
            else:
                st.error(f"Illegal waste detected in {count} region(s).")

        except Exception as e:
            st.error("Error during detection. See debug below.")
            st.text(str(e))
            with st.expander("Traceback"):
                st.text(traceback.format_exc())
