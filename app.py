##############################################################
# Robust Streamlit app for Illegal Waste Detection (YOLO)
# - NO top-level 'from ultralytics import YOLO' to avoid import-time crashes
# - Attempts lazy imports inside load_model()
# - Uses PIL for drawing (no cv2)
# - Shows clear errors/warnings in the app
##############################################################
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import importlib
import traceback

st.set_page_config(page_title="Illegal Waste Detection", layout="wide")
st.title("🚮 Illegal Waste Detection (YOLO)")
st.write("Upload an image to detect illegal dumping regions. Model loading is lazy — see sidebar for status.")

# Sidebar controls
st.sidebar.header("Settings")
model_filename = st.sidebar.text_input("Model filename (in repo root)", "best.pt")
st.sidebar.markdown("If YOLO class cannot be imported, follow instructions in the sidebar.")

# Helper: attempt to import YOLO in several known places
def try_import_yolo():
    """
    Return (YOLO_class, error_list)
    YOLO_class is None if import failed.
    error_list contains tuples (attempt, error_message)
    """
    errors = []
    YOLO = None

    # Attempt 1: common public API
    try:
        mod = importlib.import_module("ultralytics")
        YOLO = getattr(mod, "YOLO", None)
        if YOLO is not None:
            return YOLO, errors
        errors.append(("ultralytics module; attribute YOLO missing", "ultralytics loaded but YOLO attribute not present"))
    except Exception as e:
        errors.append(("from ultralytics import YOLO", repr(e)))

    # Attempt 2: submodule path used by some releases
    try:
        mod2 = importlib.import_module("ultralytics.yolo.engine.model")
        YOLO = getattr(mod2, "YOLO", None)
        if YOLO is not None:
            return YOLO, errors
        errors.append(("ultralytics.yolo.engine.model; YOLO attr missing", "module loaded but YOLO attribute missing"))
    except Exception as e:
        errors.append(("import ultralytics.yolo.engine.model", repr(e)))

    # Attempt 3: try alternative path
    try:
        mod3 = importlib.import_module("ultralytics.yolo")
        # try introspect deeper
        if hasattr(mod3, "YOLO"):
            return getattr(mod3, "YOLO"), errors
        # else try engine.model
        try:
            mod4 = importlib.import_module("ultralytics.yolo.engine.model")
            if hasattr(mod4, "YOLO"):
                return getattr(mod4, "YOLO"), errors
        except Exception:
            pass
        errors.append(("import ultralytics.yolo", "loaded but YOLO not found"))
    except Exception as e:
        errors.append(("import ultralytics.yolo", repr(e)))

    return None, errors


# Lazy-load model function
@st.cache_resource
def load_model(path):
    """
    Returns (model_object, error_message)
    model_object is None on failure.
    """
    YOLO_class, import_errors = try_import_yolo()
    if YOLO_class is None:
        # construct readable message
        msg_lines = ["Unable to import YOLO class. Import attempts:"]
        for name, err in import_errors:
            msg_lines.append(f"- {name}: {err}")
        return None, "\n".join(msg_lines)

    # YOLO class exists, try to load model file
    if not os.path.exists(path):
        return None, f"Model file '{path}' not found in repository root. Upload or place the .pt file and redeploy."

    try:
        model = YOLO_class(path)
        return model, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"YOLO import ok but failed to load model '{path}': {repr(e)}\nFull traceback:\n{tb}"


# Show status & attempt model load (only when user wants)
st.sidebar.markdown("## Model Loader")
if st.sidebar.button("Load Model Now"):
    model_obj, model_err = load_model(model_filename)
    if model_obj is None:
        st.sidebar.error("Model NOT loaded.")
        st.sidebar.text_area("Load errors (copy for debugging)", value=model_err, height=220)
        st.warning("Model is not loaded. Detection disabled until model is available.")
    else:
        st.sidebar.success("Model loaded successfully. You can run detection from main UI.")

# Show current quick check for YOLO availability (without loading model file)
yo, yo_errs = try_import_yolo()
if yo is None:
    st.sidebar.error("YOLO class not available in this environment.")
    st.sidebar.markdown("**Try these fixes:**")
    st.sidebar.markdown(
        "- Ensure `ultralytics` is in `requirements.txt`.\n"
        "- Consider pinning a stable Ultralytics version (`ultralytics==8.3.235`) in your `requirements.txt`.\n"
        "- If you previously created `cv2.py` in repo root to fake OpenCV, delete it (we do runtime patching)."
    )
    # show the errors for developer
    st.sidebar.text_area("YOLO import attempts (debug)", "\n".join([f"{n} : {e}" for n, e in yo_errs]), height=220)
else:
    st.sidebar.success("YOLO class detected in environment.")

# Main: upload image and run detection (only if model is loaded)
uploaded_file = st.file_uploader("Upload image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.info("Upload an image to run detection (model must be loaded first).")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    # Provide a button to run detection: it will attempt to load model lazily
    if st.button("🔍 Run Detection (lazy load model if not loaded)"):
        model_obj, model_err = load_model(model_filename)
        if model_obj is None:
            st.error("Cannot run detection: model not loaded.")
            st.text(model_err)
        else:
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
                st.image(draw_img, caption=f"Detections ({count})", use_column_width=True)
                if count == 0:
                    st.success("No illegal waste detected.")
                else:
                    st.error(f"Illegal waste detected in {count} region(s).")
            except Exception as e:
                st.error("Error during detection. See debug below.")
                st.text(repr(e))
                with st.expander("Full traceback"):
                    st.text(traceback.format_exc())

# End of app
