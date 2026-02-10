##############################################################
# Streamlit app: Illegal Waste Detection (YOLO)
# - avoids top-level `from ultralytics import YOLO`
# - lazy-imports Ultralytics inside load_model / try_import_yolo
# - does not import cv2 for inference or drawing (uses PIL)
# - prints helpful debug messages when imports fail
##############################################################
import os
import sys
import subprocess
import importlib
import importlib.util
import traceback
from typing import Tuple, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Ensure Ultralytics can write settings (avoids "not writable" warning)
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

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
# Enhanced runtime environment / import debug helper (temporary)
# -------------------------
def yolo_import_debug():
    """
    Provide helpful runtime diagnostics about the 'ultralytics' package and
    possible repo shadowing (cv2.py, local ultralytics package).
    Keep this active while debugging deployment; remove it later.
    """
    info = []
    info.append(f"sys.version: {sys.version.splitlines()[0]}")
    info.append(f"sys.executable: {sys.executable}")
    info.append(f"cwd: {os.getcwd()}")
    info.append(f"sys.path (first 6): {sys.path[:6]!r}")

    # find_spec details
    try:
        spec = importlib.util.find_spec("ultralytics")
        info.append(f"ultralytics find_spec: {spec!r}")
        if spec is not None:
            origin = getattr(spec, "origin", None)
            submodule_search_locations = getattr(spec, "submodule_search_locations", None)
            info.append(f"origin: {origin}")
            info.append(f"submodule_search_locations: {submodule_search_locations}")
    except Exception as e:
        info.append(f"ultralytics find_spec error: {repr(e)}")

    # Try to import (capture errors) and show basic module info
    try:
        mod = importlib.import_module("ultralytics")
        info.append(f"ultralytics module file: {getattr(mod, '__file__', repr(mod))}")
        info.append(f"ultralytics.YOLO present: {hasattr(mod, 'YOLO')}")
        try:
            YOLO_attr = getattr(mod, "YOLO", None)
            info.append(f"ultralytics.YOLO repr (short): {repr(YOLO_attr)[:200]}")
        except Exception:
            info.append("ultralytics.YOLO repr: <error obtaining repr>")
    except Exception as e:
        info.append(f"ultralytics import error: {repr(e)}")

    # quick package import/version checks (non-blocking)
    pkgs = []
    candidates = [
        ("ultralytics", "ultralytics"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("opencv-python", "cv2"),
        ("opencv-python-headless", "cv2"),
        ("numpy", "numpy"),
        ("streamlit", "streamlit"),
    ]
    for display, module_name in candidates:
        try:
            m = importlib.import_module(module_name)
            ver = getattr(m, "__version__", None)
            pkgs.append(f"{display}: {ver}")
        except Exception as ex:
            pkgs.append(f"{display}: import error ({repr(ex)})")
    info.append("quick package checks: " + "; ".join(pkgs))

    # Check for repo-shadowing files that commonly cause problems
    repo_root = os.getcwd()
    shadow_checks = []
    cv2_path = os.path.join(repo_root, "cv2.py")
    shadow_checks.append(f"cv2.py exists at repo root: {os.path.exists(cv2_path)} ({cv2_path})")
    ultra_dir = os.path.join(repo_root, "ultralytics")
    shadow_checks.append(f"ultralytics/ dir exists in repo root: {os.path.exists(ultra_dir)} ({ultra_dir})")
    try:
        root_listing = sorted(os.listdir(repo_root))
        info.append("Repo root listing (first 160 chars): " + ", ".join(root_listing)[:160])
    except Exception as e:
        info.append(f"Could not list repo root: {repr(e)}")

    info.append("")
    info.append("Shadow checks:")
    info.extend(shadow_checks)

    # If the main import error is libGL, show actionable message
    details = "\n".join(info)
    if "libGL.so.1" in details or "libGL" in details:
        details += "\n\nACTION: The environment is missing system OpenGL libs (libGL)."
        details += "\n- Preferred fix (no system packages needed on many hosts): ensure 'opencv-python-headless' is installed and that 'opencv-python' is NOT installed."
        details += "\n  -> Update requirements.txt to list opencv-python-headless (before ultralytics) and remove any opencv-python entries, then redeploy."
        details += "\n- Alternative (requires system package install): install system library 'libgl1' / 'libgl1-mesa-glx' (via apt) on hosts that allow apt."
        details += "\n- If you're on a platform that cannot apt (e.g., Streamlit Community Cloud), use headless OpenCV in requirements or deploy to a host that allows adding system packages (Hugging Face Spaces with apt or a Docker-based host)."

    st.sidebar.text_area("YOLO debug (temporary)", details, height=360)


# Call it once (remove after debugging)
yolo_import_debug()

# -------------------------
# Utility: robust conversion to numpy
# -------------------------
def to_numpy(x):
    """
    Convert common Ultralytics/torch-like objects to a numpy array of floats.
    Handles: torch tensors, numpy arrays, lists/iterables of numbers/lists.
    Returns None for None input.
    """
    if x is None:
        return None
    # torch tensors
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(float)
    except Exception:
        pass

    # numpy arrays
    if isinstance(x, np.ndarray):
        try:
            return x.astype(float)
        except Exception:
            return x

    # some ultralytics/tensor-like objects expose .numpy() or .cpu()
    if hasattr(x, "numpy"):
        try:
            return x.numpy().astype(float)
        except Exception:
            pass
    if hasattr(x, "cpu"):
        try:
            return x.cpu().numpy().astype(float)
        except Exception:
            pass

    # Try to coerce iterable-of-iterables to numpy explicitly
    try:
        return np.array([list(map(float, item)) for item in x], dtype=float)
    except Exception:
        # last resort: try a direct conversion to float array
        try:
            return np.array(x, dtype=float)
        except Exception:
            # give up and return None to indicate failure
            return None

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
        msg_lines.append("")
        msg_lines.append("Common fixes:")
        msg_lines.append("- Ensure `ultralytics==8.3.235` is listed in requirements.txt")
        msg_lines.append("- Ensure only `opencv-python-headless` is installed (remove any `opencv-python`)")
        msg_lines.append("- Remove any file named `cv2.py` or a local `ultralytics` package in the repo root")
        msg_lines.append("- If environment shows libGL error, either use headless opencv or install libgl system package (apt)")
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
    st.sidebar.markdown(
        "Try:\n"
        "- Ensure `ultralytics==8.3.235` is in `requirements.txt`.\n"
        "- Ensure only `opencv-python-headless` is listed (remove any `opencv-python`).\n"
        "- Delete any `cv2.py` file in repo root (it shadows real OpenCV).\n"
        "- Remove any local `ultralytics` folder in repo root that would shadow the installed package."
    )
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

    # show uploaded image (use fixed width instead of deprecated use_column_width)
    display_width = min(900, image.width)
    st.image(image, caption="Uploaded image", width=display_width)

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
                results = model_obj.predict(np.array(image))
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
                        st.image(draw_img, caption="No detections", width=min(900, draw_img.width))
                    else:
                        # Try to extract coordinates and confidences robustly
                        try:
                            # common ultralytics fields: boxes.xyxy and boxes.conf
                            coords = getattr(boxes, "xyxy", None)
                            confs = getattr(boxes, "conf", None)

                            # convert to numpy arrays when needed using helper
                            coords_arr = to_numpy(coords) if coords is not None else np.array([])
                            confs_arr = to_numpy(confs) if confs is not None else None

                            # If coords_arr is 1-D (single box flattened), reshape safely
                            if coords_arr is not None and isinstance(coords_arr, np.ndarray) and coords_arr.ndim == 1 and coords_arr.size >= 4:
                                coords_arr = coords_arr.reshape(-1, coords_arr.size)

                            # Draw rectangles
                            if coords_arr is None:
                                coords_arr = np.array([])

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

                        st.image(draw_img, caption=f"Detections ({count})", width=min(900, draw_img.width))
                        if count == 0:
                            st.success("No illegal waste detected (confidence threshold may be high).")
                        else:
                            st.error(f"Illegal waste detected in {count} region(s).")
            except Exception as e:
                st.error("Error during model prediction. See debug below.")
                st.text(repr(e))
                with st.expander("Full traceback"):
                    st.text(traceback.format_exc())

# -------------------------
# Footer notes
# -------------------------
st.markdown("---")
st.caption(
    "Note: This app lazy-loads the Ultralytics YOLO class. If you see import errors around `libGL.so.1`, "
    "you must install only headless OpenCV (opencv-python-headless) and remove opencv-python from requirements, "
    "or install system OpenGL libs (libgl1) if your host supports apt. On some managed hosts (Streamlit Community Cloud) "
    "you cannot apt; prefer opencv-python-headless or use a host that allows adding system packages (Hugging Face Spaces or Docker-based)."
)
