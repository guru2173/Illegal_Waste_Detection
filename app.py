# app.py
import os
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import importlib
import traceback

# optional: hf download
from huggingface_hub import hf_hub_download, HfHubHTTPError

st.set_page_config(page_title="Illegal Waste Detection", layout="wide")
st.title("🚮 Illegal Waste Detection (YOLO)")

st.write("Upload an image to detect illegal dumping regions. Model loading is lazy — see sidebar for status.")

# Sidebar settings
st.sidebar.header("Model settings")
model_filename = st.sidebar.text_input("Model filename in repo root", "best.pt")
hf_model_repo = st.sidebar.text_input("Hugging Face model repo (optional)", "")  # e.g., username/illegal-waste-model
st.sidebar.markdown("If the model file is not in repo, provide HF model repo above (or upload best.pt to repo).")

st.sidebar.markdown("## Runtime options")
device = st.sidebar.selectbox("Device", ["cpu", "cuda"] if os.environ.get("CUDA_VISIBLE_DEVICES") else ["cpu"])

# Helper to import YOLO class safely
def try_import_yolo():
    errors = []
    YOLO = None
    try:
        mod = importlib.import_module("ultralytics")
        YOLO = getattr(mod, "YOLO", None)
        if YOLO is not None:
            return YOLO, errors
        errors.append(("ultralytics module; attribute YOLO missing", "ultralytics loaded but YOLO attribute not present"))
    except Exception as e:
        errors.append(("from ultralytics import YOLO", repr(e)))

    try:
        mod2 = importlib.import_module("ultralytics.yolo.engine.model")
        YOLO = getattr(mod2, "YOLO", None)
        if YOLO is not None:
            return YOLO, errors
        errors.append(("ultralytics.yolo.engine.model; YOLO attr missing", "module loaded but YOLO attribute missing"))
    except Exception as e:
        errors.append(("import ultralytics.yolo.engine.model", repr(e)))

    try:
        mod3 = importlib.import_module("ultralytics.yolo")
        if hasattr(mod3, "YOLO"):
            return getattr(mod3, "YOLO"), errors
        errors.append(("import ultralytics.yolo", "loaded but YOLO not found"))
    except Exception as e:
        errors.append(("import ultralytics.yolo", repr(e)))

    return None, errors

# cached model loader
@st.cache_resource
def load_model_local_or_hf(local_path, hf_repo=None, hf_token=None, device="cpu"):
    YOLO_class, import_errors = try_import_yolo()
    if YOLO_class is None:
        msg = "Unable to import YOLO class. Import attempts:\n"
        for name, err in import_errors:
            msg += f"- {name}: {err}\n"
        return None, msg

    # if local file exists, use it
    if os.path.exists(local_path):
        try:
            model = YOLO_class(local_path)
            # set device if supported
            try:
                model.to(device)
            except Exception:
                pass
            return model, None
        except Exception as e:
            return None, f"Failed to load local model '{local_path}': {repr(e)}\n" + traceback.format_exc()

    # else if hf_repo specified, try downloading
    if hf_repo:
        try:
            # download file; will throw if not available
            downloaded = hf_hub_download(repo_id=hf_repo, filename=local_path, use_auth_token=hf_token)
            try:
                model = YOLO_class(downloaded)
                try:
                    model.to(device)
                except Exception:
                    pass
                return model, None
            except Exception as e:
                return None, f"Failed to load model after download from HF repo: {repr(e)}\n" + traceback.format_exc()
        except HfHubHTTPError as e:
            return None, f"HuggingFace Hub download error: {repr(e)}"
        except Exception as e:
            return None, f"Unexpected error when downloading from HF hub: {repr(e)}\n{traceback.format_exc()}"

    return None, f"Model file '{local_path}' not found in repo and no HF repo provided."

# Sidebar: load model button
st.sidebar.markdown("## Model loader")
if st.sidebar.button("Load model now"):
    hf_token = os.environ.get("HF_TOKEN", None)
    model_obj, model_err = load_model_local_or_hf(model_filename, hf_model_repo or None, hf_token, device=device)
    if model_obj is None:
        st.sidebar.error("Model NOT loaded.")
        st.sidebar.text_area("Load errors (copy for debugging)", value=str(model_err), height=240)
    else:
        st.sidebar.success("Model loaded successfully. Use the Run Detection button on main page.")

# Show quick check for YOLO presence
yo, yo_errs = try_import_yolo()
if yo is None:
    st.sidebar.error("YOLO class not available in this environment.")
    st.sidebar.markdown("Try these fixes:")
    st.sidebar.markdown("- Ensure `ultralytics` is in `requirements.txt`.\n- Use `opencv-python-headless`.\n- If you previously created `cv2.py` in repo root delete it.")
    st.sidebar.text_area("YOLO import attempts (debug)", "\n".join([f"{n} : {e}" for n, e in yo_errs]), height=240)
else:
    st.sidebar.success("YOLO class detected in environment.")

# Main UI
uploaded_file = st.file_uploader("Upload image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.info("Upload an image to run detection (model must be loaded first).")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("🔍 Run Detection (lazy load model if not loaded)"):
        hf_token = os.environ.get("HF_TOKEN", None)
        model_obj, model_err = load_model_local_or_hf(model_filename, hf_model_repo or None, hf_token, device=device)
        if model_obj is None:
            st.error("Cannot run detection: model not loaded.")
            st.text(model_err)
        else:
            try:
                # run prediction (you can tune imgsz, conf, etc)
                results = model_obj.predict(np.array(image), imgsz=640, conf=0.25)
                det = results[0]
                draw_img = image.copy()
                draw = ImageDraw.Draw(draw_img)
                font = None
                try:
                    font = ImageFont.load_default()
                except Exception:
                    pass

                count = 0
                # Ultralytics' boxes accessor
                for box in det.boxes:
                    count += 1
                    xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], "cpu") else np.array(box.xyxy[0])
                    x1, y1, x2, y2 = map(float, xyxy)
                    conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                    label = getattr(box, "cls", None)
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    text = f"{conf*100:.1f}%"
                    draw.text((x1, y1-10), text, fill="red", font=font)

                st.image(draw_img, caption=f"Detections ({count})", use_column_width=True)
                if count == 0:
                    st.success("No illegal waste detected.")
                else:
                    st.error(f"Illegal waste detected in {count} region(s).")

            except Exception as e:
                st.error("Error during detection. See debug below.")
                st.text(str(e))
                with st.expander("Full traceback"):
                    st.text(traceback.format_exc())
