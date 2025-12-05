##############################################################
# FIX: Patch cv2 BEFORE importing ultralytics
##############################################################
import sys
import types

cv2_fake = types.SimpleNamespace()

# Required attributes & methods
cv2_fake.IMREAD_COLOR = 1
cv2_fake.setNumThreads = lambda *a, **k: None
cv2_fake.getNumThreads = lambda: 1
cv2_fake.imread = lambda *a, **k: None
cv2_fake.imwrite = lambda *a, **k: None
cv2_fake.imshow = lambda *a, **k: None
cv2_fake.waitKey = lambda *a, **k: None

sys.modules['cv2'] = cv2_fake


##############################################################
# ENVIRONMENT FIXES
##############################################################
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FORCE_CPU"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["DISPLAY"] = "0"

##############################################################
# IMPORTS
##############################################################
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np



##############################################################
# STREAMLIT UI
##############################################################
st.set_page_config(page_title="Illegal Waste Detection", layout="wide")
st.title("🚮 Illegal Waste Detection using YOLOv8")
st.write("Upload an image and detect dumping areas.")


##############################################################
# LOAD MODEL
##############################################################
@st.cache_resource
def load_model():
    return YOLO("best.pt")    # best.pt must be in repo root

model = load_model()


##############################################################
# UPLOAD + DETECT
##############################################################
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Detect Illegal Waste"):
        result = model.predict(np.array(img))
        detections = result[0]

        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)

        count = 0
        for box in detections.boxes:
            count += 1
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"{conf*100:.1f}%", fill="red")

        st.image(draw_img, caption="Detections", use_column_width=True)

        st.write(f"🔴 Illegal waste detected = **{count} region(s)**")
