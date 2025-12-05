##############################################################
# STREAMLIT + YOLOv8 WASTE DETECTION (NO CV2 REQUIRED)
# FINAL FIXED VERSION (RUNS ON STREAMLIT CLOUD)
##############################################################

# ---- ENVIRONMENT PATCHES (prevent OpenCV loading) ----
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FORCE_CPU"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["DISPLAY"] = "0"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

# Prevent cv2 import in Ultralytics
import sys
sys.modules['cv2'] = None


# ---- LIBRARY IMPORTS ----
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np


##############################################################
# STREAMLIT UI HEADER
##############################################################
st.set_page_config(page_title="Illegal Waste Detection", layout="wide")

st.title("🚮 Illegal Waste Detection System (YOLOv8)")
st.write("Upload an image and the system will detect illegal waste dumping.")


##############################################################
# LOAD MODEL (cached for performance)
##############################################################
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # best.pt must be in root folder of repo

model = load_model()


##############################################################
# FILE UPLOAD SECTION
##############################################################
uploaded_file = st.file_uploader("Select Image to Analyze", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Detect Waste"):

        st.write("### Detection Result:")

        # Convert to numpy for YOLO inference
        img_array = np.array(image)

        results = model.predict(img_array, conf=0.30)
        detections = results[0]

        # Copy image for drawing boxes
        result_img = image.copy()
        draw = ImageDraw.Draw(result_img)

        # Draw bounding boxes without using cv2
        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box
            conf = float(box.conf[0])     # Confidence

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Draw confidence
            draw.text((x1, y1), f"{conf*100:.1f}%", fill="red")

        # Show final result
        st.image(result_img, caption="Detected Waste", use_column_width=True)

        # Summary
        if len(detections.boxes) == 0:
            st.success("🎉 No illegal dumping detected!")
        else:
            st.error(f"⚠️ Waste detected — {len(detections.boxes)} region(s) found.")
