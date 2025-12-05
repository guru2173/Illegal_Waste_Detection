##############################################################
# ILLEGAL WASTE DETECTION STREAMLIT APP (FINAL DEPLOYABLE)
##############################################################

# ---- ENVIRONMENT FIXES (prevents GPU + libGL crash) ----
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FORCE_CPU"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["DISPLAY"] = "0"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

# Patch system to avoid OpenCV import
import sys
sys.modules['cv2'] = None

# ---- LIBRARY IMPORTS ----
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

##############################################################
# STREAMLIT WEB UI
##############################################################
st.set_page_config(page_title="Illegal Waste Detection", layout="wide")

st.title("🚮 Illegal Waste Detection using YOLOv8")
st.write("Upload an image and the system will detect illegal dumping regions.")

##############################################################
# LOAD YOLO MODEL
##############################################################
@st.cache_resource
def load_model():
    return YOLO("best.pt")     # must exist in repo root

model = load_model()

##############################################################
# IMAGE UPLOAD + DETECTION
##############################################################
uploaded_file = st.file_uploader(
    "Upload an Image for Detection",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Detect Illegal Waste"):
        st.subheader("Detection Results:")

        # YOLO prediction
        results = model.predict(np.array(image))
        detections = results[0]

        # Duplicate the image to draw bounding boxes
        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)

        # Draw bounding boxes
        count = 0
        for box in detections.boxes:
            count += 1
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"{conf*100:.1f}%", fill="red")

        # Show output
        st.image(draw_img, caption="Detected Output", use_column_width=True)

        # Summary text
        if count == 0:
            st.success("🎉 No illegal waste found (Clean Site).")
        else:
            st.error(f"⚠️ Illegal waste detected in {count} region(s).")
