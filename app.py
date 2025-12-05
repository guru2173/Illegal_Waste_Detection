import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import time

st.set_page_config(page_title="Illegal Waste Detection")

st.title("🚮 Illegal Waste Detection System (YOLOv8)")
st.write("Upload an image to detect waste dumping areas.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")  # model file must be in same folder

model = load_model()

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Detect Waste"):
        with st.spinner("Detecting..."):
            img_array = np.array(image)
            results = model.predict(source=img_array, conf=0.30)
            result = results[0]

            annotated_img = result.plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            st.image(annotated_img, caption="Detection Result", use_column_width=True)

            st.subheader("Detection Details")
            if len(result.boxes) == 0:
                st.success("🎉 No waste detected!")
            else:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    st.write(f"Confidence: {conf:.2f}")
