import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

st.set_page_config(page_title="Illegal Waste Detection")

st.title("🚮 Illegal Waste Detection System")
st.write("Upload an image to detect dumping.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Detect Waste"):
        st.write("Detection Result:")
        results = model.predict(np.array(image))

        img = image.copy()
        draw = ImageDraw.Draw(img)

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"{conf:.2f}", fill="red")

        st.image(img, caption="Detection Output", use_column_width=True)
