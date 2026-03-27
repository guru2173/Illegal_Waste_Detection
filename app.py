##############################################################
# Streamlit app: Illegal Waste Detection (YOLO)
# Phase II: Includes Geo-Tagging and Municipal Email Alerts
##############################################################
import os
import io
import importlib
import traceback
import smtplib
from email.message import EmailMessage
from typing import Tuple, Optional

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np

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

# Add confidence slider to sidebar
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

st.sidebar.markdown("If YOLO class cannot be imported, follow instructions shown below.")

# -------------------------
# Geo-Tagging Utility Functions
# -------------------------
def get_exif_data(image):
    """Extracts EXIF data from a PIL Image."""
    exif_data = {}
    try:
        info = image._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_data = {}
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_data[sub_decoded] = value[t]
                    exif_data[decoded] = gps_data
                else:
                    exif_data[decoded] = value
    except Exception:
        pass
    return exif_data

def convert_to_degrees(value):
    """Converts GPS coordinates to decimal degrees."""
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)

def get_lat_lon(exif_data):
    """Returns latitude and longitude if available from EXIF data."""
    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]
        gps_lat = gps_info.get("GPSLatitude")
        gps_lat_ref = gps_info.get("GPSLatitudeRef")
        gps_lon = gps_info.get("GPSLongitude")
        gps_lon_ref = gps_info.get("GPSLongitudeRef")

        if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:
            lat = convert_to_degrees(gps_lat)
            if gps_lat_ref != "N":
                lat = -lat
            lon = convert_to_degrees(gps_lon)
            if gps_lon_ref != "E":
                lon = -lon
            return lat, lon
    return None, None

# -------------------------
# Alert & Notification Utility
# -------------------------
def send_municipal_alert(image, count, lat, lon):
    """Sends an email alert with the detection image and location."""
    try:
        # Retrieve credentials from Streamlit Secrets
        sender_email = st.secrets["email"]["sender_email"]
        sender_password = st.secrets["email"]["sender_password"]
        receiver_email = st.secrets["email"]["receiver_email"]

        msg = EmailMessage()
        msg['Subject'] = f"🚨 URGENT: Illegal Waste Dumping Detected ({count} region(s))"
        msg['From'] = sender_email
        msg['To'] = receiver_email

        # Construct email body
        body = f"An automated detection of illegal waste dumping has been flagged.\n\n"
        body += f"Total Detections: {count}\n"
        
        if lat is not None and lon is not None:
            body += f"Location Coordinates: {lat:.6f}, {lon:.6f}\n"
            body += f"Google Maps Link: https://www.google.com/maps?q={lat},{lon}\n\n"
        else:
            body += "Location: GPS data not available for this image.\n\n"
            
        body += "Please find the processed image attached for your review."
        msg.set_content(body)

        # Convert PIL image to bytes for attachment
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        msg.add_attachment(img_byte_arr, maintype='image', subtype='jpeg', filename='detection_alert.jpg')

        # Connect to Gmail SMTP server and send
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
            
        return True, "Alert sent successfully to Municipal Authorities."
    except Exception as e:
        return False, f"Failed to send alert. Error: {str(e)}"

# -------------------------
# Attempt to import YOLO (lazy)
# -------------------------
def try_import_yolo():
    errors = []
    try:
        ultralytics = importlib.import_module("ultralytics")
        YOLO_cls = getattr(ultralytics, "YOLO", None)
        if YOLO_cls is not None:
            return YOLO_cls, errors
        errors.append(("ultralytics module", "YOLO attribute missing"))
    except Exception as e:
        errors.append(("from ultralytics import YOLO", repr(e)))

    try:
        mod = importlib.import_module("ultralytics.yolo.engine.model")
        YOLO_cls = getattr(mod, "YOLO", None)
        if YOLO_cls is not None:
            return YOLO_cls, errors
        errors.append(("ultralytics.yolo.engine.model", "YOLO attribute missing"))
    except Exception as e:
        errors.append(("import ultralytics.yolo.engine.model", repr(e)))

    return None, errors

# -------------------------
# Lazy model loader (cached)
# -------------------------
@st.cache_resource
def load_model(path: str) -> Tuple[Optional[object], Optional[str]]:
    YOLO_cls, import_errors = try_import_yolo()
    if YOLO_cls is None:
        msg_lines = ["Unable to import YOLO class. Import attempts:"]
        for name, err in import_errors:
            msg_lines.append(f"- {name}: {err}")
        return None, "\n".join(msg_lines)

    if not os.path.exists(path):
        return None, f"Model file '{path}' not found in repo root."

    try:
        model = YOLO_cls(path)  
        return model, None
    except Exception as e:
        return None, f"Failed to load model '{path}': {repr(e)}"

# -------------------------
# Sidebar Model Loader UI
# -------------------------
st.sidebar.markdown("## Model Loader")
if st.sidebar.button("Load Model Now"):
    model_obj, model_err = load_model(model_filename)
    if model_obj is None:
        st.sidebar.error("Model NOT loaded.")
    else:
        st.sidebar.success("Model loaded successfully.")

# -------------------------
# Main UI: image upload + detection
# -------------------------
uploaded_file = st.file_uploader("Upload image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.info("Upload an image to run detection (model must be loaded first).")
else:
    try:
        # Load image for detection
        image = Image.open(uploaded_file).convert("RGB")
        
        # Reload image strictly for EXIF data (converting to RGB sometimes strips it)
        raw_image = Image.open(uploaded_file)
        exif_data = get_exif_data(raw_image)
        lat, lon = get_lat_lon(exif_data)
        
    except Exception as e:
        st.error("Could not open uploaded file as image.")
        st.stop()

    st.image(image, caption="Uploaded image", width=min(700, image.width))

    # Run detection button
    if st.button("🔍 Run Detection"):
        model_obj, model_err = load_model(model_filename)
        if model_obj is None:
            st.error("Cannot run detection: model not loaded.")
            st.text(model_err)
        else:
            with st.spinner("Analyzing image..."):
                try:
                    results = model_obj.predict(np.array(image), conf=conf_threshold)
                    
                    if results is None or len(results) == 0:
                        st.warning("Model returned no results object.")
                    else:
                        res0 = results[0]
                        boxes = getattr(res0, "boxes", None)
                        draw_img = image.copy()
                        draw = ImageDraw.Draw(draw_img)
                        count = 0

                        if boxes is not None:
                            try:
                                coords = getattr(boxes, "xyxy", None)
                                confs = getattr(boxes, "conf", None)
                                
                                coords_arr = np.array(coords) if coords is not None else np.array([])
                                # Handle torch tensor reshape issue safely
                                if coords_arr.ndim == 1 and coords_arr.size >= 4:
                                    coords_arr = coords_arr.reshape(-1, 4)
                                    
                                confs_arr = np.array(confs) if confs is not None else np.array([])

                                for i, xy in enumerate(coords_arr):
                                    if len(xy) < 4: continue
                                    x1, y1, x2, y2 = map(float, xy[:4])
                                    conf = float(confs_arr[i]) if i < len(confs_arr) else 0.0
                                    
                                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                                    txt = f"Waste {conf*100:.1f}%"
                                    draw.text((x1 + 4, y1 + 4), txt, fill="red")
                                    count += 1

                            except Exception as inner_e:
                                st.error(f"Error drawing boxes: {inner_e}")

                        st.image(draw_img, caption=f"Detections ({count})", width=min(700, draw_img.width))
                        
                        if count == 0:
                            st.success(f"✅ No illegal waste detected at {conf_threshold*100:.0f}% confidence.")
                        else:
                            st.error(f"⚠️ Illegal waste detected in {count} region(s)!")
                            
                            # ---- GEO-TAGGING MODULE UI ----
                            st.markdown("### 📍 Location Data")
                            if lat is not None and lon is not None:
                                st.success(f"GPS Coordinates Found: {lat:.5f}, {lon:.5f}")
                                map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
                                st.map(map_data, zoom=15)
                            else:
                                st.warning("No GPS data found in this image's EXIF metadata. (Ensure location services are on when taking the photo).")
                            
                            # ---- MUNICIPAL RESPONSE MODULE UI ----
                            st.markdown("### 🚨 Municipal Response Network")
                            if st.button("📧 Dispatch Municipal Team"):
                                with st.spinner("Connecting to secure server and dispatching email..."):
                                    # Ensure secrets exist before trying to send
                                    if "email" in st.secrets:
                                        success, msg = send_municipal_alert(draw_img, count, lat, lon)
                                        if success:
                                            st.success(msg)
                                            st.balloons()
                                        else:
                                            st.error(msg)
                                    else:
                                        st.error("Email configuration missing. Please add 'email' credentials to App Settings -> Secrets.")

                except Exception as e:
                    st.error(f"Error during model prediction: {e}")

st.markdown("---")
st.caption("Capstone Phase II: Smart Illegal Waste Dumping Detection and Municipal Response Network")
