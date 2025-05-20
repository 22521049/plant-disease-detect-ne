import os
import streamlit as st
from google.cloud import storage
import tensorflow as tf
import json
import numpy as np
import cv2

# =========================
# 1. TẢI MODEL VÀ FILE JSON TỪ GCS BUCKET
# =========================

BUCKET_NAME = "plant-disease-model-bucket"
MODEL_BLOB = "plant_disease_model.h5"
CLASS_INDICES_BLOB = "class_indices.json"
MODEL_LOCAL = "plant_disease_model.h5"
CLASS_INDICES_LOCAL = "class_indices.json"

def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}")

# Tải model và file json nếu chưa có
if not os.path.exists(MODEL_LOCAL):
    download_blob(BUCKET_NAME, MODEL_BLOB, MODEL_LOCAL)
if not os.path.exists(CLASS_INDICES_LOCAL):
    download_blob(BUCKET_NAME, CLASS_INDICES_BLOB, CLASS_INDICES_LOCAL)

# =========================
# 2. LOAD MODEL VÀ CLASS INDICES
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_LOCAL)

@st.cache_data
def load_class_indices():
    with open(CLASS_INDICES_LOCAL, "r") as f:
        class_indices = json.load(f)
    # Đảm bảo key là int
    return {int(k): v for k, v in class_indices.items()}

model = load_model()
class_indices_dict = load_class_indices()

IMG_SIZE = 224

# =========================
# 3. HÀM DỰ ĐOÁN
# =========================

def predict_disease(image, model, class_indices):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    pred_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    class_name = class_indices[pred_class]
    return class_name, confidence

# =========================
# 4. GIAO DIỆN STREAMLIT
# =========================

st.title("Nhận diện bệnh lá cây bằng Deep Learning (Google Cloud Run + GCS)")
st.write("Upload ảnh lá cây để dự đoán loại bệnh.")

uploaded_file = st.file_uploader("Chọn ảnh lá cây", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh từ file uploader
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Ảnh bạn upload", use_column_width=True)
    with st.spinner("Đang dự đoán..."):
        class_name, confidence = predict_disease(image, model, class_indices_dict)
    st.success(f"Kết quả: **{class_name}** (Độ tin cậy: {confidence:.2f}%)")

    # Hiển thị thêm thông tin nếu cần
else:
    st.info("Vui lòng upload ảnh lá cây để nhận diện.")

st.caption("Triển khai bởi Google Cloud Run, model và dữ liệu lưu trên Google Cloud Storage.")

