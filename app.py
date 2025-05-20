import os
import streamlit as st
from google.cloud import storage
import tensorflow as tf
import json
import numpy as np
import cv2

# ========== THÔNG TIN BỆNH ==========
disease_info = {
    "Apple___Apple_scab": {
        "name": "Apple Scab",
        "desc": "Bệnh đốm vảy trên táo do nấm Venturia inaequalis gây ra, xuất hiện các đốm nâu/xám trên lá và quả, làm giảm năng suất và chất lượng quả.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc gốc đồng hoặc thuốc chứa mancozeb, vệ sinh vườn cây."
    },
    "Apple___Cedar_apple_rust": {
        "name": "Apple Rust",
        "desc": "Bệnh gỉ sắt trên táo, gây ra các đốm màu cam trên lá, có thể làm rụng lá sớm và ảnh hưởng đến sự phát triển của cây.",
        "treatment": "Loại bỏ lá bệnh, dùng thuốc diệt nấm (như mancozeb, myclobutanil), trồng giống kháng bệnh."
    },
    "Apple___healthy": {
        "name": "Apple Healthy",
        "desc": "Lá táo khỏe mạnh, không có dấu hiệu bệnh lý.",
        "treatment": "Không có bệnh, tiếp tục chăm sóc tốt, bón phân hợp lý và tưới nước đầy đủ."
    },
    # ... (bổ sung các class khác như bạn đã làm)
}

# ========== TẢI MODEL VÀ FILE JSON TỪ GCS ==========
BUCKET_NAME = "plant-disease-model-bucket"
MODEL_BLOB = "plant_disease_model.h5"
CLASS_INDICES_BLOB = "class_indices.json"
MODEL_LOCAL = "plant_disease_model.h5"
CLASS_INDICES_LOCAL = "class_indices.json"
IMG_SIZE = 224

def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}")

@st.cache_resource
def load_model_and_indices():
    if not os.path.exists(MODEL_LOCAL):
        download_blob(BUCKET_NAME, MODEL_BLOB, MODEL_LOCAL)
    if not os.path.exists(CLASS_INDICES_LOCAL):
        download_blob(BUCKET_NAME, CLASS_INDICES_BLOB, CLASS_INDICES_LOCAL)
    model = tf.keras.models.load_model(MODEL_LOCAL)
    with open(CLASS_INDICES_LOCAL, "r") as f:
        class_indices = json.load(f)
    class_indices = {int(k): v for k, v in class_indices.items()}
    return model, class_indices

try:
    model, class_indices_dict = load_model_and_indices()
except Exception as e:
    st.error(f"Lỗi khi tải model hoặc file json: {e}")
    st.stop()

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

# ========== GIAO DIỆN STREAMLIT ==========

st.set_page_config(page_title="Nhận diện bệnh lá cây", page_icon=":herb:", layout="centered")
st.markdown("<h1 style='text-align: center; color: green;'>🌿 Nhận diện bệnh lá cây</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📁 Tải ảnh lên", "📷 Chụp ảnh webcam"])

with tab1:
    uploaded_file = st.file_uploader("Chọn ảnh lá cây", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="Ảnh lá cây", use_column_width=True)
        if st.button("Phát hiện bệnh"):
            with st.spinner("Đang dự đoán..."):
                class_name, confidence = predict_disease(image, model, class_indices_dict)
            info = disease_info.get(class_name)
            if info:
                st.markdown(
                    f"""
                    <div style="background-color:#e6fff2;padding:16px;border-radius:8px;">
                        <h3 style="color:#d7263d;">{info['name']}</h3>
                        <b>Mô tả:</b> {info['desc']}<br>
                        <b>Cách chữa trị:</b> {info['treatment']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Chưa có thông tin chi tiết về bệnh này.")
            st.caption(f"Độ tin cậy: {confidence:.2f}%")
    else:
        st.info("Vui lòng upload ảnh lá cây để nhận diện.")

with tab2:
    try:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
        import av

        class LeafDiseaseTransformer(VideoTransformerBase):
            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                return img_resized

        ctx = webrtc_streamer(
            key="webcam",
            video_transformer_factory=LeafDiseaseTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
        )

        if ctx.video_transformer and ctx.video_transformer.frame is not None:
            st.image(ctx.video_transformer.frame, caption="Ảnh webcam", use_column_width=True)
            if st.button("Phát hiện bệnh từ webcam"):
                with st.spinner("Đang dự đoán..."):
                    class_name, confidence = predict_disease(ctx.video_transformer.frame, model, class_indices_dict)
                info = disease_info.get(class_name)
                if info:
                    st.markdown(
                        f"""
                        <div style="background-color:#e6fff2;padding:16px;border-radius:8px;">
                            <h3 style="color:#d7263d;">{info['name']}</h3>
                            <b>Mô tả:</b> {info['desc']}<br>
                            <b>Cách chữa trị:</b> {info['treatment']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Chưa có thông tin chi tiết về bệnh này.")
                st.caption(f"Độ tin cậy: {confidence:.2f}%")
        else:
            st.info("Bật webcam để chụp ảnh lá cây.")
    except Exception as e:
        st.warning("Webcam không khả dụng hoặc thiếu streamlit-webrtc.")

st.caption("Triển khai bởi Google Cloud Run, model và dữ liệu lưu trên Google Cloud Storage.")

