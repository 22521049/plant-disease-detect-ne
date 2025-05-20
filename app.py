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
    "Pepper,_bell___healthy": {
        "name": "Bell Pepper Healthy",
        "desc": "Lá ớt chuông khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ để phát hiện bệnh sớm."
    },
    "Pepper,_bell___Bacterial_spot": {
        "name": "Bell Pepper Bacterial Spot",
        "desc": "Bệnh đốm vi khuẩn trên ớt chuông, gây ra các đốm nâu, lõm trên lá và quả, có thể làm rụng lá.",
        "treatment": "Loại bỏ lá bệnh, phun thuốc gốc đồng, luân canh cây trồng, sử dụng giống kháng bệnh."
    },
    "Blueberry___healthy": {
        "name": "Blueberry Healthy",
        "desc": "Lá việt quất khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, tưới nước và bón phân hợp lý."
    },
    "Cherry_(including_sour)___healthy": {
        "name": "Cherry Healthy",
        "desc": "Lá anh đào khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ."
    },
    "Corn_(maize)___Common_rust_": {
        "name": "Corn Common Rust",
        "desc": "Bệnh gỉ sắt thông thường trên ngô, xuất hiện các đốm màu nâu đỏ trên lá, làm giảm năng suất.",
        "treatment": "Sử dụng giống kháng bệnh, phun thuốc trừ nấm (như mancozeb, azoxystrobin) khi phát hiện bệnh."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "name": "Corn Northern Leaf Blight",
        "desc": "Bệnh cháy lá phương Bắc trên ngô, gây ra các vết dài màu xám trên lá, làm giảm khả năng quang hợp.",
        "treatment": "Trồng giống kháng bệnh, luân canh cây trồng, phun thuốc trừ nấm khi cần thiết."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "name": "Corn Gray Leaf Spot",
        "desc": "Bệnh đốm xám lá ngô do nấm Cercospora gây ra, xuất hiện các vết xám hình chữ nhật trên lá.",
        "treatment": "Sử dụng giống kháng bệnh, phun thuốc trừ nấm, vệ sinh tàn dư thực vật."
    },
    "Peach___healthy": {
        "name": "Peach Healthy",
        "desc": "Lá đào khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ."
    },
    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "desc": "Bệnh sớm trên khoai tây do nấm Alternaria solani, gây ra các đốm tròn nâu trên lá, có quầng vàng.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc trừ nấm (như mancozeb, chlorothalonil), luân canh cây trồng."
    },
    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "desc": "Bệnh mốc sương trên khoai tây do nấm Phytophthora infestans, gây thối lá, thân và củ.",
        "treatment": "Phun thuốc trừ nấm (như metalaxyl, cymoxanil), tiêu hủy cây bệnh, trồng giống kháng bệnh."
    },
    "Raspberry___healthy": {
        "name": "Raspberry Healthy",
        "desc": "Lá mâm xôi khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ."
    },
    "Soybean___healthy": {
        "name": "Soybean Healthy",
        "desc": "Lá đậu tương khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ."
    },
    "Squash___Powdery_mildew": {
        "name": "Squash Powdery Mildew",
        "desc": "Bệnh phấn trắng trên bí, xuất hiện lớp phấn trắng trên bề mặt lá, làm lá vàng và khô.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc trừ nấm (như sulfur, myclobutanil), tăng thông thoáng cho cây."
    },
    "Strawberry___healthy": {
        "name": "Strawberry Healthy",
        "desc": "Lá dâu tây khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ."
    },
    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "desc": "Bệnh sớm trên cà chua do nấm Alternaria solani, gây đốm nâu tròn trên lá, thân và quả.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc trừ nấm (như mancozeb, chlorothalonil), luân canh cây trồng."
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "desc": "Bệnh đốm lá Septoria trên cà chua, xuất hiện các đốm nhỏ màu nâu xám trên lá.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc trừ nấm (như mancozeb, copper), vệ sinh vườn."
    },
    "Tomato___healthy": {
        "name": "Tomato Healthy",
        "desc": "Lá cà chua khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ."
    },
    "Tomato___Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "desc": "Bệnh đốm vi khuẩn trên cà chua, gây đốm nhỏ màu nâu đen trên lá, thân và quả.",
        "treatment": "Loại bỏ lá bệnh, phun thuốc gốc đồng, sử dụng giống kháng bệnh."
    },
    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "desc": "Bệnh mốc sương trên cà chua do nấm Phytophthora infestans, gây thối lá, thân và quả.",
        "treatment": "Phun thuốc trừ nấm (như metalaxyl, cymoxanil), tiêu hủy cây bệnh, trồng giống kháng bệnh."
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "desc": "Bệnh virus khảm trên cà chua, gây lá biến dạng, vàng, giảm năng suất.",
        "treatment": "Tiêu hủy cây bệnh, vệ sinh dụng cụ, sử dụng giống kháng virus."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "desc": "Bệnh virus vàng xoăn lá trên cà chua, làm lá xoăn, vàng, cây còi cọc.",
        "treatment": "Tiêu hủy cây bệnh, kiểm soát bọ phấn, sử dụng giống kháng virus."
    },
    "Tomato___Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "desc": "Bệnh mốc lá trên cà chua, xuất hiện các mảng vàng ở mặt trên lá, mặt dưới có lớp mốc màu ô liu.",
        "treatment": "Cắt bỏ lá bệnh, tăng thông thoáng, phun thuốc trừ nấm (như mancozeb, copper)."
    },
    "Grape___healthy": {
        "name": "Grape Healthy",
        "desc": "Lá nho khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ."
    },
    "Grape___Black_rot": {
        "name": "Grape Black Rot",
        "desc": "Bệnh thối đen trên nho do nấm Guignardia bidwellii, gây đốm đen trên lá, quả bị thối đen.",
        "treatment": "Cắt bỏ lá và quả bệnh, phun thuốc trừ nấm (như mancozeb, myclobutanil), vệ sinh vườn."
    }
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

uploaded_file = st.file_uploader("Chọn ảnh lá cây", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Ảnh lá cây", use_container_width=True)
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

st.caption("Triển khai bởi Google Cloud Run, model và dữ liệu lưu trên Google Cloud Storage.")



