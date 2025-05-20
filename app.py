import streamlit as st
import numpy as np
import tensorflow as tf
import json
from PIL import Image

# ========== CẤU HÌNH ==========
IMG_SIZE = 224
MODEL_PATH = "/content/plant_disease_model (5).h5"
CLASS_INDICES_PATH = "/content/class_indices (3).json"

# ========== LOAD MODEL VÀ CLASS ==========
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_class_indices(json_path):
    with open(json_path, 'r') as f:
        class_indices_dict = json.load(f)
    class_indices_dict = {int(k): v for k, v in class_indices_dict.items()}
    return class_indices_dict

# ========== TIỀN XỬ LÝ ẢNH ==========
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = np.array(image)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy()
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ========== DỰ ĐOÁN ==========
def predict(image, model, class_indices_dict):
    img = preprocess_image(image)
    prediction = model.predict(img, verbose=0)
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    class_name = class_indices_dict[pred_class]
    return class_name, confidence

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

# ========== GIAO DIỆN STREAMLIT ==========
st.set_page_config(page_title="Nhận diện bệnh lá cây", layout="centered")
st.markdown(
    "<h2 style='text-align: center; color: #228B22; margin-bottom: 0.5em;'>🌿 Nhận diện bệnh lá cây</h2>",
    unsafe_allow_html=True
)

try:
    model = load_model(MODEL_PATH)
    class_indices_dict = load_class_indices(CLASS_INDICES_PATH)
    st.success("Đã tải mô hình thành công!")
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}")
    st.stop()

tab1, tab2 = st.tabs(["Tải ảnh lên", "Chụp ảnh webcam"])
image = None
with tab1:
    uploaded_file = st.file_uploader("Chọn ảnh lá cây", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh lá cây", width=250)
        except Exception as e:
            st.error(f"Không thể mở file ảnh: {e}")
            
with tab2:
    try:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
        class VideoTransformer(VideoTransformerBase):
            def transform(self, frame):
                return frame
        ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
        if ctx and ctx.video_transformer:
            st.info("Chụp màn hình webcam rồi tải lên!")
    except ImportError:
        st.error("Không thể tải streamlit_webrtc. Vui lòng cài đặt: pip install streamlit-webrtc")

if image and st.button("Phát hiện bệnh"):
    with st.spinner("Đang phân tích..."):
        class_name, confidence = predict(image, model, class_indices_dict)
        
        # Hiển thị kết quả
        info = disease_info.get(class_name, {"name": class_name, "desc": "Chưa có thông tin", "treatment": "Chưa có thông tin"})
        
        st.markdown(
            f"<div style='background-color:#f0fff0; padding:1em; border-radius:10px;'>"
            f"<h3 style='color:#d9534f; margin-bottom:0.5em;'>{info['name']}</h3>"
            f"<p><b>Độ tin cậy:</b> {confidence:.2f}%</p>"
            f"<p><b>Mô tả:</b> {info['desc']}</p>"
            f"<p><b>Cách chữa trị:</b> {info['treatment']}</p>"
            f"</div>", unsafe_allow_html=True
        )
else:
    if not image:
        st.info("Hãy tải lên ảnh lá cây để phát hiện bệnh.")

# Ẩn warning về use_column_width
st.markdown(
    """
    <style>
    .stImage > img {max-width: 250px !important;}
    .css-1v0mbdj {margin-bottom: 0px;}
    </style>
    """,
    unsafe_allow_html=True
)
