FROM python:3.10

# Cài đặt các package hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy requirements.txt trước để tận dụng cache Docker layer
COPY requirements.txt ./

# Cài đặt các thư viện Python cần thiết
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào image
COPY . .

# Expose port 8080 cho Cloud Run
EXPOSE 8080

# Chạy Streamlit app trên port 8080 và địa chỉ 0.0.0.0
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
