# Stroke Prediction

Dự án dự đoán khả năng đột quỵ của bệnh nhân sử dụng Machine Learning.

## Mục tiêu
Dự án xây dựng mô hình dự đoán khả năng bị đột quỵ dựa trên các đặc trưng như:  
- Giới tính  
- Tuổi  
- Tiền sử tăng huyết áp và bệnh tim  
- Tình trạng kết hôn  
- Loại công việc  
- Nơi cư trú  
- Mức đường huyết trung bình  
- Chỉ số BMI  
- Tình trạng hút thuốc  

## Cấu trúc dự án
- **data/**: Chứa bộ dữ liệu gốc.
- **notebooks/**: Notebook phân tích dữ liệu (EDA) và xây dựng/hướng dẫn huấn luyện mô hình.
- **models/**: Lưu trữ mô hình đã huấn luyện (stroke_model.pkl) và bộ scaler.
- **app.py**: Giao diện người dùng bằng Streamlit để nhập dữ liệu và dự đoán.
- **.gitignore**: Loại trừ các file không cần thiết.
- **requirements.txt**: Các thư viện cần thiết.

## Hướng dẫn sử dụng

1. **Cài đặt thư viện:**

   ```bash
   pip install -r requirements.txt
