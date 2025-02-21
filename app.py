import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Tải mô hình và bộ scaler đã huấn luyện (đảm bảo các file này nằm trong thư mục models)
model = joblib.load("models/stroke_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Dự đoán khả năng đột quỵ của bệnh nhân")
st.write("Nhập thông tin bệnh nhân dưới đây:")

# Lấy dữ liệu từ người dùng với các lựa chọn hiển thị bằng tiếng Việt
gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
age = st.number_input("Tuổi", min_value=0, max_value=120, value=50)
hypertension = st.selectbox("Tiền sử tăng huyết áp", ["Không", "Có"])
heart_disease = st.selectbox("Tiền sử bệnh tim", ["Không", "Có"])
ever_married = st.selectbox("Đã từng kết hôn?", ["Có", "Không"])
work_type = st.selectbox("Loại công việc", ["Tư nhân", "Tự kinh doanh", "Công chức", "Trẻ em", "Không làm việc"])
Residence_type = st.selectbox("Nơi cư trú", ["Đô thị", "Nông thôn"])
avg_glucose_level = st.number_input("Mức đường huyết trung bình", min_value=0.0, value=100.0)
bmi = st.number_input("Chỉ số BMI", min_value=0.0, value=25.0)
smoking_status = st.selectbox("Tình trạng hút thuốc", ["Đã hút thuốc", "Chưa từng hút thuốc", "Đang hút thuốc", "Không rõ"])

if st.button("Dự đoán"):
    # Chuyển đổi lựa chọn "Có"/"Không" thành giá trị số
    hypertension_val = 1 if hypertension == "Có" else 0
    heart_disease_val = 1 if heart_disease == "Có" else 0

    # Tạo DataFrame từ dữ liệu người dùng nhập
    input_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension_val],
        "heart_disease": [heart_disease_val],
        "ever_married": [ever_married],
        "work_type": [work_type],
        "Residence_type": [Residence_type],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [smoking_status]
    })

    # Tiền xử lý dữ liệu: mã hóa các biến phân loại theo mapping đã sử dụng trong quá trình huấn luyện
    gender_map = {"Nam": 1, "Nữ": 0}
    ever_married_map = {"Có": 1, "Không": 0}
    work_type_map = {"Tư nhân": 2, "Tự kinh doanh": 3, "Công chức": 1, "Trẻ em": 0, "Không làm việc": 4}
    Residence_type_map = {"Đô thị": 1, "Nông thôn": 0}
    smoking_status_map = {"Đã hút thuốc": 1, "Chưa từng hút thuốc": 2, "Đang hút thuốc": 0, "Không rõ": 3}

    input_data["gender"] = input_data["gender"].map(gender_map)
    input_data["ever_married"] = input_data["ever_married"].map(ever_married_map)
    input_data["work_type"] = input_data["work_type"].map(work_type_map)
    input_data["Residence_type"] = input_data["Residence_type"].map(Residence_type_map)
    input_data["smoking_status"] = input_data["smoking_status"].map(smoking_status_map)

    # Chuẩn hóa các đặc trưng số bằng bộ scaler đã lưu
    numeric_features = ["age", "avg_glucose_level", "bmi"]
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])
    
    # Dự đoán với mô hình
    prediction = model.predict(input_data)

    # Hiển thị kết quả dự đoán
    if prediction[0] == 1:
        st.error("Kết quả: Có khả năng bị đột quỵ!")
    else:
        st.success("Kết quả: Khả năng đột quỵ thấp!")
