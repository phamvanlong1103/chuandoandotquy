import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Tải mô hình và bộ scaler đã huấn luyện
model = joblib.load("models/stroke_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Dự đoán khả năng đột quỵ của bệnh nhân")
st.write("Nhập thông tin bệnh nhân dưới đây:")

# Lấy dữ liệu từ người dùng với các lựa chọn bằng tiếng Việt
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
    # Chuyển đổi "Có"/"Không" thành giá trị số cho biến tăng huyết áp và bệnh tim
    hypertension_val = 1 if hypertension == "Có" else 0
    heart_disease_val = 1 if heart_disease == "Có" else 0

    # Tạo DataFrame từ dữ liệu nhập
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

    # Hiển thị dữ liệu nhập ban đầu
    st.write("Dữ liệu nhập ban đầu:", input_data)

    # Áp dụng mapping cho các biến phân loại theo cách đã sử dụng khi huấn luyện
 # Mapping cho các biến phân loại (đảm bảo khớp với cách LabelEncoder mã hóa trong quá trình huấn luyện)
    gender_map = {"Nam": 1, "Nữ": 0}
    ever_married_map = {"Có": 1, "Không": 0}
    work_type_map = {
    "Công chức": 0,      # Govt_job
    "Không làm việc": 1,  # Never_worked
    "Tư nhân": 2,         # Private
    "Tự kinh doanh": 3,   # Self-employed
    "Trẻ em": 4           # children
    }
    Residence_type_map = {"Đô thị": 1, "Nông thôn": 0}
    smoking_status_map = {
    "Không rõ": 0,            # Unknown
    "Đã hút thuốc": 1,        # formerly smoked
    "Chưa từng hút thuốc": 2,  # never smoked
    "Đang hút thuốc": 3       # smokes
    }

    input_data["gender"] = input_data["gender"].map(gender_map)
    input_data["ever_married"] = input_data["ever_married"].map(ever_married_map)
    input_data["work_type"] = input_data["work_type"].map(work_type_map)
    input_data["Residence_type"] = input_data["Residence_type"].map(Residence_type_map)
    input_data["smoking_status"] = input_data["smoking_status"].map(smoking_status_map)


    # Hiển thị dữ liệu sau mapping
    st.write("Dữ liệu sau mapping:", input_data)

    # Chuẩn hóa các đặc trưng số bằng scaler đã lưu
    numeric_features = ["age", "avg_glucose_level", "bmi"]
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])
    
    # Hiển thị dữ liệu sau chuẩn hóa
    st.write("Dữ liệu sau chuẩn hóa:", input_data)
    
    # Dự đoán với mô hình
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    # Hiển thị xác suất dự đoán
    st.write("Xác suất dự đoán:", prediction_proba)
    
    if prediction[0] == 1:
        st.error("Kết quả: Có khả năng bị đột quỵ!")
    else:
        st.success("Kết quả: Khả năng đột quỵ thấp!")
