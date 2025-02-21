import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

# 1. Đọc dữ liệu
# Đảm bảo rằng file dữ liệu 'healthcare-dataset-stroke-data.csv' nằm trong thư mục data/
data_path = "data/healthcare-dataset-stroke-data.csv"
data = pd.read_csv(data_path)

# 2. Tiền xử lý dữ liệu
# Nếu có cột 'id', loại bỏ nó vì không cần dùng để dự đoán
if 'id' in data.columns:
    data.drop("id", axis=1, inplace=True)

# Điền giá trị thiếu cho cột 'bmi' (nếu có)
data['bmi'].fillna(data['bmi'].median(), inplace=True)

# Mã hóa các biến phân loại
# Ở đây sử dụng LabelEncoder cho đơn giản, bạn có thể thay đổi nếu cần mapping cụ thể
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['ever_married'] = le.fit_transform(data['ever_married'])
data['work_type'] = le.fit_transform(data['work_type'])
data['Residence_type'] = le.fit_transform(data['Residence_type'])
data['smoking_status'] = le.fit_transform(data['smoking_status'])

# 3. Tách dữ liệu thành đặc trưng (X) và nhãn (y)
X = data.drop("stroke", axis=1)
y = data["stroke"]

# 4. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Chuẩn hóa các đặc trưng số
scaler = StandardScaler()
numeric_features = ["age", "avg_glucose_level", "bmi"]
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# 6. Huấn luyện mô hình (RandomForestClassifier được dùng ở đây)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Đánh giá mô hình trên tập kiểm tra
y_pred = model.predict(X_test)
print("Báo cáo phân loại:")
print(classification_report(y_test, y_pred))

# 8. Tạo thư mục models nếu chưa tồn tại
if not os.path.exists("models"):
    os.makedirs("models")

# 9. Lưu mô hình và bộ scaler vào file
joblib.dump(model, "models/stroke_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Model và scaler đã được lưu vào thư mục models!")
