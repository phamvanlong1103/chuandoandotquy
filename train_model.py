import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

# Thư viện SMOTE để cân bằng dữ liệu
from imblearn.over_sampling import SMOTE

# 1. Đọc dữ liệu
data_path = "data/healthcare-dataset-stroke-data.csv"
data = pd.read_csv(data_path)

# 2. Tiền xử lý dữ liệu
# Loại bỏ cột 'id' nếu có
if 'id' in data.columns:
    data.drop("id", axis=1, inplace=True)

# Điền giá trị thiếu cho cột 'bmi' (nếu có)
data['bmi'] = data['bmi'].fillna(data['bmi'].median())

# Mã hóa các biến phân loại bằng LabelEncoder
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# 5. Cân bằng dữ liệu huấn luyện bằng SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 6. Chuẩn hóa các đặc trưng số
scaler = StandardScaler()
numeric_features = ["age", "avg_glucose_level", "bmi"]
X_train_res[numeric_features] = scaler.fit_transform(X_train_res[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# 7. Huấn luyện mô hình (RandomForestClassifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train_res, y_train_res)

# 8. Đánh giá mô hình trên tập kiểm tra
y_pred = model.predict(X_test)
print("Báo cáo phân loại:")
print(classification_report(y_test, y_pred))

# 9. Tạo thư mục models (nếu chưa có) và lưu mô hình + scaler
if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(model, "models/stroke_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Model và scaler đã được lưu vào thư mục models!")
