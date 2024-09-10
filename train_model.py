import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import joblib

# Tạo dữ liệu mẫu
data = load_iris()
X = data.data
y = data.target

# Chỉ chọn 2 lớp cho ví dụ đơn giản
X = X[y != 2]
y = y[y != 2]

# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tạo mô hình Logistic Regression
model = LogisticRegression()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Lưu mô hình vào file
joblib.dump(model, 'model/logistic_regression_model.pkl')
print("Mô hình đã được lưu vào 'model/logistic_regression_model.pkl'")
