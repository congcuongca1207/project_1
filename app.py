from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Tải mô hình đã lưu
model = joblib.load('model/logistic_regression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ biểu mẫu
    try:
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])
        feature4 = float(request.form['feature4'])
    except ValueError:
        return render_template('index.html', prediction='Dữ liệu đầu vào không hợp lệ.')

    # Chuyển đổi dữ liệu thành numpy array
    features_array = np.array([feature1, feature2, feature3, feature4]).reshape(1, -1)
    
    # Dự đoán với mô hình đã tải
    prediction = model.predict(features_array)[0]
    
    # Trả về kết quả dự đoán
    return render_template('index.html', prediction=int(prediction))

if __name__ == "__main__":
    app.run(debug=True)
