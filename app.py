from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Tải mô hình đã lưu
model = joblib.load('model/logistic_regression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ biểu mẫu
        feature1 = float(request.form['loan_amnt'])
        feature2 = float(request.form['term'])  # Được giữ nguyên dưới dạng chuỗi
        feature3 = float(request.form['int_rate'])
        feature4 = float(request.form['installment'])
        feature5 = float(request.form['grade'])  # Được giữ nguyên dưới dạng chuỗi
        feature6 = float(request.form['emp_length'])  # Được giữ nguyên dưới dạng chuỗi
        feature7 = float(request.form['home_ownership'])  # Được giữ nguyên dưới dạng chuỗi
        feature8 = float(request.form['annual_inc'])
        feature9 = float(request.form['dti'])
        feature10 = float(request.form['delinq_2yrs'])
        feature11 = float(request.form['fico_range_low'])
        feature12 = float(request.form['fico_range_high'])
    except ValueError:
        return render_template('index.html', prediction='Dữ liệu đầu vào không hợp lệ.')

    # Chuyển đổi dữ liệu thành numpy array
    features_array = np.array([feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12]).reshape(1, -1)
    
    # Dự đoán với mô hình đã tải
    prediction = model.predict(features_array)[0]
    
    # Trả về kết quả dự đoán
    return render_template('index.html', prediction=float(prediction))

if __name__ == "__main__":
    app.run(debug=True)
