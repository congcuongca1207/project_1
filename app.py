import joblib
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer

# Tải mô hình đã lưu
model = joblib.load('model/best_sentiment_model.pkl')

# Tải vectorizer (nếu bạn đã lưu nó trước đó, nếu không, bạn sẽ phải huấn luyện lại nó)
vectorizer = joblib.load('model/vectorizer.pkl')  # Hoặc tạo lại nếu cần

# Khởi tạo Flask app
app = Flask(__name__)

# Trang chủ
@app.route('/')
def home():
    return render_template('index.html')

# Trang dự đoán cảm xúc
@app.route('/predict', methods=['POST'])
def predict():
    # Lấy văn bản người dùng nhập
    text = request.form['text']
    
    # Biến đổi văn bản đầu vào
    text_vectorized = vectorizer.transform([text])
    
    # Dự đoán
    prediction = model.predict(text_vectorized)
    
    # Chuyển đổi kết quả dự đoán thành nhãn cảm xúc
    sentiment = ['Negative', 'Neutral', 'Positive']
    result = sentiment[prediction[0]]
    
    return render_template('index.html', prediction_text='Sentiment: {}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)
