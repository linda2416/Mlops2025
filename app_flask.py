from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

FASTAPI_URL = "http://127.0.0.1:8000/predict/"

@app.route('/')
def home():
    return '''
        <form action="/predict" method="post">
            <input type="text" name="features" placeholder="Enter features as comma-separated values" required>
            <input type="submit" value="Predict">
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form['features']
    feature_list = [float(x) for x in features.split(',')]
    response = requests.post(FASTAPI_URL, json={"features": feature_list})
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True)
