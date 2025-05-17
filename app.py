from flask import Flask, render_template, request, jsonify
import os
from flask_cors import CORS, cross_origin
from ChickenDiseaseClassification.pipeline.predict import PredictionPipeline
from ChickenDiseaseClassification.utils.common import decodeImage

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

# Load model ONCE globally
classifier = PredictionPipeline()  # adjust path as needed
filename = "inputImage.jpg"

@app.route("/", methods=['GET'])
@cross_origin()
def home(): 
    return render_template('index.html')

@app.route("/train", methods=['GET'])
@cross_origin()
def train_model():
    os.system("python main.py")
    return "✅ Model Trained Successfully"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict_route():
    try:
        image = request.json['image']
        decodeImage(image, filename)
        result = classifier.predict(filename)
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
