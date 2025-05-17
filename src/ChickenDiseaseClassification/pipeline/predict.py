import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

class PredictionPipeline:
    def __init__(self, model_path="artifacts/training/trained_model.h5"):
        self.model = load_model(model_path)  # Load once here
        self.class_names = ["Diseased Chicken", "Healthy Chicken"]  # Adjust if you have more classes

    def predict(self, filename):
        test_image = image.load_img(filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        predictions = self.model.predict(test_image)
        print("Raw model prediction:", predictions)

        result = np.argmax(predictions, axis=1)
        print("Class index prediction:", result)

        prediction = self.class_names[result[0]]
        return [{"image": prediction}]
