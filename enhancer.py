import time
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the model once
model = load_model("enhancement_model", compile=False)


# Resize and normalize the input
def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

# Predict enhanced image and measure time
def enhance_image(image):
    start = time.time()
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    prediction = np.clip(prediction * 255, 0, 255).astype("uint8")
    end = time.time()
    return prediction, end - start
