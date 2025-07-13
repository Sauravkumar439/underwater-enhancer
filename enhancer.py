import time
import numpy as np
import cv2
import tensorflow as tf
from keras.layers import TFSMLayer  # Keras 3 specific import

# Load the model using TFSMLayer (Keras 3 way)
model = TFSMLayer("enhancement_model", call_endpoint="serving_default")

def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

def enhance_image(image):
    start = time.time()
    processed = preprocess_image(image)
    prediction = model(processed)[0].numpy()
    prediction = np.clip(prediction * 255, 0, 255).astype("uint8")
    end = time.time()
    return prediction, end - start
