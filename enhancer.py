import time
import numpy as np
import cv2
import os
import tensorflow as tf
from keras.layers import TFSMLayer

# Disable GPU (optional but recommended for Hugging Face)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load SavedModel using Keras 3 compatible TFSMLayer
model = TFSMLayer("enhancement_model", call_endpoint="serving_default")

def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

def enhance_image(image):
    start = time.time()
    processed = preprocess_image(image)
    output = model(processed)

    # Handle dict output (some SavedModels return a dict)
    if isinstance(output, dict):
        output = list(output.values())[0]

    prediction = np.clip(output[0].numpy() * 255, 0, 255).astype("uint8")
    end = time.time()
    return prediction, end - start
