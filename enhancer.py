import time
import numpy as np
import cv2
import os
import tensorflow as tf
from keras.layers import TFSMLayer

# Optional: Disable GPU usage (recommended for Hugging Face CPU backend)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ✅ Load the model using Keras 3's new TFSMLayer API
model = TFSMLayer("enhancement_model", call_endpoint="serving_default")

# ✅ Resize + normalize input image
def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

# ✅ Run prediction and measure time taken
def enhance_image(image):
    start = time.time()
    processed = preprocess_image(image)
    
    # Run inference (TFSMLayer returns a dict or list)
    prediction = model(processed)[0].numpy()
    
    # Post-process output
    prediction = np.clip(prediction * 255, 0, 255).astype("uint8")
    end = time.time()
    return prediction, end - start
