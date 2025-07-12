import gradio as gr
from PIL import Image
import numpy as np
import time
import cv2
from enhancer import enhance_image

def process_image(input_img):
    img = np.array(input_img.convert("RGB"))
    enhanced, elapsed = enhance_image(img)
    enhanced_pil = Image.fromarray(enhanced)
    return enhanced_pil, f"{elapsed:.2f} seconds"

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload Underwater Image"),
    outputs=[gr.Image(type="pil", label="Enhanced Image"), gr.Text(label="Processing Time")],
    title="🌊 Underwater Image Enhancer",
    description="Upload a low-quality underwater image and enhance it using a U-Net + CBAM deep learning model."
)

if __name__ == "__main__":
    demo.launch()
