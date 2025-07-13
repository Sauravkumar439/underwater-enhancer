import gradio as gr
from PIL import Image
import numpy as np
import cv2
import time
from enhancer import enhance_image

# Main processing function
def process_image(input_img):
    if input_img is None:
        return None, None, "No image uploaded", None

    img = np.array(input_img.convert("RGB"))
    enhanced, elapsed = enhance_image(img)
    enhanced_pil = Image.fromarray(enhanced)
    return input_img, enhanced_pil, f"{elapsed:.2f} seconds", enhanced_pil

# App UI layout and styling
with gr.Blocks(css="""
body {
    background: linear-gradient(135deg, #eaf6ff, #ffffff);
    font-family: 'Segoe UI', sans-serif;
    animation: fadeIn 0.6s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
h1, p {
    text-align: center;
    animation: fadeInUp 0.8s ease-in-out;
}
@keyframes fadeInUp {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
.gr-button {
    background: linear-gradient(to right, #007cf0, #00dfd8);
    color: white;
    font-size: 16px;
    border-radius: 10px;
    padding: 10px 20px;
    transition: all 0.3s ease-in-out;
}
.gr-button:hover {
    background: linear-gradient(to right, #0050b3, #00b0b0);
    transform: scale(1.05);
}
footer {display: none !important;}
""") as demo:

    gr.Markdown("""
        <h1>🌊 Underwater Image Enhancer</h1>
        <p style="font-size: 16px; color: #333;">
            Upload your underwater photo and enhance it using a deep learning model (U-Net + CBAM).
        </p>
        <hr>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="📤 Upload Underwater Image")
            enhance_btn = gr.Button("✨ Enhance Image")

        with gr.Column(scale=2):
            with gr.Row():
                original = gr.Image(label="Original Image")
                enhanced = gr.Image(label="Enhanced Image")

            time_text = gr.Textbox(label="⏱️ Processing Time", interactive=False)
            download_file = gr.File(label="📥 Download Enhanced Image", file_types=[".png"])

    enhance_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[original, enhanced, time_text, download_file]
    )

demo.launch()
