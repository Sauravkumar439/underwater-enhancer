import gradio as gr
from PIL import Image
import numpy as np
import cv2
import time
from enhancer import enhance_image

def process_image(input_img):
    img = np.array(input_img.convert("RGB"))
    enhanced, elapsed = enhance_image(img)
    enhanced_pil = Image.fromarray(enhanced)
    return input_img, enhanced_pil, f"{elapsed:.2f} seconds", enhanced_pil

with gr.Blocks(css="""
body {
    background: linear-gradient(135deg, #eaf6ff, #ffffff);
    font-family: 'Segoe UI', sans-serif;
    animation: fadeIn 1s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

h1, p {
    text-align: center;
    margin-bottom: 0.5em;
    animation: fadeInUp 1s ease-in-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.gr-button {
    background: linear-gradient(to right, #007cf0, #00dfd8);
    color: white;
    font-size: 16px;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 4px 14px rgba(0,0,0,0.1);
}
.gr-button:hover {
    background: linear-gradient(to right, #0066cc, #00b8b8);
    transform: scale(1.05);
}

.gr-image {
    transition: opacity 0.5s ease;
}

footer {display: none !important;}
""") as demo:

    gr.Markdown("""
    <h1>🌊 Underwater Image Enhancer</h1>
    <p style="font-size: 17px; color: #333;">
        Upload your underwater image and watch it come to life in vivid clarity using deep learning (U-Net + CBAM).
    </p>
    <hr>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="📤 Upload Image")
            enhance_button = gr.Button("✨ Enhance Image")

        with gr.Column(scale=2):
            with gr.Row():
                original_output = gr.Image(label="Original Image", show_label=True)
                enhanced_output = gr.Image(label="Enhanced Image", show_label=True)

            time_output = gr.Textbox(label="🕒 Time Taken", interactive=False)
            download_output = gr.File(label="📥 Download Enhanced Image", file_types=[".png"])

    enhance_button.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[original_output, enhanced_output, time_output, download_output]
    )

demo.launch()
