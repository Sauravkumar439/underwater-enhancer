# Underwater Image Enhancer

A deep learning-powered app to enhance low-quality underwater images using a U-Net model with CBAM (Convolutional Block Attention Module).

Model trained on the [EUVP dataset](https://www.kaggle.com/datasets/ejlok1/underwater-image-enhancement-euvp).

## Features

- Upload an underwater image
- Automatically enhances image clarity and colors
- Built with U-Net + CBAM in TensorFlow
- Keras 3-compatible SavedModel format
- Deployed with Gradio on Hugging Face Spaces
- Animated UI with download option & processing time

## Sample Screenshot

![demo](https://huggingface.co/spaces/sauravkumar439/underwater-enhancer/resolve/main/demo.jpg)

<img width="1309" height="490" alt="image" src="https://github.com/user-attachments/assets/3396b059-b3b9-42c2-8ca1-958e2bd9e5e8" />


## Try It Live

👉 [Click to Use the App](https://huggingface.co/spaces/sauravkumar439/underwater-enhancer)

---

## Tech Stack

- Python · TensorFlow · Keras 3
- OpenCV · NumPy · PIL
- Gradio (UI Framework)
- Hugging Face Spaces (Deployment)

## How to Run Locally

git clone https://huggingface.co/spaces/sauravkumar439/underwater-enhancer
cd underwater-enhancer
pip install -r requirements.txt
python app.py
