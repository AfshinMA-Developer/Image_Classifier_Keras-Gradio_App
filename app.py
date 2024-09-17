import requests
import tensorflow as tf
import gradio as gr
from PIL import Image
import numpy as np

def classify_image(input_image):
    # Download human-readable labels for ImageNet.
    try:
        response = requests.get("https://git.io/JJkYN")
        response.raise_for_status()  # Ensure the request was successful
        labels = response.text.split("\n")
    except Exception as e:
        print("Error fetching labels:", e)
        labels = ["Unknown"] * 1000  # Fallback in case the request fails

    # Load the MobileNetV2 model
    inception_net = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=True,
        weights="imagenet",
        classes=1000,
        classifier_activation="softmax"
    )

    # Handle input_image (ensure it's a PIL Image)
    if isinstance(input_image, str):  # If it's a file path or URL
        input_image = Image.open(input_image).convert("RGB")
    elif isinstance(input_image, np.ndarray):  # If it's a numpy array
        input_image = Image.fromarray(input_image).convert("RGB")

    # Resize the image to 224x224
    input_image = input_image.resize((224, 224))

    # Convert image to a numpy array
    input_image = np.array(input_image)

    # Ensure it's in the right format (RGB channels only)
    if input_image.shape[-1] == 4:  # If there's an alpha channel
        input_image = input_image[..., :3]  # Remove the alpha channel

    # Reshape for a single prediction
    input_image = input_image.reshape((1, 224, 224, 3))

    # Preprocess the image
    input_image = tf.keras.applications.mobilenet_v2.preprocess_input(input_image)

    # Perform prediction
    prediction = inception_net.predict(input_image).flatten()

    # Get the top indices and their respective confidence scores
    top_indices = np.argsort(prediction)[-3:][::-1]  # Get the top 3 indices
    confidences = {labels[i]: float(prediction[i]) for i in top_indices}
    
    return confidences

image = gr.Image(interactive=True, label="Upload Image")
label = gr.Label(num_top_classes=3, label="Top Predictions")

demo = gr.Interface(
    title="Image Classifier Keras",
    fn=classify_image,
    inputs=image,
    outputs=label,
    examples=[["./images/banana.jpg"], ["./images/car.jpg"], ["./images/guitar.jpg"], ["./images/lion.jpg"]],
    theme="default",
    css=".footer{display:none !important}"
)

if __name__ == "__main__":
    demo.launch(share=True)