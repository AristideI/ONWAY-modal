import uvicorn
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load the trained model
model = load_model("models/traffic_model.keras")  # Change path if needed

# Define the class labels
class_labels = ["Empty", "High", "Low", "Medium", "Traffic Jam"]


def preprocess_image(img: Image.Image):
    img = img.resize((150, 150))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand to batch dimension
    img_array = img_array / 255.0  # Normalize (same as training)
    return img_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    img = Image.open(BytesIO(contents))

    # Preprocess and predict
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    result = {
        "predicted_label": class_labels[predicted_class],
        "confidence": round(float(confidence), 2),
    }
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
