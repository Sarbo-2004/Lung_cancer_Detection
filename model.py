# model.py
import tensorflow as tf
import numpy as np
from PIL import Image
import io

IMG_WIDTH, IMG_HEIGHT = 224, 224
classes = ["lung_aca", "lung_n", "lung_scc"]

def load_model():
    return tf.keras.models.load_model("best_lung_model.h5")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_bytes, model, threshold=0.7):
    img_array = preprocess_image(image_bytes)
    pred = model.predict(img_array)
    pred_probs = pred[0]

    max_prob = np.max(pred_probs)
    predicted_class = classes[np.argmax(pred_probs)]

    if max_prob < threshold:
        return "Uncertain", max_prob * 100, pred_probs.tolist()
    else:
        return predicted_class, max_prob * 100, pred_probs.tolist()
