import os
import numpy as np
import tensorflow as tf
import warnings
from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk
warnings.filterwarnings("ignore")

# ------------ CONSTANTS -----------------
MODEL_PATH ="Best_Cattle_Breed.h5"   #your trained cattle breed model
DATA_DIR = "data/"
LABELS_PATH = "labels.txt"


def load_class_names():
    if os.path.isfile(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as labels_file:
            labels = [line.strip() for line in labels_file if line.strip()]
        if labels:
            return labels

    if os.path.isdir(DATA_DIR):
        labels = sorted(
            name for name in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, name))
        )
        if labels:
            return labels

    output_shape = model.output_shape
    num_classes = output_shape[-1] if isinstance(output_shape, tuple) else output_shape[0][-1]
    return [f"Class {index}" for index in range(num_classes)]

# -------------------- LOAD MODEL --------------------------
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = load_class_names()
print(f"Loaded {len(CLASS_NAMES)} classes: {CLASS_NAMES}")

# -------------------- IMAGE PREPROCESS --------------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img = np.asarray(img, dtype=np.float32)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)

# ---------------- PREDICTION FUNCTION ------------------------
def predict_image(image_path):
    try:
        img = preprocess_image(image_path)
        preds = model.predict(img)
        class_id = np.argmax(preds[0])
        confidence = preds[0][class_id] * 100
        return CLASS_NAMES[class_id], confidence
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None, None
    
# ------------------ GUI CALLBACKS ----------------------------
def browse_image():
    file_path = filedialog.askopenfilename(
        title=" Select Cattle Image",
        filetypes=[(" Image files", "*.jpg *.jpeg *.png")]
    ) 
    if file_path:
        image = Image.open(file_path)
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

        predicted_class, confidence = predict_image(file_path)
        if predicted_class:
            result_label.config(
                text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%"
            )


# --------------------- GUI SETUP -----------------------
root = Tk()
root.title("Cattle Breed Classifier")
root.geometry("400x500")

Label(root, text="Indian Cattle Breed Classifier", font=("Arial", 16)).pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

browse_btn = Button(root, text= "Select Cattle Image", command=browse_image)
browse_btn.pack(pady=20)

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
