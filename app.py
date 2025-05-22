import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import cv2
import os
import socket
import json
import base64
from pymongo import MongoClient
from datetime import datetime


# TCP constants
SERVER_IP = "127.0.0.1"
SERVER_PORT = 12345
MONGO_URI = "mongodb+srv://sgshivapalaksha:phGTYpu27cidpgVh@mycluster0.hx7xo.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["bacteria_db"]
collection = db["predictions"]

def send_to_tcp_server(image, prediction, encryption_key, transformation):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    encrypted_prediction = ''.join(
        chr(ord(char) ^ ord(encryption_key[i % len(encryption_key)]))
        for i, char in enumerate(prediction)
    )

    data = {
        "image": img_base64,
        "prediction": encrypted_prediction,
        "encryption_key": encryption_key
    }

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((SERVER_IP, SERVER_PORT))
            s.sendall(json.dumps(data).encode("utf-8"))
            response = s.recv(1024).decode()
            st.success(f"📡 TCP Server Response: {response}")

        # Save to MongoDB
        record = {
            "transformation": transformation,
            "prediction": prediction,
            "encrypted_prediction": encrypted_prediction,
            "image_base64": img_base64,
            "timestamp": datetime.now()
        }
        collection.insert_one(record)

    except Exception as e:
        st.error(f"❌ Failed to send data to TCP server or MongoDB: {e}")


# Set page configuration
st.set_page_config(page_title="Micro Organism Analyzer", layout="wide")

page = st.sidebar.radio("Navigation", [
    "Home",
    "Image Classifier",
    "Comparison Page (Before vs After Transformation)",
    "Feedback",
    "📁Records"  # <-- New Page
])


# --- PAGE 1: HOME ---
if page == "Home":
    st.title("🔬 Welcome to the Micro Organism Analysis Portal")
    st.header("📖 What are Micro Organisms?")
    st.write("""
    Microorganisms or microbes are microscopic organisms that exist as unicellular, multicellular, or cell clusters. 
    They are found in all types of environments and can be both beneficial and harmful to other forms of life.
    """)
    st.subheader("🦠 Types of Micro Organisms")
    st.markdown("""
    - **Bacteria**: Single-celled organisms that thrive in diverse environments.
    - **Viruses**: Infectious agents that replicate only inside the living cells of organisms.
    - **Fungi**: Includes yeasts and molds which decompose and recycle organic material.
    - **Protozoa**: Single-celled organisms with animal-like behaviors.
    - **Algae**: Photosynthetic organisms found mostly in water.
    """)
    st.subheader("🎥 Micro Organism Video")
    st.video("C:/sem 6/golang cat-2 bacteria/bacteria prediction/Bacteria.mp4")

# --- PAGE 2: IMAGE CLASSIFIER ---
elif page == "Image Classifier":
    st.title("🦠 Micro Organism Image Classifier with Selected Transformations")

    MODEL_PATH = 'inception-v3.keras'
    model = tf.keras.models.load_model(MODEL_PATH)

    class_names = sorted(os.listdir("C:/sem 6/golang cat-2 bacteria/bacteria prediction/Micro_Organism"))
    IMG_SIZE = (256, 256)

    def preprocess_image(image):
        img = image.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_image(image):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        pred_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return pred_class, confidence

    def apply_transformation(image, transformation):
        if transformation == "Upside Down":
            return ImageOps.flip(image)
        elif transformation == "Rotation by 45 degrees":
            return image.rotate(45)
        elif transformation == "Rotation using Shear Matrix":
            return Image.fromarray(
                cv2.warpAffine(
                    np.array(image),
                    cv2.getAffineTransform(
                        np.float32([[50, 50], [200, 50], [50, 200]]),
                        np.float32([[10, 100], [200, 50], [100, 250]])
                    ),
                    (image.width, image.height)
                )
            )
        elif transformation == "Grayscale":
            return ImageOps.grayscale(image).convert('RGB')
        elif transformation == "Box Blur":
            return image.filter(ImageFilter.BoxBlur(5))
        elif transformation == "Gaussian Blur":
            return image.filter(ImageFilter.GaussianBlur(5))
        elif transformation == "Sobel Edge Detection":
            return Image.fromarray(
                cv2.cvtColor(
                    cv2.convertScaleAbs(
                        cv2.Sobel(
                            cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY),
                            cv2.CV_64F, 1, 1, ksize=5
                        )
                    ),
                    cv2.COLOR_GRAY2RGB
                )
            )

    uploaded_file = st.file_uploader("Upload an image of a bacteria", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Original Uploaded Image', use_column_width=True)

        st.markdown("---")
        st.subheader("🔧 Select Transformations")

        selected_transformations = []
        if st.checkbox("Upside Down Transformation"):
            selected_transformations.append("Upside Down")
        if st.checkbox("Rotation by 45 degrees"):
            selected_transformations.append("Rotation by 45 degrees")
        if st.checkbox("Rotation using Shear Matrix"):
            selected_transformations.append("Rotation using Shear Matrix")
        if st.checkbox("Grayscale Conversion"):
            selected_transformations.append("Grayscale")
        if st.checkbox("Box Blur"):
            selected_transformations.append("Box Blur")
        if st.checkbox("Gaussian Blur"):
            selected_transformations.append("Gaussian Blur")
        if st.checkbox("Sobel Edge Detection"):
            selected_transformations.append("Sobel Edge Detection")

        if selected_transformations:
            st.markdown("---")
            st.subheader("🖼 Transformed Images and Predictions")

            encryption_key = st.text_input("🔐 Enter 8-digit Encryption Key to Send to Server")

            for transformation in selected_transformations:
                st.markdown(f"### 🛠 {transformation}")
                transformed_img = apply_transformation(image, transformation)

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(transformed_img, caption=f"{transformation} Image", use_column_width=True)
                with col2:
                    pred_class, confidence = predict_image(transformed_img)
                    st.markdown(f"### 🧬 Predicted Class:")
                    st.success(f"{pred_class}")
                    st.markdown(f"### 🔍 Confidence:")
                    st.info(f"{confidence:.2f}%")

                    # Send to TCP Server if encryption key is valid
                    if encryption_key.isdigit() and len(encryption_key) == 8:
                        if st.button(f"📤 Send '{transformation}' data to server", key=transformation):
                            send_to_tcp_server(transformed_img, pred_class, encryption_key, transformation)

                    else:
                        st.warning("Please enter a valid 8-digit encryption key to enable sending.")
        else:
            st.info("Please select at least one transformation above ☝.")
            


# --- PAGE 3: COMPARISON PAGE ---
elif page == "Comparison Page (Before vs After Transformation)":
    st.title("🆚 Comparison Page: Original vs Transformed Prediction")

    MODEL_PATH = 'inception-v3.keras'
    model = tf.keras.models.load_model(MODEL_PATH)

    class_names = sorted(os.listdir("C:/sem 6/golang cat-2 bacteria/bacteria prediction/Micro_Organism"))
    IMG_SIZE = (256, 256)

    def preprocess_image(image):
        img = image.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_image(image):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        pred_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return pred_class, confidence

    def apply_transformation(image, transformation):
        if transformation == "Upside Down":
            return ImageOps.flip(image)
        elif transformation == "Rotation by 45 degrees":
            return image.rotate(45)
        elif transformation == "Rotation using Shear Matrix":
            return Image.fromarray(
                cv2.warpAffine(
                    np.array(image),
                    cv2.getAffineTransform(
                        np.float32([[50, 50], [200, 50], [50, 200]]),
                        np.float32([[10, 100], [200, 50], [100, 250]])
                    ),
                    (image.width, image.height)
                )
            )
        elif transformation == "Grayscale":
            return ImageOps.grayscale(image).convert('RGB')
        elif transformation == "Box Blur":
            return image.filter(ImageFilter.BoxBlur(5))
        elif transformation == "Gaussian Blur":
            return image.filter(ImageFilter.GaussianBlur(5))
        elif transformation == "Sobel Edge Detection":
            return Image.fromarray(
                cv2.cvtColor(
                    cv2.convertScaleAbs(
                        cv2.Sobel(
                            cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY),
                            cv2.CV_64F, 1, 1, ksize=5
                        )
                    ),
                    cv2.COLOR_GRAY2RGB
                )
            )

    def generate_pdf_report(orig_img, trans_img, transformation, orig_class, orig_conf, trans_class, trans_conf):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 16)
        c.drawString(30, height - 40, "🧾 Micro Organism Prediction Comparison Report")

        c.setFont("Helvetica", 12)
        c.drawString(30, height - 70, f"Transformation Applied: {transformation}")
        c.drawString(30, height - 90, f"Original Prediction: {orig_class} ({orig_conf:.2f}%)")
        c.drawString(30, height - 110, f"Transformed Prediction: {trans_class} ({trans_conf:.2f}%)")

        confidence_diff = trans_conf - orig_conf
        c.drawString(30, height - 130, f"Confidence Change: {confidence_diff:.2f}%")
        c.drawString(30, height - 150, f"Transformation Impact Score: {abs(confidence_diff):.2f}")

        if orig_class != trans_class:
            c.setFillColorRGB(1, 0, 0)
            c.drawString(30, height - 170, f"⚠️ Misclassification Detected")
            c.setFillColorRGB(0, 0, 0)
        else:
            c.setFillColorRGB(0, 0.5, 0)
            c.drawString(30, height - 170, "✅ Prediction remains consistent.")
            c.setFillColorRGB(0, 0, 0)

        # Convert images to ImageReader and resize
        orig_io = BytesIO()
        orig_img.save(orig_io, format='PNG')
        orig_reader = ImageReader(orig_io)

        trans_io = BytesIO()
        trans_img.save(trans_io, format='PNG')
        trans_reader = ImageReader(trans_io)

        c.drawString(30, height - 200, "Original Image:")
        c.drawImage(orig_reader, 30, height - 400, width=250, preserveAspectRatio=True, mask='auto')

        c.drawString(300, height - 200, f"Transformed Image ({transformation}):")
        c.drawImage(trans_reader, 300, height - 400, width=250, preserveAspectRatio=True, mask='auto')

        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer

    uploaded_file = st.file_uploader("Upload an image for comparison", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        transformation = st.selectbox("Choose a Transformation", [
            "Upside Down", "Rotation by 45 degrees", "Rotation using Shear Matrix",
            "Grayscale", "Box Blur", "Gaussian Blur", "Sobel Edge Detection"
        ])

        transformed_image = apply_transformation(image, transformation)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
            orig_class, orig_conf = predict_image(image)
            st.markdown(f"**Prediction**: `{orig_class}`")
            st.markdown(f"**Confidence**: `{orig_conf:.2f}%`")

        with col2:
            st.image(transformed_image, caption=f"Transformed: {transformation}", use_column_width=True)
            trans_class, trans_conf = predict_image(transformed_image)
            st.markdown(f"**Prediction**: `{trans_class}`")
            st.markdown(f"**Confidence**: `{trans_conf:.2f}%`")

        st.markdown("---")
        st.subheader("📊 Comparison Metrics")

        confidence_diff = trans_conf - orig_conf
        impact_score = abs(confidence_diff)

        if orig_class != trans_class:
            st.error(f"⚠️ Misclassification Detected: Changed from `{orig_class}` to `{trans_class}`")
        else:
            st.success("✅ Prediction remains consistent.")

        st.info(f"📉 Confidence Change: `{confidence_diff:.2f}%`")
        st.warning(f"🧮 Transformation Impact Score: `{impact_score:.2f}`")

        st.markdown("---")

        pdf = generate_pdf_report(
            image, transformed_image, transformation,
            orig_class, orig_conf, trans_class, trans_conf
        )

        st.download_button(
            label="📄 Download Comparison Report (PDF)",
            data=pdf,
            file_name="comparison_report.pdf",
            mime="application/pdf"
        )

# --- PAGE 4: FEEDBACK ---
elif page == "Feedback":
    st.title("📩 Feedback Page")

    st.subheader("🧪 Share Your Experience")
    st.image("C:/sem 6/golang cat-2 bacteria/bacteria prediction/feedback_image.jpg", width=500)

    with st.form("feedback_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Your Feedback")

        submitted = st.form_submit_button("Submit")

        if submitted:
            st.success("✅ Thank you for your feedback!")

# --- PAGE 5: MONGODB RECORDS ---
elif page == "📁 Records":
    st.title("📁 Stored Predictions in MongoDB")

    records = list(collection.find().sort("timestamp", -1))
    if not records:
        st.warning("No predictions stored yet.")
    else:
        for rec in records:
            st.markdown("---")
            st.subheader(f"🛠 Transformation: {rec['transformation']}")
            image_data = base64.b64decode(rec["image_base64"])
            image = Image.open(BytesIO(image_data))
            st.image(image, caption="Stored Image", use_column_width=True)
            st.markdown(f"**🔍 Prediction:** `{rec['prediction']}`")
            st.markdown(f"**🕒 Timestamp:** `{rec['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}`")
