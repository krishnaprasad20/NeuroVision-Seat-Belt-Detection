import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

model = load_model("seatbelt_detector.h5")

class_labels = {0: "with_seatbelt", 1: "without_seatbelt"}

st.set_page_config(page_title="Seatbelt Detection", page_icon="🚗")

st.title(" Seatbelt Detection System")
st.write("Upload an image to check if the driver is wearing a seatbelt.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    display_image = Image.open(uploaded_file).convert("RGB")
    st.image(display_image, caption="Uploaded Image", use_container_width=True)

    image = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(image)

    img_array = img_array / 255.0
    img_array = np.clip(img_array * 1.2, 0, 1)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    threshold = 0.6
    predicted_class = 1 if prediction > threshold else 0
    label = class_labels[predicted_class]
    confidence = prediction*100 if predicted_class == 1 else (1 - prediction) * 100

    st.subheader(" Prediction Result")

    if predicted_class == 0:
        st.success(f"success {label}")
    else:
        st.error(f"error {label}")

    st.write(f"**Confidence:** {confidence:.2f}%")
    st.progress(int(confidence))

    st.caption(f"Raw prediction value: {prediction:.4f}")
    st.caption(f"Predicted class index: {predicted_class}")
    