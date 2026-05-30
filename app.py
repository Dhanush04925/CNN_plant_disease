import streamlit as st
import numpy as np
import pickle

from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("models/cnn_model.h5")

with open("models/labels.pkl", "rb") as f:
    class_indices = pickle.load(f)

labels = {v: k for k, v in class_indices.items()}

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿"
)

st.title("🌿 Plant Disease Detection System")

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )

    img = image.resize((64,64))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(prediction)

    confidence = prediction[predicted_index] * 100

    class_name = labels[predicted_index]

    # Reject non-plant images
    if class_name.lower() == "nonplant":

        st.error(
            "❌ Invalid Image\n\nPlease upload a plant leaf image."
        )

        st.stop()

    # Healthy plants
    if "healthy" in class_name.lower():

        st.success(
            "🌿 Plant Status: HEALTHY"
        )

    else:

        disease = class_name.replace(
            "___",
            " - "
        ).replace(
            "_",
            " "
        )

        st.error(
            "🦠 Plant Status: DISEASED"
        )

        st.write(
            f"### Disease: {disease}"
        )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    st.subheader("Top Predictions")

    top3 = np.argsort(prediction)[::-1]

    count = 0

    for idx in top3:

        disease = labels[idx]

        if disease.lower() == "nonplant":
            continue

        disease = disease.replace(
            "___",
            " - "
        ).replace(
            "_",
            " "
        )

        st.write(
            f"• {disease}: {prediction[idx]*100:.2f}%"
        )

        count += 1

        if count == 3:
            break