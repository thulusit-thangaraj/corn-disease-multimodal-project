import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = load_model("crop_disease_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

classes = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

st.title("Corn Leaf Disease Predictor")
st.write("Upload a leaf image and enter soil/rainfall values to predict the disease.")

uploaded_img = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

N = st.number_input("Nitrogen (N)", 0, 150, 50)
P = st.number_input("Phosphorus (P)", 0, 150, 50)
K = st.number_input("Potassium (K)", 0, 150, 50)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, 100.0)

if uploaded_img and st.button("Predict"):
    img = load_img(uploaded_img, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    tab_input = scaler.transform([[N, P, K, temperature, humidity, ph, rainfall]])

    prediction = model.predict([img_array, tab_input])
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Predicted Disease: {predicted_class} ({confidence:.1f}% confidence)")
    st.image(uploaded_img, caption="Uploaded Leaf", width=300)