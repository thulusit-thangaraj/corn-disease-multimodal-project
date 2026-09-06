# Multimodal Deep Learning for Crop Disease Severity Prediction

A deep learning project predicting corn leaf disease using leaf images combined with soil and rainfall data.

## Project Overview
This project builds a multimodal model combining leaf images with soil and rainfall data, and compares it to an image-only baseline model.

## Dataset
Leaf images: Corn/Maize Leaf Disease Dataset (Kaggle), 4188 images, 4 classes (Blight, Common Rust, Gray Leaf Spot, Healthy). Soil/rainfall data: Crop Recommendation Dataset (Kaggle), maize rows only.

## Model
Image branch uses MobileNetV2 (pretrained). Tabular branch processes soil and rainfall features. Combined for final classification.

## Results
Multimodal model: ~93% test accuracy. Image-only model: similar accuracy, showing no significant improvement from the randomly-paired soil/rainfall data.

## How to Run
Install: pip install pandas numpy tensorflow scikit-learn matplotlib seaborn streamlit
Train: python check.py
Run app: streamlit run app.py

## Tech Stack
Python, TensorFlow, Keras, MobileNetV2, Scikit-learn, Streamlit
