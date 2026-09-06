import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle

# ---- STEP A: Find your files automatically ----
csv_matches = glob.glob("**/*.csv", recursive=True)
print("CSV files found:", csv_matches)
csv_file = csv_matches[0]

blight_matches = glob.glob("**/Blight", recursive=True)
print("Blight folder found at:", blight_matches)
data_root = os.path.dirname(blight_matches[0]) or "."
print("Using data_root:", data_root)

for folder in os.listdir(data_root):
    path = os.path.join(data_root, folder)
    if os.path.isdir(path):
        print(folder, "->", len(os.listdir(path)), "images")

# ---- STEP B: Load soil/rainfall data ----
soil_df = pd.read_csv(csv_file)
maize_df = soil_df[soil_df['label'] == 'maize'].reset_index(drop=True)
print("Maize rows available:", len(maize_df))

# ---- STEP C: Build combined table ----
classes = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
records = []
for cls in classes:
    folder_path = os.path.join(data_root, cls)
    for img_file in os.listdir(folder_path):
        soil_row = maize_df.sample(1).iloc[0]
        records.append({
            "image_path": os.path.join(folder_path, img_file),
            "label": cls,
            "N": soil_row["N"], "P": soil_row["P"], "K": soil_row["K"],
            "temperature": soil_row["temperature"],
            "humidity": soil_row["humidity"],
            "ph": soil_row["ph"],
            "rainfall": soil_row["rainfall"]
        })
combined_df = pd.DataFrame(records)
print("Combined table shape:", combined_df.shape)

# ---- STEP D: Encode, scale, split ----
le = LabelEncoder()
combined_df["label_encoded"] = le.fit_transform(combined_df["label"])
tabular_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
scaler = StandardScaler()
combined_df[tabular_features] = scaler.fit_transform(combined_df[tabular_features])
train_df, test_df = train_test_split(
    combined_df, test_size=0.2, random_state=42, stratify=combined_df["label_encoded"]
)
print("Train size:", len(train_df), "Test size:", len(test_df))

# ---- STEP E: Load images ----
IMG_SIZE = 128
def load_images(df):
    images = []
    for path in df["image_path"]:
        img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
        img = img_to_array(img) / 255.0
        images.append(img)
    return np.array(images)

print("Loading training images... this takes a while, please wait")
X_train_img = load_images(train_df)
X_test_img = load_images(test_df)
X_train_tab = train_df[tabular_features].values
X_test_tab = test_df[tabular_features].values
y_train = train_df["label_encoded"].values
y_test = test_df["label_encoded"].values
print("Images loaded:", X_train_img.shape, X_test_img.shape)

# ---- STEP F: Class weights ----
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# ---- STEP G: Build model ----
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")
x = base_model(image_input, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)

tab_input = Input(shape=(len(tabular_features),), name="tabular_input")
y = Dense(32, activation='relu')(tab_input)
y = Dense(16, activation='relu')(y)

combined = Concatenate()([x, y])
z = Dense(32, activation='relu')(combined)
z = Dropout(0.3)(z)
output = Dense(len(classes), activation='softmax')(z)

model = Model(inputs=[image_input, tab_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---- STEP H: Train ----
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
history = model.fit(
    [X_train_img, X_train_tab], y_train,
    validation_data=([X_test_img, X_test_tab], y_test),
    epochs=30, batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop]
)

# ---- STEP I: Evaluate ----
test_loss, test_acc = model.evaluate([X_test_img, X_test_tab], y_test)
print("Test Accuracy:", test_acc)

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Model Accuracy over Training')
plt.legend(); plt.grid(True); plt.show()

y_pred = model.predict([X_test_img, X_test_tab])
y_pred_classes = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred_classes, target_names=classes))

# ---- STEP J: Save model for later use (Streamlit) ----
model.save("crop_disease_model.h5")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Model and files saved successfully!")
# ============================================
# IMAGE-ONLY MODEL (for comparison)
# ============================================
print("\n\n--- Training image-only model for comparison ---\n")

img_only_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="img_only_input")
base_model2 = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model2.trainable = False

a = base_model2(img_only_input, training=False)
a = GlobalAveragePooling2D()(a)
a = Dense(64, activation='relu')(a)
a = Dense(32, activation='relu')(a)
a = Dropout(0.3)(a)
img_only_output = Dense(len(classes), activation='softmax')(a)

img_only_model = Model(inputs=img_only_input, outputs=img_only_output)
img_only_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
img_only_model.summary()

early_stop2 = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

img_only_history = img_only_model.fit(
    X_train_img, y_train,
    validation_data=(X_test_img, y_test),
    epochs=30,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop2]
)

img_only_loss, img_only_acc = img_only_model.evaluate(X_test_img, y_test)

# ============================================
# FINAL COMPARISON
# ============================================
print("\n\n========== FINAL COMPARISON ==========")
print(f"Image-only model accuracy:      {img_only_acc:.4f}")
print(f"Multimodal model accuracy:      {test_acc:.4f}")
print(f"Improvement from adding soil/rainfall: {(test_acc - img_only_acc)*100:.2f} percentage points")
print("=======================================")