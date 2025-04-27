import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
import os

# Define dataset path
data_dir = 'dataset/train'  # Change to actual dataset path

# Image dimensions and batch size
img_height, img_width = 224, 224
batch_size = 32

# Data Augmentation and Image Loading
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',  # Labels are integers, not one-hot
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',  # Labels are integers
    subset='validation'
)

# Load Pretrained Feature Extractor (if used in original model)
feature_extractor_layer = keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
feature_extractor_layer.trainable = False  # Freeze feature extractor

# Define Model Architecture
model = keras.Sequential([
    feature_extractor_layer,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(9, activation='softmax')  # 9 classes in original model
])

# Compile the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Model
epochs = 5
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

# Save the Trained Model
os.makedirs("model", exist_ok=True)
model.save('model/skin_disease_model.h5')

print(" Model trained and saved successfully!")
