import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# 1. Dataset ka path set karein (Yahan apna sahi folder path likhein)
# Agar cell_images folder usi jagah hai jahan script hai, to ye thik hai:
base_dir = 'cell_images'

# 2. Data ko Normalize aur Split karna (80% Train, 20% Test)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    base_dir,
    target_size=(64, 64), # Speed ke liye size chota rakha hai
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    base_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 3. AI Model ka Structure (CNN)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') # Binary: 0=Infected, 1=Uninfected
])

# 4. Training shuru karein
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training shuru ho rahi hai... Thora sabar karein.")
model.fit(train_data, validation_data=val_data, epochs=3) # 3 rounds kafi hain test ke liye

# 5. Model ko Save karein
model.save('malaria_model.h5')
print("Mubarak ho! 'malaria_model.h5' file ban gayi hai.")
