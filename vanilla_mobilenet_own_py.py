from tensorflow import keras 
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications.mobilenet import preprocess_input
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.applications.mobilenet import MobileNet
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import os


datagen = ImageDataGenerator()

Path = "/home/atik666/Documents/KD/knowledge-distillation-keras/data/"
train_generator = datagen.flow_from_directory(
    directory=Path+"/train/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

val_generator = datagen.flow_from_directory(
    directory=Path+"/val/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
counter = len(Counter(train_generator.classes))

model = MobileNet(weights=None, classes=counter,classifier_activation="softmax")
model.count_params()

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True), 
    loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy']
)

model.fit(
    train_generator, 
    steps_per_epoch=400, epochs=30, verbose=1,
    callbacks=[
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, epsilon=0.007),
        EarlyStopping(monitor='val_accuracy', patience=4, min_delta=0.01)
    ],
    validation_data=val_generator, validation_steps=80, workers=4
)




