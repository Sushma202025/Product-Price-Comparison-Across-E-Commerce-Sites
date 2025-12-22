import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import json

print("TRAINING STARTED")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "product_model.keras")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "class_indices.json")

print("DATASET EXISTS:", os.path.exists(DATASET_DIR))
os.makedirs(MODEL_DIR, exist_ok=True)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224,224),
    batch_size=8,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224,224),
    batch_size=8,
    class_mode="categorical",
    subset="validation"
)

with open(CLASS_INDEX_PATH, "w") as f:
    json.dump(train_gen.class_indices, f)

print("CLASS INDICES SAVED")

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
out = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(base.input, out)
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_gen, validation_data=val_gen, epochs=3)

print("SAVING MODEL TO:", MODEL_PATH)
model.save(MODEL_PATH)
print("MODEL SAVED")
