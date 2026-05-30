import os
import pickle

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

DATASET_PATH = "data/plantvillage"

# Handle nested PlantVillage folder
if os.path.exists(os.path.join(DATASET_PATH, "PlantVillage")):
    DATASET_PATH = os.path.join(DATASET_PATH, "PlantVillage")

os.makedirs("models", exist_ok=True)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

print("\nDetected Classes:")
print(train_generator.class_indices)

with open("models/labels.pkl", "wb") as f:
    pickle.dump(train_generator.class_indices, f)

model = Sequential([
    Input(shape=(64,64,3)),

    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(
        train_generator.num_classes,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    restore_best_weights=True
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=3,
    callbacks=[early_stop]
)

model.save("models/cnn_model.h5")

print("\nModel saved successfully!")