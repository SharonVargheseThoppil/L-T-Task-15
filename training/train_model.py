import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


MODEL_PATH = "../api/cifar10_cnn_model.keras"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def build_model():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():

    print("=" * 60)
    print("CIFAR-10 DEEP LEARNING MODEL TRAINING")
    print("=" * 60)

    print("\nLoading CIFAR-10 dataset...")

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(f"Training images: {x_train.shape}")
    print(f"Testing images:  {x_test.shape}")

    # Normalize pixel values
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    print("\nBuilding CNN model...")

    model = build_model()

    model.summary()

    os.makedirs("../api", exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    print("\nStarting training...")

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    print("\nEvaluating model...")

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=1
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

    print("\nSaving model...")

    model.save(MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")

    print("\nClass names:")

    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{index}: {class_name}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()