import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_cnn_model(input_shape, num_classes, learning_rate=0.001, dropout_rate=0.2):
    """Builds and compiles the EfficientNetB0 model."""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Start with frozen base

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Resizing(224, 224),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_model(model, x_train, y_train, x_val, y_val, batch_size=32, epochs=25, finetune=False, unfreeze_layers=0):
    """Trains or fine-tunes the CNN model."""
    if finetune:
        # Unfreeze the top N layers of the base model
        model.layers[2].trainable = True # The base_model layer
        for layer in model.layers[2].layers[:-unfreeze_layers]:
            layer.trainable = False
        # Re-compile with a lower learning rate for fine-tuning
        optimizer = Adam(learning_rate=1e-4, clipnorm=1.0, weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Fine-tuning enabled. Unfroze the top {unfreeze_layers} layers.")

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1 if finetune else 0.0, # Add zoom for fine-tuning as in the report
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    if finetune:
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6, verbose=1))
    
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history
