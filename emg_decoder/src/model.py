import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout, 
    Flatten, BatchNormalization, Bidirectional,
    TimeDistributed, Input, concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os

def build_cnn_lstm_model(input_shape=(200, 8), num_classes=3, learning_rate=0.001):
    """
    Builds an enhanced CNN-LSTM model for EMG signal processing.
    
    Args:
        input_shape: Tuple of (time_steps, features)
        num_classes: Number of output classes
        learning_rate: Learning rate for the optimizer
    
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Parallel CNN branches for different temporal scales
    branch1 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    branch1 = BatchNormalization()(branch1)
    branch1 = MaxPooling1D(pool_size=2)(branch1)
    
    branch2 = Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    branch2 = BatchNormalization()(branch2)
    branch2 = MaxPooling1D(pool_size=2)(branch2)
    
    # Merge branches
    merged = concatenate([branch1, branch2])
    
    # Additional CNN layers
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(merged)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # Bidirectional LSTM for better temporal modeling
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = tf.keras.Model(inputs=inputs, outputs=x)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def save_model(model, model_path='models/emg_decoder.h5'):
    """Save the model to disk"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_saved_model(model_path='models/emg_decoder.h5'):
    """Load a saved model from disk"""
    if os.path.exists(model_path):
        return load_model(model_path)
    raise FileNotFoundError(f"No model found at {model_path}")

def convert_to_tflite(model, output_path='microcontroller/model.tflite'):
    """Convert the model to TensorFlow Lite format"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {output_path}")

def train_model(model, X_train, y_train, X_val, y_val, 
                batch_size=32, epochs=100, model_path='models/emg_decoder.h5'):
    """Train the model with callbacks and validation"""
    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
