"""
Model Training Script for StrikeSense Boxing AI
Trains an LSTM classifier on collected pose data.
"""

import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from config import *
from utils import create_sliding_window

def load_all_data():
    """Load all session data from raw data directory"""
    print("Loading training data...")
    
    all_sequences = []
    all_labels = []
    
    # Find all session files
    session_files = glob.glob(f"{RAW_DATA_DIR}/session_*.npz")
    
    if len(session_files) == 0:
        print(f"Error: No training data found in {RAW_DATA_DIR}/")
        print("Please run data_collector.py first to collect training data.")
        return None, None
    
    print(f"Found {len(session_files)} session files")
    
    for file in session_files:
        data = np.load(file, allow_pickle=True)
        sequences = data['sequences']
        labels = data['labels']
        
        for seq, label in zip(sequences, labels):
            all_sequences.append(np.array(seq))
            all_labels.append(label)
    
    print(f"Loaded {len(all_sequences)} sequences")
    return all_sequences, all_labels

def prepare_training_data(sequences, labels):
    """
    Convert variable-length sequences into fixed-length windows
    """
    print("Preparing training windows...")
    
    X_windows = []
    y_windows = []
    
    for seq, label in zip(sequences, labels):
        seq_array = np.array(seq)
        
        # Skip sequences that are too short
        if len(seq_array) < WINDOW_SIZE:
            continue
        
        # Create sliding windows
        windows = create_sliding_window(seq_array, WINDOW_SIZE, STRIDE)
        
        # Each window gets the same label
        for window in windows:
            X_windows.append(window)
            y_windows.append(label)
    
    X = np.array(X_windows)
    y = np.array(y_windows)
    
    print(f"Created {len(X)} training windows")
    print(f"Input shape: {X.shape}")
    
    return X, y

def build_model():
    """Build LSTM classification model"""
    print("Building model...")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=INPUT_SHAPE),
        
        # LSTM layers
        layers.LSTM(LSTM_UNITS, return_sequences=True),
        layers.Dropout(DROPOUT_RATE),
        
        layers.LSTM(LSTM_UNITS // 2, return_sequences=False),
        layers.Dropout(DROPOUT_RATE),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dropout(DROPOUT_RATE),
        
        # Output layer
        layers.Dense(len(PUNCH_CLASSES), activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    return model

def plot_training_history(history):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/training_history.png")
    print(f"Training plots saved to {MODEL_DIR}/training_history.png")
    plt.show()

def main():
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load data
    sequences, labels = load_all_data()
    if sequences is None:
        return
    
    # Print class distribution
    print("\nClass distribution:")
    for i, class_name in enumerate(PUNCH_CLASSES):
        count = sum(1 for label in labels if label == i)
        print(f"  {class_name}: {count} sequences")
    
    # Prepare training data
    X, y = prepare_training_data(sequences, labels)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Build model
    model = build_model()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n=== Final Evaluation ===")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Save model
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    # Plot training history
    plot_training_history(history)
    
    # Class-wise accuracy
    print("\n=== Per-Class Performance ===")
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    for i, class_name in enumerate(PUNCH_CLASSES):
        mask = y_val == i
        if mask.sum() > 0:
            class_acc = (y_pred_classes[mask] == i).sum() / mask.sum()
            print(f"{class_name}: {class_acc:.4f}")

if __name__ == "__main__":
    main()