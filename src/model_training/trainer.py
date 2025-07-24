# src/model_training/trainer.py
import numpy as np
import os
import json
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from src.utils.console_utils import clear_screen, print_colored, progress_bar
from src.config import get_config
from src.constants import COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_CYAN, COLOR_BOLD, ACTIVE_ACTIONS_FILE_PATH

def train_model():
    """
    Loads collected gesture data, trains a neural network model,
    and saves the trained model and action mappings.
    """
    config = get_config()
    data_dir = config['data_dir']
    model_path = config['model_path']
    actions = config['actions']

    clear_screen()
    print_colored("--- Training Gesture Recognition Model ---", COLOR_CYAN + ";" + COLOR_BOLD)

    if not os.path.exists(data_dir) or not any(os.listdir(data_dir)):
        print_colored("No gesture data found. Please collect data first.", COLOR_RED)
        time.sleep(2)
        return

    # Load data
    print_colored("Loading gesture data...", COLOR_BLUE)
    X, y = [], []
    label_map = {label: num for num, label in enumerate(actions)}

    total_files = sum([len(files) for r, d, files in os.walk(data_dir)])
    files_processed = 0

    for action in actions:
        action_path = os.path.join(data_dir, action)
        if not os.path.exists(action_path):
            print_colored(f"Warning: No data found for action '{action}'. Skipping.", COLOR_YELLOW)
            continue
        for sample_file in os.listdir(action_path):
            if sample_file.endswith('.npy'):
                sample_data = np.load(os.path.join(action_path, sample_file))
                X.append(sample_data)
                y.append(label_map[action])
                files_processed += 1
                progress_bar(files_processed, total_files, title="Loading Data")

    if not X:
        print_colored("No data loaded. Training aborted.", COLOR_RED)
        time.sleep(2)
        return

    X = np.array(X)
    y = to_categorical(np.array(y)).astype(int)

    # Split data
    print_colored("\nSplitting data into training and testing sets...", COLOR_BLUE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    print_colored("Building neural network model...", COLOR_BLUE)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(y.shape[1], activation='softmax') # Output layer for classification
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train the model
    print_colored("Training model...", COLOR_BLUE)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=100, # Max epochs
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model
    print_colored("\nEvaluating model...", COLOR_BLUE)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print_colored(f"Test Loss: {loss:.4f}", COLOR_CYAN)
    print_colored(f"Test Accuracy: {accuracy:.4f}", COLOR_CYAN)

    # Save the model
    print_colored(f"Saving model to {model_path}...", COLOR_GREEN)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print_colored("Model saved successfully!", COLOR_GREEN)

    # Save the mapping of actions to their numerical labels
    print_colored(f"Saving active actions to {ACTIVE_ACTIONS_FILE_PATH}...", COLOR_GREEN)
    with open(ACTIVE_ACTIONS_FILE_PATH, 'w') as f:
        json.dump(actions, f, indent=4) # Save the list of action labels
    print_colored("Active actions saved successfully!", COLOR_GREEN)

    print_colored("\nModel training complete!", COLOR_GREEN + ";" + COLOR_BOLD)
    time.sleep(2)