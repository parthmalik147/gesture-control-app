# Gesture Control Application

A console-based application for controlling your computer using hand gestures, built with MediaPipe and TensorFlow. This project demonstrates real-time hand tracking, gesture data collection, model training, and execution of system commands (e.g., brightness, volume, scrolling, media control).

## Features

* **Real-time Hand Tracking**: Utilizes MediaPipe to detect and track hand landmarks.
* **Custom Gesture Data Collection**: Collects hand landmark data for user-defined actions.
* **Gesture Model Training**: Trains a neural network using TensorFlow/Keras to classify custom gestures.
* **System Controls**:
    * Adjust screen brightness (Windows/macOS/Linux via `screen_brightness_control`).
    * Adjust system volume (Windows/macOS/Linux via `pyautogui`).
    * Scroll windows (Windows/macOS/Linux via `pyautogui`).
    * Media playback control (play/pause, next/previous track via `pyautogui`).
    * Open web browser (cross-platform).
    * Minimize active window (cross-platform).
* **Configurable Settings**: Customize sample count, confidence thresholds, camera index, and display options.
* **Cross-Platform Compatibility**: Designed to work on Windows, macOS, and Linux where libraries support.
* **Console-based UI**: Interactive menu for easy navigation.
* **Data Reset Functionality**: Option to delete all collected data, trained models, and reset settings.

## Project Structure
gesture_control_app/
├── .ges_con_app_venv/         # Python Virtual Environment
├── src/                       # Source Code
│   ├── main.py                # Main application entry point
│   ├── config.py              # Configuration loading/saving
│   ├── constants.py           # Global constants (paths, colors, defaults)
│   ├── utils/                 # Utility functions
│   │   ├── camera_utils.py    # Camera interaction, landmark extraction/drawing
│   │   ├── console_utils.py   # Console output formatting
│   │   └── file_utils.py      # Generic file operations (if needed)
│   ├── data_collection/       # Logic for collecting gesture data
│   │   └── collector.py
│   ├── model_training/        # Logic for training the gesture recognition model
│   │   └── trainer.py
│   ├── gesture_recognition/   # Real-time gesture prediction and action handling
│   │   ├── recognizer.py
│   │   └── action_handler.py  # Maps recognized gestures to system controls
│   └── system_controls/       # Platform-specific system control functions
│       ├── brightness.py
│       ├── volume.py
│       ├── scroll.py
│       ├── media.py
│       └── window_management.py
├── data/                      # Stores collected gesture data and trained models
│   ├── simple_gesture_data/
│   └── trained_models/
├── config/                    # Application configuration files
│   └── app_config.json
├── logs/                      # Application log files
├── tests/                     # Unit and Integration Tests
├── .gitignore                 # Files to ignore in Git
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── setup.py (Optional)        # For packaging/distribution

## Installation

### Prerequisites

* Python 3.8+
* A webcam

### Steps

1.  **Clone the repository (or create the project structure manually):**

    ```bash
    git clone [https://github.com/parthmalik147/gesture-control-app.git](https://github.com/parthmalik147/gesture-control-app.git)
    cd gesture_control_app
    ```
    (If you are setting up manually, create the directories and empty files as per the "Project Structure" section above.)

2.  **Create a Python Virtual Environment:**

    It's highly recommended to use a virtual environment to manage dependencies.

    ```bash
    python3 -m venv .ges_con_app_venv
    ```
    * On some systems, `python` might be aliased to Python 3, so `python -m venv .ges_con_app_venv` might work instead.

3.  **Activate the Virtual Environment:**

    * **On macOS/Linux:**
        ```bash
        source .ges_con_app_venv/bin/activate
        ```
    * **On Windows (Command Prompt):**
        ```bash
        .ges_con_app_venv\Scripts\activate.bat
        ```
    * **On Windows (PowerShell):**
        ```powershell
        .ges_con_app_venv\Scripts\Activate.ps1
        ```
    You should see `(.ges_con_app_venv)` prefixing your terminal prompt, indicating the virtual environment is active.

4.  **Install Dependencies:**

    Install all required Python packages listed in `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Activate your virtual environment** (if not already active).
2.  **Run the main application:**

    ```bash
    python src/main.py
    ```

3.  **Follow the on-screen menu:**

    * **[n] Add New Gestures & Train Model**: Guides you through collecting data for new gestures and then trains a machine learning model based on that data.
    * **[a] Activate Real-time Gesture Control**: Uses the trained model to recognize gestures from your webcam feed and execute corresponding system actions.
    * **[s] Settings**: Allows you to adjust various application parameters like the number of samples for training, confidence threshold, camera index, etc.
    * **[r] Reset All Data**: **CAUTION!** This option will permanently delete all collected gesture data, trained models, and reset your application settings to default. Use with care.
    * **[e] Exit Application**: Closes the application.

## Troubleshooting

* **`screen_brightness_control` issues (Windows/Linux)**: Ensure you have the necessary system permissions or dependencies for `screen_brightness_control` to work. On some Linux systems, you might need `xrandr`.
* **`pyautogui` issues**: For `pyautogui` to work reliably, ensure your terminal has accessibility/permissions granted (especially on macOS).
* **No Camera Detected**: Check if your camera is properly connected and not in use by another application. Try changing the `Camera Index` in settings.
* **`tensorflow` warnings**: You might see warnings related to TensorFlow, especially if your system doesn't have a compatible GPU setup. These can often be ignored for CPU-only training/inference.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is open-source and available under the [MIT License](LICENSE.txt).