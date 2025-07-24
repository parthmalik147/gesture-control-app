# src/main.py
import os
import shutil
import sys
import time

# Import functions/constants from our modularized project
from src.config import load_config, save_config, get_config
from src.constants import (
    COLOR_BLUE, COLOR_BOLD, COLOR_CYAN, COLOR_GREEN, COLOR_MAGENTA,
    COLOR_RED, COLOR_WHITE, COLOR_YELLOW, COLOR_RESET,
    APP_DATA_BASE_DIR, CONFIG_FILE_PATH, ACTIVE_ACTIONS_FILE_PATH
)
from src.utils.console_utils import clear_screen, print_colored
from src.utils.camera_utils import get_available_cameras
from src.data_collection.collector import start_gathering
from src.model_training.trainer import train_model
from src.gesture_recognition.recognizer import activate_gesture_control


def settings_menu():
    """Provides a menu to view and change application settings."""
    config = get_config()
    clear_screen()
    print_colored("--- APPLICATION SETTINGS ---", COLOR_YELLOW + ";" + COLOR_BOLD)
    print(f"1. Number of samples per action: {config['num_samples_per_action']}")
    print(f"2. Prediction Confidence Threshold: {config['confidence_threshold']:.2f}")
    print(f"3. Gesture Cooldown Seconds: {config['cooldown_seconds']:.1f}")
    print(f"4. Action Message Display Duration: {config['display_duration_seconds']:.1f}s")
    print(f"5. Camera Index: {config['camera_index']} (Available: {', '.join(map(str, get_available_cameras()))})")
    print(f"6. Show FPS in Live Mode: {'Yes' if config['show_fps'] else 'No'}")
    print(f"7. Show Confidence in Live Mode: {'Yes' if config['show_confidence'] else 'No'}")
    print(f"8. Show Landmarks in Live Mode: {'Yes' if config['show_landmarks_in_live_mode'] else 'No'}")
    print("\n[b] Back to Main Menu")

    while True:
        sys.stdout.write(f"\033[{COLOR_WHITE};{COLOR_BOLD}mEnter setting number to change, or 'b' to go back: \033[0m")
        sys.stdout.flush()
        choice = input().strip().lower()

        if choice == 'b':
            save_config() # Save changes made in settings
            print_colored("Settings saved and returning to main menu.", COLOR_GREEN)
            time.sleep(2)
            return
        elif choice == '1':
            try:
                new_val = int(input(f"Enter new number of samples per action (current: {config['num_samples_per_action']}): "))
                if new_val > 0:
                    config['num_samples_per_action'] = new_val
                    print_colored("Updated.", COLOR_GREEN)
                else:
                    print_colored("Value must be positive.", COLOR_RED)
            except ValueError:
                print_colored("Invalid input. Please enter a number.", COLOR_RED)
        elif choice == '2':
            try:
                new_val = float(input(f"Enter new confidence threshold (0.0-1.0, current: {config['confidence_threshold']:.2f}): "))
                if 0.0 <= new_val <= 1.0:
                    config['confidence_threshold'] = new_val
                    print_colored("Updated.", COLOR_GREEN)
                else:
                    print_colored("Value must be between 0.0 and 1.0.", COLOR_RED)
            except ValueError:
                print_colored("Invalid input. Please enter a number.", COLOR_RED)
        elif choice == '3':
            try:
                new_val = float(input(f"Enter new cooldown seconds (current: {config['cooldown_seconds']:.1f}): "))
                if new_val >= 0:
                    config['cooldown_seconds'] = new_val
                    print_colored("Updated.", COLOR_GREEN)
                else:
                    print_colored("Value must be non-negative.", COLOR_RED)
            except ValueError:
                print_colored("Invalid input. Please enter a number.", COLOR_RED)
        elif choice == '4':
            try:
                new_val = float(input(f"Enter new action message display duration in seconds (current: {config['display_duration_seconds']:.1f}): "))
                if new_val >= 0.5:
                    config['display_duration_seconds'] = new_val
                    print_colored("Updated.", COLOR_GREEN)
                else:
                    print_colored("Value must be at least 0.5 seconds.", COLOR_RED)
            except ValueError:
                print_colored("Invalid input. Please enter a number.", COLOR_RED)
        elif choice == '5':
            available_cams = get_available_cameras()
            if not available_cams:
                print_colored("No cameras detected.", COLOR_RED)
                time.sleep(1)
                continue
            print(f"Available cameras: {', '.join(map(str, available_cams))}")
            try:
                new_val = int(input(f"Enter new camera index (current: {config['camera_index']}): "))
                if new_val in available_cams:
                    config['camera_index'] = new_val
                    print_colored("Updated.", COLOR_GREEN)
                else:
                    print_colored("Invalid camera index.", COLOR_RED)
            except ValueError:
                print_colored("Invalid input. Please enter a number.", COLOR_RED)
        elif choice == '6':
            config['show_fps'] = not config['show_fps']
            print_colored(f"Show FPS set to: {'Yes' if config['show_fps'] else 'No'}. Updated.", COLOR_GREEN)
        elif choice == '7':
            config['show_confidence'] = not config['show_confidence']
            print_colored(f"Show Confidence set to: {'Yes' if config['show_confidence'] else 'No'}. Updated.", COLOR_GREEN)
        elif choice == '8':
            config['show_landmarks_in_live_mode'] = not config['show_landmarks_in_live_mode']
            print_colored(f"Show Landmarks in Live Mode set to: {'Yes' if config['show_landmarks_in_live_mode'] else 'No'}. Updated.", COLOR_GREEN)
        else:
            print_colored("Invalid choice.", COLOR_RED)
        time.sleep(0.5) # Short pause before redrawing menu
        clear_screen()
        print_colored("--- APPLICATION SETTINGS ---", COLOR_YELLOW + ";" + COLOR_BOLD)
        print(f"1. Number of samples per action: {config['num_samples_per_action']}")
        print(f"2. Prediction Confidence Threshold: {config['confidence_threshold']:.2f}")
        print(f"3. Gesture Cooldown Seconds: {config['cooldown_seconds']:.1f}")
        print(f"4. Action Message Display Duration: {config['display_duration_seconds']:.1f}s")
        print(f"5. Camera Index: {config['camera_index']} (Available: {', '.join(map(str, get_available_cameras()))})")
        print(f"6. Show FPS in Live Mode: {'Yes' if config['show_fps'] else 'No'}")
        print(f"7. Show Confidence in Live Mode: {'Yes' if config['show_confidence'] else 'No'}")
        print(f"8. Show Landmarks in Live Mode: {'Yes' if config['show_landmarks_in_live_mode'] else 'No'}")
        print("\n[b] Back to Main Menu")


def reset_application_data():
    """
    Deletes all collected gesture data, trained models, and resets configuration
    to default settings. Requires user confirmation.
    """
    clear_screen()
    print_colored("--- RESET ALL APPLICATION DATA ---", COLOR_RED + ";" + COLOR_BOLD)
    print_colored("WARNING: This will permanently delete:", COLOR_RED)
    print_colored(f"  - Trained Model: {get_config()['model_path']}", COLOR_RED)
    print_colored(f"  - All Gesture Data: {get_config()['data_dir']}", COLOR_RED)
    print_colored(f"  - Active Actions File: {ACTIVE_ACTIONS_FILE_PATH}", COLOR_RED)
    print_colored(f"  - Your Settings File: {CONFIG_FILE_PATH}", COLOR_RED)
    print_colored("\nThis action CANNOT be undone.", COLOR_RED + ";" + COLOR_BOLD)

    sys.stdout.write(f"\033[{COLOR_YELLOW};{COLOR_BOLD}mAre you sure you want to proceed? (yes/no): \033[0m")
    sys.stdout.flush()
    confirmation = input().strip().lower()

    if confirmation == 'yes':
        # Delete model file
        model_path = get_config()['model_path']
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                print_colored(f"Deleted model: {model_path}", COLOR_GREEN)
            except OSError as e:
                print_colored(f"Error deleting model file: {e}", COLOR_RED)
        else:
            print_colored("Model file not found (already deleted or never trained).", COLOR_YELLOW)

        # Delete gesture data directory
        data_dir = get_config()['data_dir']
        if os.path.exists(data_dir):
            try:
                shutil.rmtree(data_dir)
                print_colored(f"Deleted gesture data directory: {data_dir}", COLOR_GREEN)
            except OSError as e:
                print_colored(f"Error deleting data directory: {e}", COLOR_RED)
        else:
            print_colored("Gesture data directory not found (already deleted or no data collected).", COLOR_YELLOW)

        # Delete active actions file
        if os.path.exists(ACTIVE_ACTIONS_FILE_PATH):
            try:
                os.remove(ACTIVE_ACTIONS_FILE_PATH)
                print_colored(f"Deleted active actions file: {ACTIVE_ACTIONS_FILE_PATH}", COLOR_GREEN)
            except OSError as e:
                print_colored(f"Error deleting active actions file: {e}", COLOR_RED)
        else:
            print_colored("Active actions file not found.", COLOR_YELLOW)

        # Reset settings to default by deleting current config and reloading
        if os.path.exists(CONFIG_FILE_PATH):
            try:
                os.remove(CONFIG_FILE_PATH)
                print_colored(f"Deleted settings file: {CONFIG_FILE_PATH}", COLOR_GREEN)
            except OSError as e:
                print_colored(f"Error deleting settings file: {e}", COLOR_RED)
        else:
            print_colored("Settings file not found.", COLOR_YELLOW)

        print_colored("\nApplication data has been reset to default.", COLOR_GREEN + ";" + COLOR_BOLD)
        print_colored("Configuration will be reloaded.", COLOR_YELLOW)
        time.sleep(3)
    else:
        print_colored("Reset cancelled.", COLOR_CYAN)
        time.sleep(1.5)


def main():
    """Main function to run the gesture control application."""
    load_config() # Load configuration at startup

    choice = ''
    while choice != 'e':
        clear_screen()
        print_colored("==============================================", COLOR_BLUE + ";" + COLOR_BOLD)
        print_colored("     WELCOME TO GESTURE CONTROL APPLICATION!    ", COLOR_CYAN + ";" + COLOR_BOLD)
        print_colored("==============================================", COLOR_BLUE + ";" + COLOR_BOLD)
        print_colored("\n[n] Add New Gestures & Train Model", COLOR_GREEN)
        print_colored("[a] Activate Real-time Gesture Control", COLOR_YELLOW)
        print_colored("[s] Settings", COLOR_MAGENTA)
        print_colored("[r] Reset All Data (Caution: Deletes all trained models and settings!)", COLOR_RED)
        print_colored("[e] Exit Application", COLOR_RED)
        print_colored("----------------------------------------------", COLOR_BLUE)

        sys.stdout.write(f"\033[{COLOR_WHITE};{COLOR_BOLD}mEnter your choice (n/a/s/r/e): \033[0m")
        sys.stdout.flush()
        choice = input().strip().lower()

        if choice == 'n':
            start_gathering()
            train_model()
        elif choice == 'a':
            activate_gesture_control()
        elif choice == 's':
            settings_menu()
        elif choice == 'r':
            reset_application_data()
            load_config() # Reload config after reset
        elif choice == 'e':
            print_colored("\nExiting application. Goodbye!", COLOR_MAGENTA)
            time.sleep(1)
            break
        else:
            print_colored("Invalid choice. Please enter 'n', 'a', 's', 'r', or 'e'.", COLOR_RED)
            time.sleep(1.5)

    print_colored("Application closed.", COLOR_MAGENTA + ";" + COLOR_BOLD)

if __name__ == "__main__":
    main()