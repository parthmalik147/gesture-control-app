# gesture_control_app/tests/test_config.py
import unittest
import os
import json
from unittest.mock import patch, mock_open

# Adjust import path for testing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.constants import DEFAULT_CONFIG, CONFIG_FILE_PATH, APP_DATA_BASE_DIR

class TestConfig(unittest.TestCase):

    def setUp(self):
        # Ensure a clean state before each test
        if os.path.exists(CONFIG_FILE_PATH):
            os.remove(CONFIG_FILE_PATH)
        # Reset the CONFIG global variable in the module
        config.CONFIG = {}
        # Ensure data directories exist for paths to be set correctly
        os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
        os.makedirs(os.path.join(APP_DATA_BASE_DIR, 'simple_gesture_data'), exist_ok=True)
        os.makedirs(os.path.join(APP_DATA_BASE_DIR, 'trained_models'), exist_ok=True)


    def tearDown(self):
        # Clean up after each test
        if os.path.exists(CONFIG_FILE_PATH):
            os.remove(CONFIG_FILE_PATH)
        # Optionally remove created data/model dirs if they were created specifically by tests
        if os.path.exists(APP_DATA_BASE_DIR):
            import shutil
            shutil.rmtree(APP_DATA_BASE_DIR)


    @patch('src.utils.console_utils.print_colored') # Mock print_colored to prevent console output during tests
    def test_load_config_no_file(self, mock_print):
        """Test that default config is loaded and saved if no config file exists."""
        config.load_config()
        self.assertIsNotNone(config.get_config())
        self.assertEqual(config.get_config()['num_samples_per_action'], DEFAULT_CONFIG['num_samples_per_action'])
        self.assertTrue(os.path.exists(CONFIG_FILE_PATH)) # Should have created the file

        # Verify the content of the created file
        with open(CONFIG_FILE_PATH, 'r') as f:
            loaded_json = json.load(f)
            self.assertEqual(loaded_json['camera_index'], DEFAULT_CONFIG['camera_index'])

    @patch('src.utils.console_utils.print_colored')
    def test_load_config_existing_valid_file(self, mock_print):
        """Test that config is loaded correctly from an existing valid file."""
        test_config_data = DEFAULT_CONFIG.copy()
        test_config_data['num_samples_per_action'] = 500
        test_config_data['camera_index'] = 1

        with open(CONFIG_FILE_PATH, 'w') as f:
            json.dump(test_config_data, f)

        config.load_config()
        loaded_config = config.get_config()
        self.assertEqual(loaded_config['num_samples_per_action'], 500)
        self.assertEqual(loaded_config['camera_index'], 1)
        # Ensure new default keys are merged in if they exist in DEFAULT_CONFIG but not in test_config_data
        self.assertTrue('show_fps' in loaded_config)


    @patch('src.utils.console_utils.print_colored')
    def test_load_config_invalid_json_file(self, mock_print):
        """Test that default config is used if existing file is invalid JSON."""
        with open(CONFIG_FILE_PATH, 'w') as f:
            f.write("this is not valid json {")

        config.load_config()
        self.assertEqual(config.get_config()['num_samples_per_action'], DEFAULT_CONFIG['num_samples_per_action'])
        mock_print.assert_any_call(self.assertRegex('Error reading config file'), config.COLOR_RED) # Check if error message was printed


    @patch('src.utils.console_utils.print_colored')
    def test_save_config(self, mock_print):
        """Test that config is saved correctly."""
        config.load_config() # Load default first
        current_config = config.get_config()
        current_config['num_samples_per_action'] = 250
        current_config['confidence_threshold'] = 0.95

        config.save_config()

        with open(CONFIG_FILE_PATH, 'r') as f:
            saved_json = json.load(f)

        self.assertEqual(saved_json['num_samples_per_action'], 250)
        self.assertEqual(saved_json['confidence_threshold'], 0.95)
        # Ensure derived paths are NOT saved
        self.assertNotIn('data_dir', saved_json)
        self.assertNotIn('model_path', saved_json)

    @patch('src.utils.console_utils.print_colored')
    def test_get_config(self, mock_print):
        """Test that get_config returns the current configuration."""
        config.load_config()
        retrieved_config = config.get_config()
        self.assertIs(retrieved_config, config.CONFIG) # Should return the same object

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)