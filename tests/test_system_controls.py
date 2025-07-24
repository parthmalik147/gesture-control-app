# gesture_control_app/tests/test_system_controls.py
import unittest
import sys
from unittest.mock import patch, MagicMock

# Adjust import path for testing
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the external libraries that might not be available or should not execute during tests
sys.modules['screen_brightness_control'] = MagicMock()
sys.modules['pyautogui'] = MagicMock()

from src.system_controls.brightness import adjust_brightness
from src.system_controls.volume import adjust_volume
from src.system_controls.scroll import scroll_window
from src.system_controls.media import play_pause_media, next_track, prev_track
from src.system_controls.window_management import open_web_browser, minimize_active_window

class TestSystemControls(unittest.TestCase):

    def setUp(self):
        # Reset mocks before each test
        if 'screen_brightness_control' in sys.modules:
            sys.modules['screen_brightness_control'].reset_mock()
            sys.modules['screen_brightness_control'].get_brightness.return_value = [50] # Default brightness
            sys.modules['screen_brightness_control'].ScreenBrightnessError = type('ScreenBrightnessError', (Exception,), {}) # Mock the exception class
        if 'pyautogui' in sys.modules:
            sys.modules['pyautogui'].reset_mock()

    @patch('src.utils.console_utils.print_colored')
    def test_adjust_brightness_up(self, mock_print):
        with patch('src.system_controls.brightness.sbc') as mock_sbc:
            mock_sbc.get_brightness.return_value = [50]
            new_val = adjust_brightness(10, 50)
            mock_sbc.set_brightness.assert_called_once_with(60)
            self.assertEqual(new_val, 60)

    @patch('src.utils.console_utils.print_colored')
    def test_adjust_brightness_down(self, mock_print):
        with patch('src.system_controls.brightness.sbc') as mock_sbc:
            mock_sbc.get_brightness.return_value = [50]
            new_val = adjust_brightness(-20, 50)
            mock_sbc.set_brightness.assert_called_once_with(30)
            self.assertEqual(new_val, 30)

    @patch('src.utils.console_utils.print_colored')
    def test_adjust_brightness_min_max(self, mock_print):
        with patch('src.system_controls.brightness.sbc') as mock_sbc:
            mock_sbc.get_brightness.return_value = [95]
            new_val = adjust_brightness(10, 95) # Should cap at 100
            mock_sbc.set_brightness.assert_called_once_with(100)
            self.assertEqual(new_val, 100)

            mock_sbc.reset_mock()
            mock_sbc.get_brightness.return_value = [5]
            new_val = adjust_brightness(-10, 5) # Should cap at 0
            mock_sbc.set_brightness.assert_called_once_with(0)
            self.assertEqual(new_val, 0)

    @patch('src.utils.console_utils.print_colored')
    def test_adjust_volume_up(self, mock_print):
        with patch('src.system_controls.volume.pyautogui') as mock_pyautogui:
            result = adjust_volume("up")
            mock_pyautogui.press.assert_called_once_with('volumeup')
            self.assertTrue(result)

    @patch('src.utils.console_utils.print_colored')
    def test_adjust_volume_down(self, mock_print):
        with patch('src.system_controls.volume.pyautogui') as mock_pyautogui:
            result = adjust_volume("down")
            mock_pyautogui.press.assert_called_once_with('volumedown')
            self.assertTrue(result)

    @patch('src.utils.console_utils.print_colored')
    def test_scroll_window_up(self, mock_print):
        with patch('src.system_controls.scroll.pyautogui') as mock_pyautogui:
            result = scroll_window("up")
            mock_pyautogui.scroll.assert_called_once_with(100)
            self.assertTrue(result)

    @patch('src.utils.console_utils.print_colored')
    def test_scroll_window_down(self, mock_print):
        with patch('src.system_controls.scroll.pyautogui') as mock_pyautogui:
            result = scroll_window("down", amount=50)
            mock_pyautogui.scroll.assert_called_once_with(-50)
            self.assertTrue(result)

    @patch('src.utils.console_utils.print_colored')
    def test_play_pause_media(self, mock_print):
        with patch('src.system_controls.media.pyautogui') as mock_pyautogui:
            result = play_pause_media()
            mock_pyautogui.press.assert_called_once_with('playpause')
            self.assertTrue(result)

    @patch('src.utils.console_utils.print_colored')
    def test_next_track(self, mock_print):
        with patch('src.system_controls.media.pyautogui') as mock_pyautogui:
            result = next_track()
            mock_pyautogui.press.assert_called_once_with('nexttrack')
            self.assertTrue(result)

    @patch('src.utils.console_colored')
    def test_open_web_browser(self, mock_print):
        with patch('src.system_controls.window_management.webbrowser.open') as mock_webbrowser_open:
            result = open_web_browser("http://example.com")
            mock_webbrowser_open.assert_called_once_with("http://example.com")
            self.assertTrue(result)

    @patch('src.utils.console_utils.print_colored')
    def test_minimize_active_window_windows(self, mock_print):
        with patch('src.system_controls.window_management.pyautogui') as mock_pyautogui:
            with patch('sys.platform', 'win32'):
                result = minimize_active_window()
                mock_pyautogui.hotkey.assert_called_once_with('win', 'down')
                self.assertTrue(result)

    @patch('src.utils.console_utils.print_colored')
    def test_minimize_active_window_macos(self, mock_print):
        with patch('src.system_controls.window_management.pyautogui') as mock_pyautogui:
            with patch('sys.platform', 'darwin'):
                result = minimize_active_window()
                mock_pyautogui.hotkey.assert_called_once_with('command', 'h')
                self.assertTrue(result)

    @patch('src.utils.console_utils.print_colored')
    def test_minimize_active_window_linux(self, mock_print):
        with patch('src.system_controls.window_management.pyautogui') as mock_pyautogui:
            with patch('sys.platform', 'linux'):
                result = minimize_active_window()
                mock_pyautogui.hotkey.assert_any_call('alt', 'space')
                mock_pyautogui.press.assert_called_once_with('n')
                self.assertTrue(result)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)