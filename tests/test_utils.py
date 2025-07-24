# gesture_control_app/tests/test_utils.py
import unittest
import os
import sys
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

# Adjust import path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock external libraries to prevent actual camera access or display during tests
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.solutions.hands'] = MagicMock()
sys.modules['mediapipe.solutions.drawing_utils'] = MagicMock()
sys.modules['cv2'] = MagicMock() # Mock cv2 itself as well

from src.utils.camera_utils import get_available_cameras, extract_hand_landmarks, draw_hand_landmarks
from src.utils.console_utils import clear_screen, print_colored, progress_bar
from src.constants import COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_CYAN

class TestCameraUtils(unittest.TestCase):

    @patch('src.utils.camera_utils.cv2.VideoCapture')
    @patch('src.utils.console_utils.print_colored')
    def test_get_available_cameras(self, mock_print_colored, MockVideoCapture):
        # Mocking scenarios:
        # 0: True, 1: False, 2: True, others False
        def side_effect(index):
            mock_cap = MagicMock()
            if index in [0, 2]:
                mock_cap.isOpened.return_value = True
            else:
                mock_cap.isOpened.return_value = False
            return mock_cap

        MockVideoCapture.side_effect = side_effect

        available_cams = get_available_cameras()
        self.assertEqual(available_cams, [0, 2])
        mock_print_colored.assert_not_called() # Should not print warning if cameras found

        # Test no cameras found
        MockVideoCapture.side_effect = lambda x: MagicMock(isOpened=MagicMock(return_value=False))
        available_cams = get_available_cameras()
        self.assertEqual(available_cams, [])
        mock_print_colored.assert_called_with("Warning: No cameras found. Please ensure a webcam is connected.", COLOR_YELLOW)


    def test_extract_hand_landmarks_with_hand(self):
        # Create a mock results object with right_hand_landmarks
        mock_results = MagicMock()
        mock_results.right_hand_landmarks = MagicMock()
        
        # Create 21 mock landmarks with distinct values
        mock_landmarks = []
        for i in range(21):
            lm = MagicMock()
            # Use non-zero values to test normalization
            lm.x = float(i * 0.1)
            lm.y = float(i * 0.2)
            lm.z = float(i * 0.3)
            mock_landmarks.append(lm)
        
        mock_results.right_hand_landmarks.landmark = mock_landmarks

        extracted = extract_hand_landmarks(mock_results)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted.shape, (63,)) # 21 landmarks * 3 coords (x,y,z)

        # Verify normalization: the first landmark (wrist) should be (0,0,0) relative
        # (after normalization, the first 3 values in the flattened array)
        self.assertAlmostEqual(extracted[0], 0.0)
        self.assertAlmostEqual(extracted[1], 0.0)
        self.assertAlmostEqual(extracted[2], 0.0)

        # Check a few other normalized points
        # For landmark 1 (index 3,4,5 in flattened array), its original values were 0.1, 0.2, 0.3
        # Wrist was 0.0, 0.0, 0.0. So it should be the same.
        self.assertAlmostEqual(extracted[3], 0.1)
        self.assertAlmostEqual(extracted[4], 0.2)
        self.assertAlmostEqual(extracted[5], 0.3)


    def test_extract_hand_landmarks_no_hand(self):
        mock_results = MagicMock()
        mock_results.right_hand_landmarks = None
        extracted = extract_hand_landmarks(mock_results)
        self.assertIsNone(extracted)

    @patch('src.utils.camera_utils.mp_drawing.draw_landmarks')
    def test_draw_hand_landmarks(self, mock_draw_landmarks):
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_results = MagicMock()
        mock_results.right_hand_landmarks = MagicMock() # Simulate hand detected

        # Just check if the drawing function is called
        drawn_image = draw_hand_landmarks(mock_image.copy(), mock_results)
        mock_draw_landmarks.assert_called_once()
        self.assertTrue(np.array_equal(drawn_image, mock_image)) # Image is modified in-place by draw_landmarks (mocked)

    @patch('src.utils.camera_utils.mp_drawing.draw_landmarks')
    def test_draw_hand_landmarks_no_hand(self, mock_draw_landmarks):
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_results = MagicMock()
        mock_results.right_hand_landmarks = None # Simulate no hand detected

        drawn_image = draw_hand_landmarks(mock_image.copy(), mock_results)
        mock_draw_landmarks.assert_not_called()
        self.assertTrue(np.array_equal(drawn_image, mock_image)) # Image should remain unchanged


class TestConsoleUtils(unittest.TestCase):

    @patch('os.system')
    def test_clear_screen(self, mock_system):
        clear_screen()
        # On Windows, it's 'cls', on others 'clear'
        if os.name == 'nt':
            mock_system.assert_called_once_with('cls')
        else:
            mock_system.assert_called_once_with('clear')

    @patch('sys.stdout.write')
    @patch('sys.stdout.flush')
    def test_print_colored(self, mock_flush, mock_write):
        text = "Hello Test"
        color = COLOR_GREEN
        print_colored(text, color)
        mock_write.assert_called_once_with(f"\033[{color}m{text}\033[0m\n")
        mock_flush.assert_called_once()

    @patch('sys.stdout.write')
    @patch('sys.stdout.flush')
    def test_progress_bar(self, mock_flush, mock_write):
        # Test an intermediate state
        progress_bar(5, 10, bar_length=10, title="Loading")
        expected_output = "\rLoading: [=====>     ] 50% (5/10)"
        mock_write.assert_any_call(expected_output) # Use any_call because it might be called multiple times

        # Test completion
        mock_write.reset_mock()
        mock_flush.reset_mock()
        progress_bar(10, 10, bar_length=10, title="Loading")
        expected_final_output_part1 = "\rLoading: [=========>] 100% (10/10)"
        expected_final_output_part2 = "\n"
        # Check that both the progress line and the newline are written
        mock_write.assert_any_call(expected_final_output_part1)
        mock_write.assert_any_call(expected_final_output_part2)
        mock_flush.assert_called()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)