#!/usr/bin/enc python3
"""
Unit Tests for Consol Logger
"""
import io
import os
import sys
import unittest
from unittest.mock import Mock, patch

import utils.consol_log as log


# Helper functions
def strip_stdout_buffer_stream(self, from_key):
    # Get the string after the from_key
    output = self.stdout_buffer.getvalue().strip()
    start_index = output.find(from_key)
    log_str = output[start_index:].strip()
    return log_str


# Test class
class TestConsolLog(unittest.TestCase):
    """Test class for testing core zeta functionality"""

    TestLogObj = None
    stdout_buffer = None
    print("param init")

    @classmethod
    def setUpClass(cls):
        """
        Setup the class' parameters
        Should run once at construction
        """
        print("class init")
        cls.stdout_buffer = io.StringIO()
        sys.stdout = cls.stdout_buffer
        cls.TestLogObj = log.get_logger(level=log.INFO, name="Test Logger", alt_stream=sys.stdout)

    @classmethod
    def tearDownClass(cls):
        """
        Teardown the class' parameters
        Should run once at destruction / out of scope
        """
        sys.stdout = sys.__stdout__
        assert cls.TestLogObj is not None
        cls.TestLogObj.removeHandler(cls.TestLogObj.handlers[0])
        return super().tearDownClass()

    def setUp(self):
        """
        Setup method to run before each test.
        Clears the StringIO buffer.
        """
        if self.stdout_buffer:
            self.stdout_buffer.seek(0)
            self.stdout_buffer.truncate(0)

        self._original_stdout = sys.stdout

        # Create an in-memory text buffer (StringIO)
        self._captured_output = io.StringIO()

        # Redirect all print statements to the buffer
        sys.stdout = self._captured_output

    def tearDown(self):
        """Restore original stdout after each test."""
        sys.stdout = self._original_stdout

    def test_init_logger(self):
        # Initialise a new logger and test its behaviour
        stdout_buffer = io.StringIO()
        sys.stdout = stdout_buffer
        self_mock = Mock()
        self_mock.stdout_buffer = stdout_buffer
        LogObj = log.get_logger(level=log.INFO, name="Temp Logger", alt_stream=sys.stdout)
        info_msg = "This is a info message"
        LogObj.info(info_msg)
        info_log = strip_stdout_buffer_stream(self_mock, "INFO")
        self.assertIn(info_msg, info_log)

        debug_msg = "This is a debug message"
        LogObj.debug(debug_msg)
        debug_log = strip_stdout_buffer_stream(self_mock, "DEBUG")
        self.assertNotIn(debug_msg, debug_log)  # debug level should not be logged

        msg = "This is a error message"
        LogObj.error(msg)
        error_log = strip_stdout_buffer_stream(self_mock, "ERROR")
        self.assertIn(msg, error_log)  # debug level should not be logged

        msg = "This is a warning message"
        LogObj.warning(msg)
        warn_log = strip_stdout_buffer_stream(self_mock, "WARNING")
        self.assertIn(msg, warn_log)  # debug level should not be logged

    def test_file_log(self):
        # Test that we can successfully log to a file
        file_path = os.path.abspath(os.path.join("tests", "log.txt"))
        LogObj = log.get_logger(level=log.DEBUG, name="File Logger", log_file=file_path)
        info_msg = "This is a info message"
        LogObj.info(info_msg)
        with open(file_path, "r") as f:
            log_txt = f.read()
        self.assertIn(info_msg, log_txt)

        debug_msg = "Debug written to file"
        LogObj.debug(debug_msg)
        with open(file_path, "r") as f:
            log_txt = f.read()
        self.assertIn(debug_msg, log_txt)

        # cleanup
        os.remove(file_path)

    def test_change_level_error(self):
        # Test if the level is set to error that error and warning messages
        # are logged. Assuming everthing else works
        assert self.TestLogObj is not None
        log.set_level(self.TestLogObj, log_level=log.ERROR)

        msg = "This is a warning message"
        self.TestLogObj.warning(msg)
        warn_log = strip_stdout_buffer_stream(self, "WARNING")
        self.assertNotIn(msg, warn_log)  # debug level should not be logged

        msg = "This is a error message"
        self.TestLogObj.error(msg)
        error_log = strip_stdout_buffer_stream(self, "ERROR")
        self.assertIn(msg, error_log)

    def test_change_level_debug(self):
        # Test that debug messages can be sent

        assert self.TestLogObj is not None
        log.set_level(self.TestLogObj, log_level=log.DEBUG)

        msg = "This is a debug message"
        self.TestLogObj.debug(msg)
        debug_log = strip_stdout_buffer_stream(self, "DEBUG")
        self.assertIn(msg, debug_log)

    def test_raw_print_when_level_matches(self):
        """
        Test case: Logger level equals print level (INFO <= INFO). Should print.
        """

        assert self.TestLogObj is not None
        log.set_level(self.TestLogObj, log_level=log.INFO)
        test_message = "Message at INFO level."

        self.TestLogObj.raw_print(test_message, level=log.INFO)
        raw_log = self._captured_output.getvalue()

        self.assertEqual(raw_log.strip(), test_message)

    def test_raw_print_when_logger_is_lower(self):
        """
        Test case: Logger level is lower than print level (DEBUG <= INFO). Should print.
        """

        assert self.TestLogObj is not None
        log.set_level(self.TestLogObj, log_level=log.DEBUG)
        test_message = "RAW: Message at INFO level with DEBUG logger."

        self.TestLogObj.raw_print(test_message, level=log.INFO)
        raw_log = self._captured_output.getvalue()

        self.assertEqual(raw_log.strip(), test_message)

    def test_raw_print_when_level_is_suppressed(self):
        """
        Test case: Logger level is higher than print level (WARNING > INFO). Should NOT print.
        """

        assert self.TestLogObj is not None
        log.set_level(self.TestLogObj, log_level=log.WARNING)  # Logger only shows WARNING and above
        test_message = "RAW: This message should be suppressed."

        self.TestLogObj.raw_print(test_message, level=log.INFO)  # Trying to print INFO
        raw_log = self._captured_output.getvalue()

        self.assertEqual(
            raw_log.strip(), "", "Output should be empty when print level is below logger level."
        )

    def test_raw_print_notset_boundary(self):
        """
        Test case: NOTSET (0) is the lowest level.
        If logger level is NOTSET, it should always print (unless print level is also NOTSET).
        """

        assert self.TestLogObj is not None
        log.set_level(self.TestLogObj, log_level=log.NOTSET)
        test_message_debug = "Always prints at DEBUG."
        test_message_info = "Always prints at INFO."

        self.TestLogObj.raw_print(test_message_debug, level=log.DEBUG)
        self.TestLogObj.raw_print(test_message_info, level=log.INFO)

        expected_output = f"{test_message_debug}\n{test_message_info}"
        raw_log = self._captured_output.getvalue()
        self.assertEqual(raw_log.strip(), expected_output)

    def test_raw_print_critical_levels(self):
        """
        Test case: Logger set to CRITICAL. Only CRITICAL prints should show.
        """

        assert self.TestLogObj is not None
        log.set_level(self.TestLogObj, log_level=log.CRITICAL)

        # Should be suppressed
        self.TestLogObj.raw_print("Should be hidden.", level=log.ERROR)
        raw_log = self._captured_output.getvalue()
        self.assertEqual(raw_log.strip(), "")

        # Should be visible
        test_message = "CRITICAL print visible."
        self.TestLogObj.raw_print(test_message, level=log.CRITICAL)
        raw_log = self._captured_output.getvalue()
        self.assertEqual(raw_log.strip(), test_message)

    @patch("utils.config_loader.get_config")
    def test_logger_uses_explicit_argument(self, mock_get_config):
        """Should prioritize the 'level' argument over the YAML config."""
        # Setup: YAML says INFO, but we pass DEBUG
        mock_get_config.return_value = {"logging": {"log_level": "INFO"}}

        logger = log.get_logger(level=log.DEBUG, name="test_explicit")

        self.assertEqual(logger.level, log.DEBUG)

    @patch("utils.config_loader.get_config")
    def test_logger_falls_back_to_yaml(self, mock_get_config):
        """Should pull from YAML if no level is passed to log.get_logger."""
        # Setup: YAML says ERROR, we pass None
        mock_get_config.return_value = {"logging": {"log_level": "ERROR"}}

        # Ensure level=None triggers the lookup logic
        logger = log.get_logger(level=None, name="test_yaml_fallback")

        self.assertEqual(logger.level, log.ERROR)

    @patch("utils.config_loader.get_config")
    def test_logger_defaults_on_missing_config(self, mock_get_config):
        """Should default to INFO if YAML is empty or missing the key."""
        # Setup: YAML is empty dictionary
        mock_get_config.return_value = {}

        logger = log.get_logger(level=None, name="test_default")

        self.assertEqual(logger.level, log.INFO)

    @patch("utils.config_loader.get_config")
    def test_logger_handles_lowercase_yaml(self, mock_get_config):
        """Should handle 'debug' (lowercase) from YAML correctly."""
        mock_get_config.return_value = {"logging": {"log_level": "debug"}}

        logger = log.get_logger(level=None, name="test_lowercase")

        self.assertEqual(logger.level, log.DEBUG)


if __name__ == "__main__":

    unittest.main()
