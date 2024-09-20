import logging
import argparse

def get_log_level(verbosity_str):
    # Map string verbosity levels to logging levels
    level_dict = {
        'ERROR': UserInfo.ERROR,
        'WARNING': UserInfo.WARNING,
        'INFO': UserInfo.INFO,
        'DETAILED_INFO': UserInfo.DETAILED_INFO,
    }
    return level_dict.get(verbosity_str, UserInfo.INFO)


class UserInfo:
    # Define log levels
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DETAILED_INFO = logging.DEBUG

    @staticmethod
    def get_verbosity_string_types():
        return ['ERROR', 'WARNING', 'INFO', 'DETAILED_INFO']

    def __init__(self, verbosity_str: str):
        assert verbosity_str in UserInfo.get_verbosity_string_types()
        log_level = get_log_level(verbosity_str)
        # Configure the basic logger
        self.logger = logging.getLogger('UserInfo')
        self.logger.setLevel(log_level)

        # Console handler for logger
        ch = logging.StreamHandler()
        ch.setLevel(log_level)

        # Formatter
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        #self.formatter = logging.Formatter('')
        ch.setFormatter(self.formatter)

        # Add handlers to logger
        self.logger.addHandler(ch)

    def _format_multiline_message(self, message, level):
        # Split the message into lines
        lines = message.split('\n')

        # Generate a sample log record to determine the formatter output length
        sample_record = logging.LogRecord(
            name="UserInfo",
            level=level,
            pathname=__file__,
            lineno=0,
            msg="",
            args=None,
            exc_info=None
        )

        # Use the formatter to format the sample log record
        sample_message = self.formatter.format(sample_record)

        # Find the length of the formatted log without the message part
        base_length = len(sample_message) - len(sample_record.getMessage())

        # Create the indentation using this base length
        indent = ' ' * base_length

        # Join the lines with the indent
        return '\n'.join([lines[0]] + [indent + line for line in lines[1:]])

    def log(self, message, level=INFO):
        message = self._format_multiline_message(message, level)
        if level == self.ERROR:
            self.logger.error(message)
        elif level == self.WARNING:
            self.logger.warning(message)
        elif level == self.INFO:
            self.logger.info(message)
        elif level == self.DETAILED_INFO:
            self.logger.debug(message)
        else:
            self.logger.info(message)

    def set_level(self, verbosity_str: str):
        ''' It allows changing the logging level at runtime.
            It updates the level for both the logger and its handlers.
        '''
        assert verbosity_str in UserInfo.get_verbosity_string_types()
        level = get_log_level(verbosity_str)
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)