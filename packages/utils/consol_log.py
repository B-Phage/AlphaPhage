"""
Simple console logging
Code modified from example here: https://docs.python.org/3/howto/logging.html
"""

import logging
import sys
from logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    INFO,
    NOTSET,
    WARNING,
    FileHandler,
    Logger,
    StreamHandler,
    _nameToLevel,
)
from typing import Optional, TextIO, Union

from colorlog import ColoredFormatter

# Define a flexible type for the handler to avoid MyPy errors
LogHandler = Union[FileHandler, StreamHandler]


def get_logger(
    level: int = INFO,
    name: Optional[str] = None,
    log_file: Optional[str] = None,
    alt_stream: Optional[TextIO] = None,
    reset_handler: bool = False,
):
    """
    Get a consol logger that prints to the terminal when run from shell
    The logger is formatted to display the time, log level, callee file and
    the message.

    The log level can be reset with `logger.setLevel(logging.level)` after the fact

    Inputs:
        level: the logging level (INFO, ERROR, DEBUG etc) as per the python logging

    Return: the logger object which can be used to log as per python logging

    """
    # create logger
    logging.setLoggerClass(LoggerWithPrint)  # added for printing
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create console handler and set level to debug
    if not logger.handlers or reset_handler == True:
        # Declare the type of 'ch' explicitly as the flexible Union type
        ch: LogHandler
        if log_file != None:
            ch = logging.FileHandler(log_file)
        elif alt_stream == None:
            ch = logging.StreamHandler(sys.stdout)
        else:
            ch = logging.StreamHandler(alt_stream)
        ch.setLevel(level)

        # create formatter
        formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s  | %(levelname)s | %(filename)s  | %(funcName)s   | %(name)s  | %(message)s"
        )

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

        logger.propagate = False  # prevent root logger doesn't duplicate msgs

    return logger


def set_level(logger: Logger, log_level=NOTSET):
    # Set the log level based either on the verbose level or log level
    # logging.setlevel(logger.handlers[0], log_level)
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)
    return


class LoggerWithPrint(logging.Logger):
    def __init__(self, name, level=NOTSET):
        super().__init__(name, level)

    def raw_print(self, message, level=NOTSET):
        """
        A custom logging method that prints to the consol if
        the log level is appropriate. If the level is not set
        it will always print.

        Relys on this setup:
        ERROR = 40
        WARNING = 30
        INFO = 20
        DEBUG = 10
        NOTSET = 0
        """
        if self.level <= level:
            print(message)


# End of file
