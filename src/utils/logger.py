import datetime
import logging
import os
import traceback
from logging.handlers import RotatingFileHandler


class Logger:
    _loggers = {}

    def __init__(
        self, log_file: str, level=logging.INFO, disable=False, backup_count=5, max_bytes=2 * 1024 * 1024
    ):
        """
        create a logger object

        Args:
            log_file (str): the path to the log file
            level (int): the logging level, options are logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
            disable (bool): if True, do not log anything
        """

        self.disable = disable
        self.backup_count = backup_count
        self.max_bytes = max_bytes

        # if it's a filename, save log file in log directory
        if "/" not in log_file and "\\" not in log_file:
            log_file = "logs/" + log_file
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file = log_file

        # init loggers
        self._init_loggers(level, log_file)

    def _init_loggers(self, level, name=None):
        if self.disable:
            return
        # print("_loggers: ", Logger._loggers)
        if name in Logger._loggers:
            self.logger = Logger._loggers[name]
            return
        # create a logger object
        name = name or __name__

        # debug, info logger
        logger = logging.getLogger(name)
        print("get logger with name:", logger.name)
        logger.setLevel(level)
        fh = RotatingFileHandler(
            self.log_file, maxBytes=self.max_bytes, backupCount=self.backup_count, encoding="utf-8"
        )
        fh.setLevel(level)
        # create formatter
        formatter = logging.Formatter(
            "[%(levelname)s - %(asctime)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        self.logger = logger
        Logger._loggers[name] = self.logger

        # # warning, error, critical logger
        # logger = logging.getLogger(name + "_warn")
        # logger.setLevel(level)
        # handler = RotatingFileHandler(
        #     self.log_file, maxBytes=1 * 1024 * 1024, backupCount=10, encoding="utf-8"
        # )
        # handler.setLevel(level)
        # # create formatter
        # formatter = logging.Formatter(
        #     "[%(levelname)s - %(asctime)s]: %(message)s",
        #     datefmt="%Y-%m-%d %H:%M:%S",
        # )
        # handler.setFormatter(formatter)
        # logger.addHandler(handler)
        # self.warn_logger = logger

    def _process_messages(self, messages):
        if len(messages) == 1:
            message = str(messages[0])
        else:
            messages = [str(message) for message in messages]
            message = " ".join(messages)
        return message

    # define a decorator to disable logging
    def check_logging_open_status(func):
        def wrapper(self: "Logger", *args, **kwargs):
            if self.disable:
                return
            # show_time = kwargs.pop("time", False)
            # if show_time:
            #     self.logger.critical("TIME: %s", str(datetime.datetime.now()))
            return func(self, *args, **kwargs)

        return wrapper

    @check_logging_open_status
    def log(self, *messages, **kwargs):
        """
        equivalent to info function, but log the message without formatting.
        """
        message = self._process_messages(messages)
        self.logger.info(message, **kwargs)

    @check_logging_open_status
    def debug(self, *messages, **kwargs):
        message = self._process_messages(messages)
        self.logger.debug(message, **kwargs)

    @check_logging_open_status
    def info(self, *messages, **kwargs):
        message = self._process_messages(messages)
        self.logger.info(message, **kwargs)

    @check_logging_open_status
    def warning(self, *messages, **kwargs):
        message = self._process_messages(messages)
        self.logger.warning(message, **kwargs)

    @check_logging_open_status
    def error(self, *messages, **kwargs):
        message = self._process_messages(messages)
        message += "\nTraceback:\n" + traceback.format_exc()
        self.logger.error(message, **kwargs)

    @check_logging_open_status
    def critical(self, *messages, **kwargs):
        message = self._process_messages(messages)
        self.logger.critical(message, **kwargs)


logger = Logger("global.log", disable=True)


def test_logger():
    logger = Logger("test.log")
    logger.log("test log")
    logger.debug("test debug")
    logger.info("test info")
    logger.warning("test warning")
    logger.error("test error")
    logger.critical("test critical")
    logger.log("test log with time")
    logger.info("hhhh", "test", "info")


if __name__ == "__main__":
    test_logger()
