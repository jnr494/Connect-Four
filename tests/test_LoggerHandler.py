import logging
import unittest

import ConfigHandler
import LoggerHandler


class LoggerHandlerTests(unittest.TestCase):
    def test_get_logger(self: "LoggerHandlerTests") -> None:
        config_handler = ConfigHandler.ConfigHandler()

        logger_handler = LoggerHandler.LoggerHandler(config_handler)

        logger = logger_handler.get_logger("default")

        self.assertEqual(logger.name,"default")
        self.assertEqual(logger.level, logging.DEBUG)
