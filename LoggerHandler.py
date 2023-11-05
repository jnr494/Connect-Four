import logging

import ConfigHandler


class LoggerHandler:
    _config_handler: ConfigHandler.ConfigHandler

    def __init__(self: "LoggerHandler ", config_handler: ConfigHandler.ConfigHandler) -> None:
        self._config_handler = config_handler

    def _get_logger_config(self: "LoggerHandler", name: str) -> ConfigHandler.LoggerConfig:
        return ConfigHandler.LoggerConfig(self._config_handler, name)

    def get_logger(self: "LoggerHandler", name: str) -> logging.Logger:
        logger_config = self._get_logger_config(name)

        logger = logging.getLogger(name)
        logger.setLevel(logger_config.log_level_default)

        if logger.hasHandlers():
            return logger

        # Create handlers
        file_handler = logging.FileHandler(logger_config.log_path)
        file_handler.setLevel(logger_config.log_level)

        # Create formatters and add it to handlers
        file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_format)

        # Add handlers to the logger
        logger.addHandler(file_handler)

        return logger


if __name__ == "__main__":
    logger_handler = LoggerHandler()
    logger = logger_handler.get_playconnect4_logger()
    logger.info("info test")
    logger.debug("debug test")
