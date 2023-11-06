import configparser


class ConfigHandler:
    _config_file: str = "config.ini"
    _sections: list[str]

    def __init__(self: "ConfigHandler") -> None:
        self._config_parser = configparser.ConfigParser()
        self._config_parser.read(self._config_file)
        self._sections = self._config_parser.sections()

    def get_sections(self: "ConfigHandler") -> list[str]:
        return self._sections

    def get_config(self: "ConfigHandler", section: str, key: str) -> str:
        return self._config_parser[section].get(key)

    def get_config_boolean(self: "ConfigHandler", section: str, key: str) -> bool:
        return self._config_parser[section].getboolean(key)

    def get_config_or_alternative(self: "ConfigHandler", section: str, default_section: str, key: str) -> str:
        config = self.get_config(section, key)

        if config is None:
            config = self.get_config(default_section, key)

        return config

    def get_config_boolean_or_alternative(self: "ConfigHandler", section: str, default_section: str, key: str) -> bool:
        config = self.get_config_boolean(section, key)

        if config is None:
            config = self.get_config_boolean(default_section, key)

        return config


class ConfigTypeConverter:
    @staticmethod
    def to_int(config: str) -> int | None:
        return int(float(config)) if config != "None" else None

    @staticmethod
    def to_float(config: str) -> float | None:
        return float(config) if config != "None" else None


class MCTSPlayerConfig:
    name: str
    max_count: int | None
    max_depth: int | None
    confidence_value: float | None = 4
    rave_param: float | None
    reuse_tree: bool
    randomize_action: bool

    def __init__(self: "MCTSPlayerConfig", config_handler: ConfigHandler, name: str) -> None:
        config_section_default = "MCTSPlayer.default"
        config_section: str = f"MCTSPlayer.{name}"

        # Get configs from config_handler
        max_count = config_handler.get_config_or_alternative(config_section, config_section_default, "max_count")
        max_depth = config_handler.get_config_or_alternative(config_section, config_section_default, "max_depth")
        confidence_value = config_handler.get_config_or_alternative(
            config_section,
            config_section_default,
            "confidence_value",
        )
        rave_param = config_handler.get_config_or_alternative(config_section, config_section_default, "rave_param")
        reuse_tree = config_handler.get_config_boolean_or_alternative(
            config_section,
            config_section_default,
            "reuse_tree",
        )
        randomize_action = config_handler.get_config_boolean_or_alternative(
            config_section,
            config_section_default,
            "randomize_action",
        )

        # Convert config type and set
        self.name = name
        self.max_count = ConfigTypeConverter.to_int(max_count)
        self.max_depth = ConfigTypeConverter.to_int(max_depth)
        self.confidence_value = ConfigTypeConverter.to_float(confidence_value)
        self.rave_param = ConfigTypeConverter.to_float(rave_param)
        self.reuse_tree = reuse_tree
        self.randomize_action = randomize_action


class LoggerConfig:
    log_path: str
    log_level: str
    log_level_default: str

    def __init__(self: "LoggerConfig", config_handler: ConfigHandler, name: str) -> None:
        config_section: str = "log"

        # Get configs from config_handler
        self.log_path = config_handler.get_config(config_section, "log_path")
        self.log_level = config_handler.get_config(config_section, f"loglevel_{name.lower()}").upper()
        self.log_level_default = config_handler.get_config(config_section, "loglevel_default").upper()


if __name__ == "__main__":
    config_handler = ConfigHandler()
    sections = config_handler.get_sections()
    print(sections)
    player = MCTSPlayerConfig(config_handler, "normal")
    print(player)
    logger_config = LoggerConfig(config_handler, "playconnect4")
    print(logger_config)
