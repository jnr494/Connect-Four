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
    max_count: int
    max_depth: int
    confidence_value: float
    rave_param: float | None
    reuse_tree: bool
    randomize_action: bool
    _config_section: str
    _config_section_default: str
    _config_handler: ConfigHandler

    def __init__(self: "MCTSPlayerConfig", config_handler: ConfigHandler, name: str) -> None:
        self._config_section_default = "MCTSPlayer.default"
        self._config_section = f"MCTSPlayer.{name}"
        self._config_handler = config_handler

        self.name = name
        self._get_max_count()
        self._get_max_depth()
        self._get_confidence_value()
        self._get_rave_param()
        self._get_reuse_tree()
        self._get_randomize_action()

    def _get_max_count(self: "MCTSPlayerConfig") -> None:
        max_count = self._config_handler.get_config_or_alternative(
            self._config_section, self._config_section_default, "max_count",
        )
        max_count_int = ConfigTypeConverter.to_int(max_count)
        if max_count_int is not None:
            self.max_count = max_count_int

    def _get_max_depth(self: "MCTSPlayerConfig") -> None:
        max_depth = self._config_handler.get_config_or_alternative(
            self._config_section, self._config_section_default, "max_depth",
        )
        max_depth_int = ConfigTypeConverter.to_int(max_depth)
        if max_depth_int is not None:
            self.max_depth = max_depth_int

    def _get_confidence_value(self: "MCTSPlayerConfig") -> None:
        confidence_value = self._config_handler.get_config_or_alternative(
            self._config_section,
            self._config_section_default,
            "confidence_value",
        )
        confidence_value_float = ConfigTypeConverter.to_float(confidence_value)
        if confidence_value_float is not None:
            self.confidence_value = confidence_value_float

    def _get_rave_param(self: "MCTSPlayerConfig") -> None:
        rave_param = self._config_handler.get_config_or_alternative(
            self._config_section, self._config_section_default, "rave_param",
        )
        rave_param_float = ConfigTypeConverter.to_float(rave_param)
        self.rave_param = rave_param_float  # Okay for rav_param to be None

    def _get_reuse_tree(self: "MCTSPlayerConfig") -> None:
        reuse_tree = self._config_handler.get_config_boolean_or_alternative(
            self._config_section,
            self._config_section_default,
            "reuse_tree",
        )
        self.reuse_tree = reuse_tree

    def _get_randomize_action(self: "MCTSPlayerConfig") -> None:
        randomize_action = self._config_handler.get_config_boolean_or_alternative(
            self._config_section,
            self._config_section_default,
            "randomize_action",
        )
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
