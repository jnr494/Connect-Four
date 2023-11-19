import unittest

import ConfigHandler


class ConfigHandlerTests(unittest.TestCase):
    def test_mctsplayer_config(self: "ConfigHandlerTests") -> None:
        config_handler = ConfigHandler.ConfigHandler()

        names = ["default", "normal", "hard", "god"]
        for name in names:
            mctsplayer = ConfigHandler.MCTSPlayerConfig(config_handler, name)
            self.assertIsInstance(mctsplayer.max_count, int)
            self.assertIsInstance(mctsplayer.max_depth, int)
            self.assertIsInstance(mctsplayer.confidence_value, float)
            self.assertIsInstance(mctsplayer.rave_param, (float, type(None)))
            self.assertIsInstance(mctsplayer.reuse_tree, bool)
            self.assertIsInstance(mctsplayer.randomize_action, bool)
            self.assertIsInstance(mctsplayer.rollout_weight, float)

    def test_increase_difficulty(self: "ConfigHandlerTests") -> None:
        config_handler = ConfigHandler.ConfigHandler()
        normal_player = ConfigHandler.MCTSPlayerConfig(config_handler, "normal")
        hard_player = ConfigHandler.MCTSPlayerConfig(config_handler, "hard")
        god_player = ConfigHandler.MCTSPlayerConfig(config_handler, "god")

        # compare normal and hard player
        self.assertGreater(hard_player.max_count, normal_player.max_count)
        self.assertGreater(hard_player.max_depth, normal_player.max_depth)

        # compare hard and god player
        self.assertGreater(god_player.max_count, hard_player.max_count)
        self.assertGreater(god_player.max_depth, hard_player.max_depth)

    def test_logger_config(self: "ConfigHandlerTests") -> None:
        config_handler = ConfigHandler.ConfigHandler()
        logger_config = ConfigHandler.LoggerConfig(config_handler, type(self).__name__)

        #compare logger_config to expected values
        self.assertEqual(logger_config.log_path, "logs/Connect4.log")
        self.assertEqual(logger_config.log_level, "INFORMATION")
        self.assertEqual(logger_config.log_level_default, "DEBUG")
