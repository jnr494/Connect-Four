import cProfile
import pstats

import ConfigHandler
import Connect4Game
import Connect4GameHandler
import GameTurnHandler
import LoggerHandler
import MCTSPlayerFactory


def run_one_game() -> None:
    config_handler = ConfigHandler.ConfigHandler()
    logger_handler = LoggerHandler.LoggerHandler(config_handler)

    game_turn_handler = GameTurnHandler.GameTurnHandler([1, -1])
    game = Connect4Game.Connect4(game_turn_handler=game_turn_handler)

    player0 = MCTSPlayerFactory.MCTSPlayerFactory.create_player(
        game,
        1,
        -1,
        "normal",
        config_handler,
        logger_handler,
    )
    player1 = MCTSPlayerFactory.MCTSPlayerFactory.create_player(game, -1, 1, "god", config_handler, logger_handler)
    game_handler = Connect4GameHandler.Connect4GameHandler(game, player0, player1, logger_handler, config_handler)
    number_of_games = 1

    game_handler.play_n_games(number_of_games)


def main() -> None:
    run_one_game()
    profiler = cProfile.Profile()
    profiler.enable()
    run_one_game()
    profiler.disable()
    profiler.dump_stats("connect4.prof")

if __name__ == "__main__":
    main()
