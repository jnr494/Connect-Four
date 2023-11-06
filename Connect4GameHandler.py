import logging

import ConfigHandler
from Connect4Game import Connect4
from IPlayer import IPlayer
from LoggerHandler import LoggerHandler


# class for handling gameplay
class Connect4GameHandler:
    game: Connect4
    _logger: logging.Logger
    _config_handler: ConfigHandler.ConfigHandler

    def __init__(
        self: "Connect4GameHandler",
        game: Connect4,
        player0: IPlayer,
        player1: IPlayer,
        logger_handler: LoggerHandler,
        config_handler: ConfigHandler.ConfigHandler,
    ) -> None:
        self.game = game
        self.game_size = self.game.no_rows * self.game.no_cols

        self.players = [player0, player1]
        self.no_players = len(self.players)
        self.player_values = [1, -1]

        self._logger = logger_handler.get_logger(type(self).__name__)
        self._config_handler = config_handler

        self.reset_game()

    def reset_game(self: "Connect4GameHandler") -> None:
        self.game.reset()

        for player in self.players:
            player.reset()

    def play_game(self: "Connect4GameHandler") -> None:
        for round in range(self.game_size):
            current_player_turn = self.game.get_current_player_turn()

            # get available (clever) actions
            clever_available_actions = self.game.get_clever_available_actions_using_turn_handler()

            # get new action
            action = self.players[current_player_turn].make_action(self.game, clever_available_actions)
            self._logger.debug(
                f"Round [{round}]: player=[{self.game.get_current_player()}] with name=[{self.players[self.game.get_current_player_turn()].get_name()}] made action=[{action}] with available actions={clever_available_actions}.",
            )

            # perform new action
            is_game_won = self.game.place_disc_using_turn_handler(action)

            # check if game is done
            if is_game_won:
                break

            # update player turn
            self.game.next_turn()

    def play_n_games(self: "Connect4GameHandler", no_games: int, plot: bool = False) -> list[int]:
        winners: list[int] = []

        for game_number in range(no_games):
            self._logger.debug(f"Game [{game_number}] starting.")

            self.reset_game()

            for _ in range(game_number % self.no_players):
                self.game.next_turn()

            self.play_game()

            winner = self.game.get_winner()
            self._logger.debug(f"Game  [{game_number}]: was won by player=[{winner}].")
            winner = winner if winner is not None else 0
            winners.append(winner)

            if plot:
                self.game.plot_board_state()
        return winners
