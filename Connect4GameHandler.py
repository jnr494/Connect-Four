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

    def __init__(  # noqa: PLR0913
        self: "Connect4GameHandler",
        game: Connect4,
        player0: IPlayer,
        player1: IPlayer,
        logger_handler: LoggerHandler,
        config_handler: ConfigHandler.ConfigHandler,
    ) -> None:
        self.game = game
        self.game_size = self.game.get_max_rounds()

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

    def play_game(self: "Connect4GameHandler") -> int:
        for round in range(self.game_size):
            current_player_turn = self.game.get_current_player_turn()

            # get available (clever) actions
            clever_available_actions = self.game.get_clever_available_actions()

            # get new action
            action = self.players[current_player_turn].make_action(self.game, clever_available_actions)
            self._logger.debug(
                f"Round [{round}]: player=[{self.game.get_current_player()}] with name=[{self.players[self.game.get_current_player_turn()].get_name()}] made action=[{action}] with available actions={clever_available_actions}.",  # noqa: E501
            )

            # perform new action
            is_game_won = self.game.place_disc(action)

            # check if game is done
            if is_game_won:
                break

            # update player turn
            self.game.next_turn()

        return round

    def play_n_games(self: "Connect4GameHandler", no_games: int, plot: bool = False) -> list[int]:
        self.winners: list[int] = []
        self.rounds: list[int] = []

        for game_number in range(no_games):
            self._logger.debug(f"Game [{game_number}] starting.")

            self.reset_game()

            for _ in range(game_number % self.no_players):
                self.game.next_turn()

            end_round = self.play_game()
            self.rounds.append(end_round)

            winner = self.game.get_winner()
            self._logger.debug(f"Game  [{game_number}]: was won by player=[{winner}].")
            winner = winner if winner is not None else 0
            self.winners.append(winner)

            if plot:
                self.game.plot_board_state()
        return self.winners
