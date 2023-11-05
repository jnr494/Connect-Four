import logging

import numba
import numpy as np
import numpy.typing as npt

import ConfigHandler
from Connect4Game import Connect4
from IPlayer import IPlayer
from LoggerHandler import LoggerHandler


# class for handling gameplay
class Connect4GameHandler:
    game: Connect4

    actions: list[int]
    player_turns: list[int]
    action_probabilities: list[float]
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
        # variables to save game states and actions
        self.actions = []
        self.states = [self.game.get_board()]
        self.player_turns = []
        self.action_probabilities = []
        self.winner = 0
        for player in self.players:
            player.reset()

    def adjust_board_state(
        self: "Connect4GameHandler",
        board_state: npt.NDArray[np.float64],
        player_turn: int,
        flip: bool,
    ) -> npt.NDArray[np.float64]:
        return adjust_board_state_numba(board_state, self.player_values[player_turn], self.game.no_cols, flip)

    def adjust_action(self: "Connect4GameHandler", action: int, flipped_bool: int) -> int:
        if flipped_bool:
            action = (action - (self.game.no_cols - 1)) * (-1)
        return action

    def play_game(self: "Connect4GameHandler") -> None:
        for round in range(self.game_size):
            current_player_turn = self.game.get_current_player_turn()

            # get available (clever) actions
            clever_available_actions = self.game.get_clever_available_actions_using_turn_handler()

            # get new action
            action = self.players[current_player_turn].make_action(self.game, clever_available_actions)
            self._logger.debug(
                f"Round {round}: player={self.game.get_current_player()} made action={action} with available actions={clever_available_actions}.",
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
            self._logger.debug(f"Game {game_number} starting.")

            self.reset_game()

            for _ in range(game_number % self.no_players):
                self.game.next_turn()

            self.play_game()

            winner = self.game.get_winner()
            self._logger.debug(f"Game  {game_number}: was won by player: {winner}.")
            winner = winner if winner is not None else 0
            winners.append(winner)

            if plot:
                self.game.plot_board_state()
        return winners

    def save_data(self: "Connect4GameHandler") -> None:
        pass


@numba.njit
def adjust_board_state_numba(
    board_state: npt.NDArray[np.float64],
    player_value: int,
    no_cols: int,
    flip: bool,
) -> npt.NDArray[np.float64]:
    board_state = board_state * player_value  # TODO improve this. adjust board so player is always 1 and opponent -1

    if flip:
        # flip board so most discs are on the left side
        half_no_cols = int(no_cols / 2)
        no_discs_left_side = np.count_nonzero(board_state[:, :half_no_cols])
        no_discs_right_side = np.count_nonzero(board_state[:, -half_no_cols:])
        if no_discs_right_side > no_discs_left_side:
            board_state = np.fliplr(board_state)
            flipped_bool = True
        else:
            flipped_bool = False
    else:
        flipped_bool = False

    return board_state, flipped_bool


if __name__ == "__main__":
    import Connect4Players
    import GameTurnHandler
    import MCTSPlayerFactory

    config_handler = ConfigHandler.ConfigHandler()
    logger_handler = LoggerHandler(config_handler)
    game_turn_handler = GameTurnHandler.GameTurnHandler([1, -1])
    game = Connect4(game_turn_handler=game_turn_handler)
    player0 = Connect4Players.RandomPlayer()
    player1 = MCTSPlayerFactory.MCTSPlayerFactory.create_player(game, -1, 1, "normal", config_handler)
    game_handler = Connect4GameHandler(game, player0, player1, logger_handler, config_handler)
    winners = game_handler.play_n_games(10)
    print(np.average(winners))
